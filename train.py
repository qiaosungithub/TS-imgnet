# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Any

from absl import logging
from flax import jax_utils
from flax.training import train_state
import jax, optax, os, ml_collections, wandb, re
from jax import lax, random
import jax.numpy as jnp

import input_pipeline
from input_pipeline import prepare_batch_data
import models.tarflow as tarflow
from models.tarflow import TeacherStudent, generate_student, generate, edm_ema_scales_schedules, generate_prior
from utils.vae_util import LatentManager
from utils.info_util import print_params
from utils.vis_util import make_grid_visualization, float_to_uint8
from utils.ckpt_util import restore_checkpoint, restore_pretrained, save_checkpoint
import utils.fid_util as fid_util
import utils.sample_util as sample_util

from configs.load_config import sanity_check
from utils.logging_utils import log_for_0, GoodLogger
from utils.metric_utils import Timer, MyMetrics

def create_model(*, model_cls, half_precision, num_classes, **kwargs):
    """
    Create a model using the given model class.
    """
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
    """
    Initialize the model, and return the model parameters.
    """
    fake_bz = 2
    input_shape = (fake_bz, 32, 32, 4)
    label_shape = (fake_bz,)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing params...")
    variables = init(
        {"params": key, "gen": key, "dropout": key},
        jnp.ones(input_shape, model.dtype),
        jnp.ones(label_shape, jnp.int32),
    )
    if "batch_stats" not in variables:
        variables["batch_stats"] = {}
    log_for_0("Initializing params done.")
    return variables["params"], variables["batch_stats"]


def create_learning_rate_fn(
    training_config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
    """
    Create learning rate schedule.
    """
    warmup_fn = optax.linear_schedule(
        init_value=1e-6,
        end_value=training_config.learning_rate,
        transition_steps=training_config.warmup_epochs * steps_per_epoch,
    )
    if training_config.lr_schedule in ['cosine', 'cos']:
        cosine_epochs = max(training_config.num_epochs - training_config.warmup_epochs, 1)
        schedule_fn = optax.cosine_decay_schedule(
            init_value=training_config.learning_rate,
            decay_steps=cosine_epochs * steps_per_epoch,
            alpha=1e-6,
    )
    elif training_config.lr_schedule == "const":
        schedule_fn = optax.constant_schedule(value=training_config.learning_rate)
    else: raise NotImplementedError
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, schedule_fn],
        boundaries=[training_config.warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


class TrainState(train_state.TrainState):
    batch_stats: Any
    ema_params: Any


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
    """
    Create initial training state.
    """

    params, batch_stats = initialized(rng, image_size, model)

    # overwrite the ema net initialization
    ema_params = jax.tree_map(lambda x: jnp.array(x), params)
    assert batch_stats == {}  # we don't handle this in ema

    log_for_0("Info of model params:")
    log_for_0(params, logging_fn=print_params)

    if config.training.optimizer == "sgd":
        assert False
        log_for_0("Using SGD")
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.training.momentum,
            nesterov=True,
        )
    elif config.training.optimizer == "adamw":
        log_for_0(f"Using AdamW with wd {config.training.weight_decay}")
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=config.training.adam_b1,
            b2=config.training.adam_b2,
            weight_decay=config.training.weight_decay,
            # mask=mask_fn,  # TODO{km}
        )
    elif config.training.optimizer == "radam":
        assert False
        log_for_0(f"Using RAdam with wd {config.training.weight_decay}")
        assert config.training.weight_decay == 0.0
        tx = optax.radam(
            learning_rate=learning_rate_fn,
            b1=config.training.adam_b1,
            b2=config.training.adam_b2,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats={},
        ema_params=ema_params,
    )
    return state


def compute_metrics(dict_losses):
    """
    Utils function to compute metrics, used in train_step and eval_step.
    """
    metrics = dict_losses.copy()
    metrics = lax.all_gather(metrics, axis_name="batch")
    metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
    return metrics


def train_step_with_vae(state, batch, rng_init, learning_rate_fn, ema_scales_fn, noise_level, dropout_rate=0.0, latent_manager: LatentManager = None):
    """
    Perform a single training step.
    """
    # ResNet has no dropout; but maintain rng_dropout for future usage
    rng_step = random.fold_in(rng_init, state.step)
    rng_device = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    rng_noise, rng_gen, rng_dropout, rng_latent = random.split(rng_device, 4)
    
    cached = batch["image"]
    # print(f"cached: {cached}", flush=True)
    images = latent_manager.cached_encode(cached, rng_latent)
    # print(f"images: {images}", flush=True)

    ema_decay, scales = ema_scales_fn(state.step)
    
    # noise injection
    noise = jax.random.normal(rng_noise, images.shape)
    images = images + noise_level * noise
    
    rng_label, rng_dropout = random.split(rng_dropout, 2)
    
    if dropout_rate > 0.0:
        mask = jax.random.bernoulli(rng_label, dropout_rate, batch["label"].shape)
        label = jnp.where(mask, -jnp.ones_like(batch["label"]), batch["label"])
        # print(f"label: {label}", flush=True)
    else:
        label = batch["label"]
    
    def loss_fn(wrt):
        """loss function used for training."""
        outputs, new_model_state = state.apply_fn(
            {"params": wrt, "batch_stats": state.batch_stats},
            x=images,
            y=label,
            mutable=["batch_stats"],
            rngs=dict(gen=rng_gen, dropout=rng_dropout),
        )
        loss, dict_losses, imgs = outputs

        return loss, (new_model_state, dict_losses, imgs)

    # compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    new_model_state, dict_losses, ret_images = aux[1]
    grads = lax.pmean(grads, axis_name="batch")

    # apply gradients
    state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )

    # update ema
    ema_params = jax.tree_map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, state.ema_params, state.params)
    state = state.replace(ema_params=ema_params)

    # for simplicity, we don't all gather images
    lr = learning_rate_fn(state.step)
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr
    metrics["ema_decay"] = ema_decay
    # return state, metrics, ret_images
    return state, metrics


def sample_step(params, sample_idx, model, rng_init, device_batch_size, noise_level, guidance, temperature=1.0, label_cond=True, student=True, just_prior=False):
    """
    Generate samples from the train state.

    sample_idx: each random sampled image corrresponds to a seed
    """
    rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx

    if just_prior: # for debug, first teacher x then student z
        x, z = generate_prior(params, model, rng_sample, n_sample=device_batch_size, guidance=guidance, noise_level=noise_level, temperature=temperature, label_cond=label_cond)
        z_all = lax.all_gather(z, axis_name="batch")  # each device has a copy
        z_all = z_all.reshape(-1, *z_all.shape[2:])
        x_all = lax.all_gather(x, axis_name="batch")
        x_all = x_all.reshape(-1, *x_all.shape[2:])
        return z_all
    
    generate_fn = generate_student if student else generate
    images = generate_fn(params, model, rng_sample, n_sample=device_batch_size, guidance=guidance, noise_level=noise_level, temperature=temperature, label_cond=label_cond)
    images = images.transpose((0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
    assert images.shape == (device_batch_size, 4, 32, 32)
    images_all = lax.all_gather(images, axis_name="batch")  # each device has a copy
    images_all = images_all.reshape(-1, *images_all.shape[2:])
    return images_all


@partial(jax.pmap, axis_name="x")
def cross_replica_mean(x):
    """
    Compute an average of a variable across workers.
    """
    return lax.pmean(x, "x")

def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    if state.batch_stats == {} or all(not v for v in state.batch_stats.values()):
        return state
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def run_p_sample_step(p_sample_step, state, sample_idx, latent_manager, ema=False):
    # redefine the interface
    images = p_sample_step(
        params={"params": state.params if not ema else state.ema_params, "batch_stats": {}},
        sample_idx=sample_idx,
    )
    images = images[0]
    assert images.shape[1:] == (4, 32, 32), f"images.shape: {images.shape}"
    samples = latent_manager.decode(images)
    assert samples.shape[1:] == (3, 256, 256), f"samples.shape: {samples.shape}"
    assert not jnp.any(
        jnp.isnan(samples)
    ), f"There is nan in decoded samples! latent range: {images.min()}, {images.max()}. Whether there is nan in latents: {jnp.any(jnp.isnan(images))}"
    samples = samples.transpose((0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
    
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return samples  # images have been all gathered

def get_fid_evaluater(config, p_sample_step, logger, latent_manager):
    inception_net = fid_util.build_jax_inception()
    stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)
    run_p_sample_step_inner = partial(run_p_sample_step, latent_manager=latent_manager)

    def eval_fid(state, ema_only=False, use_teacher=False):
        step = int(jax.device_get(state.step)[0])
        
        # NOTE: logging for FID should be done inside this function
        if not ema_only:
            samples_all = sample_util.generate_samples_for_fid_eval(
                state, config, p_sample_step, run_p_sample_step_inner, ema=False
            )
            num = samples_all.shape[0]
            kk = num // 1000
            mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
            fid_score = fid_util.compute_fid(
                mu, stats_ref["mu"], sigma, stats_ref["sigma"]
            )
            log_for_0(f"w/o ema: FID at {num} samples: {fid_score}")
            logger.log_dict(step+1, {f"FID-{kk}k": fid_score})

        samples_all = sample_util.generate_samples_for_fid_eval(
            state, config, p_sample_step, run_p_sample_step_inner, ema=True
        )
        num = samples_all.shape[0]
        kk = num // 1000
        mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
        fid_score_ema = fid_util.compute_fid(
            mu, stats_ref["mu"], sigma, stats_ref["sigma"]
        )
        T = 'teacher_' if use_teacher else ''
        log_for_0(f"{T} w/ ema: FID at {num} samples: {fid_score_ema}")
        logger.log_dict(step+1, {f"{T}FID-{kk}k_ema": fid_score_ema})

        vis = make_grid_visualization(samples_all, to_uint8=False)
        vis = jax.device_get(vis)
        vis = vis[0]
        logger.log_image(step+1, {"gen_fid": vis})
        
        return fid_score_ema
    
    return eval_fid

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    ######################################################################
    #                       Initialize training                          #
    ######################################################################
    sanity_check(config)
    log_for_0(config)
    rank = index = jax.process_index()
    training_config = config.training
    # update config root
    if "cached" in config.dataset.root and "kmh-nfs-us-mount" in workdir:
        config.dataset.root = config.dataset.root.replace(
            "kmh-nfs-ssd-eu-mount", "kmh-nfs-us-mount"
        )

    assert (
        (workdir + config.dataset.root).count("kmh-nfs-ssd-eu-mount")
        * (workdir + config.dataset.root).count("kmh-nfs-us-mount")
    ) % 4 == 0, "root path should be consistent with workdir"

    if config.training.wandb and rank == 0:
        wandb.init(project="TS_imgnet" if not config.just_evaluate else "TS_imgnet_eval", dir=workdir, tags=['sqa'], name=config.wandb_name)
        wandb.config.update(config.to_dict())
        try:
            ka = re.search(
                r"kmh-tpuvm-v[234]-(\d+)(-preemptible)?-(\d+)", workdir
            ).group()
        except AttributeError:
            ka = re.search(
                r"kmh-tpuvm-v4-32-preemptible-yiyang(\d*)", workdir
            ).group()
        wandb.config.update({"ka": ka})
        wandb.run.notes = config.wandb_notes
    logger = GoodLogger(use_wandb=config.training.wandb, workdir=workdir)

    rng = random.key(config.training.seed)
    
    global_batch_size = training_config.batch_size
    log_for_0("config.batch_size: {}".format(global_batch_size))
    if global_batch_size % jax.process_count() > 0:
        raise ValueError("Batch size must be divisible by the number of processes")
    local_batch_size = global_batch_size // jax.process_count()
    log_for_0("local_batch_size: {}".format(local_batch_size))
    log_for_0("jax.local_device_count: {}".format(jax.local_device_count()))

    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError("Local batch size must be divisible by the number of local devices")

    ######################################################################
    #                           Create Dataloaders                       #
    ######################################################################
    # config.dataset.use_flip = not config.aug.use_edm_aug
    if not config.just_evaluate:
        train_loader, steps_per_epoch = input_pipeline.create_split(
            config.dataset,
            local_batch_size,
            split="train",
        )
        # eval_loader, steps_per_eval = input_pipeline.create_split(
        #   config.dataset,
        #   local_batch_size,
        #   split='val',
        # )

        log_for_0("steps_per_epoch: {}".format(steps_per_epoch))
        # log_for_0('steps_per_eval: {}'.format(steps_per_eval))

        base_learning_rate = training_config.learning_rate

        learning_rate_fn = create_learning_rate_fn(
            training_config, base_learning_rate, steps_per_epoch
        )
    else: learning_rate_fn = lambda x: 1.0 # for create train state

    ######################################################################
    #                       Create Train State                           #
    ######################################################################

    model: TeacherStudent = create_model(
        model_cls = getattr(tarflow, config.model.name),
        half_precision=config.training.half_precision,
        num_classes=config.dataset.num_classes,
        **config.model,
    )

    state = create_train_state(rng, config, model, config.dataset.image_size, learning_rate_fn)

    if config.load_from != "":
        assert os.path.exists(config.load_from), "checkpoint not found. You should check GS bucket"
        log_for_0("Restoring from: {}".format(config.load_from))
        state = restore_checkpoint(state, config.load_from)
    elif config.pretrain != "":
        raise NotImplementedError("Note that this 'pretrained' have a different meaning, see how `restore_pretrained` is implemented")
        assert os.path.exists(config.pretrain), "pretrained model not found. You should check GS bucket"
        log_for_0("Loading pre-trained from: {}".format(config.pretrain))
        
        # before load
        # example_element = jax.tree_leaves(state.params["blocks_0"])[0].reshape(-1)[0]
        # logging.info(f'before load example_element at layer 0: {example_element}')
        
        state = restore_pretrained(state, config.pretrain, config)
        
        # after load
        # example_element = jax.tree_leaves(state.params["blocks_0"])[0].reshape(-1)[0]
        # logging.info(f'after load example_element at layer 0: {example_element}')

    ########################## load teacher #########################
    if config.load_teacher_from != "" and not config.just_evaluate:
        teacher_model = create_model(
            model_cls = getattr(tarflow, config.teacher.name),
            half_precision=config.training.half_precision,
            num_classes=config.dataset.num_classes,
            **config.teacher,
        )
        teacher_state = create_train_state(rng, config, teacher_model, config.dataset.image_size, learning_rate_fn)
        log_for_0("Restoring from: {}".format(config.load_teacher_from))
        teacher_state = restore_checkpoint(teacher_state, config.load_teacher_from)
        state.params['teacher'] = teacher_state.ema_params
        state.batch_stats['teacher'] = teacher_state.batch_stats
        state.ema_params['teacher'] = teacher_state.ema_params
        
        del teacher_model, teacher_state

    if not config.just_evaluate:
        # step_offset > 0 if restarting from checkpoint
        step_offset = int(state.step)
        epoch_offset = step_offset // steps_per_epoch  # sanity check for resuming
        assert epoch_offset * steps_per_epoch == step_offset, "step_offset must be divisible by steps_per_epoch, but get {} vs {}".format(step_offset, steps_per_epoch)

    state = jax_utils.replicate(state)

    latent_manager = LatentManager(config.dataset.vae, config.fid.device_batch_size)
    

    ######################################################################
    #                     Prepare for Training Loop                      #
    ######################################################################
    assert config.training.ema_schedule in ['edm', 'const']
    ema_scales_fn = partial(
        edm_ema_scales_schedules, steps_per_epoch=steps_per_epoch, config=config
    ) if config.training.ema_schedule == 'edm' else lambda x: (config.training.ema, 0)
    p_train_step = jax.pmap(
        partial(
            train_step_with_vae,
            rng_init=rng,
            learning_rate_fn=learning_rate_fn,
            ema_scales_fn=ema_scales_fn,
            noise_level=config.training.noise_level,
            dropout_rate=config.training.label_drop_rate,
            latent_manager=latent_manager,
        ),
        axis_name="batch",
    )
    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=jax.random.PRNGKey(0),
            device_batch_size=config.fid.device_batch_size,
            guidance=config.fid.guidance,
            noise_level=config.training.noise_level,
            temperature=config.fid.temperature,
            label_cond=config.fid.label_cond,
            student=True,
            just_prior=config.just_prior,
        ),
        axis_name="batch",
    )

    vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())  # for visualization
    logging.info(f"fixed_sample_idx: {vis_sample_idx}")

    # compile p_sample_step
    logging.info("Compiling p_sample_step...")
    timer = Timer()
    lowered = p_sample_step.lower(
        params={"params": state.params, "batch_stats": {}},
        sample_idx=vis_sample_idx,
    )
    p_sample_step = lowered.compile()
    logging.info("p_sample_step compiled in {}s".format(timer.elapse_with_reset()))
    
    ######################### handling just_prior ####################
    if config.just_prior:
        # just prior sampling
        log_for_0("Sampling for prior ...")
        vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, ema=True)
        print(f"vis shape: {vis.shape}")
        vis = make_grid_visualization(vis)
        logger.log_image(1, {"vis_sample": vis[0]})
        
        samples_all = sample_util.generate_samples_for_prior_eval(
            state, config, p_sample_step, run_p_sample_step, ema=True
        )
        print(f"vis shape: {samples_all.shape}")
        prior = 0.5 * (samples_all ** 2).mean()
        logger.log_dict(1, {"prior": prior})
        return state

    # prepare for FID evaluation
    if config.fid.on_use:
        fid_evaluator = get_fid_evaluater(config, p_sample_step, logger, latent_manager)
        # handle just_evaluate
        if config.just_evaluate:
            log_for_0("Sampling for images ...")
            vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager, ema=False)
            vis = jax.device_get(vis)
            vis = float_to_uint8(vis)
            for i in range(7):
                logger.log_image(1, {f"vis_sample_{i}": vis[i]})
            # vis = make_grid_visualization(vis)
            # logger.log_image(1, {"vis_sample": vis[0]})
            fid_score_ema = fid_evaluator(state, ema_only=True)
            return state

    ##################### handling teacher sanity #######################
    p_sample_step_teacher = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=rng,
            device_batch_size=config.fid.device_batch_size,
            guidance=config.fid.guidance,
            noise_level=config.training.noise_level,
            temperature=config.fid.temperature,
            label_cond=config.fid.label_cond,
            student=False,
            just_prior=False,
        ),
        axis_name="batch",
    )
    # compile p_sample_step_teacher
    logging.info("Compiling p_sample_step_teacher...")
    timer = Timer()
    lowered = p_sample_step_teacher.lower(
        params={"params": state.params, "batch_stats": {}},
        sample_idx=vis_sample_idx,
    )
    p_sample_step_teacher = lowered.compile()
    logging.info("p_sample_step_teacher compiled in {}s".format(timer.elapse_with_reset()))

    if config.fid.sanity_teacher:
        # eval teacher fid and log
        log_for_0("evaluating teacher fid...")
        fid_score_teacher = fid_evaluator(state, p_sample_step_teacher, ema_only=True, use_teacher=True)
        assert fid_score_teacher < 10, 'bad teacher fid score {}'.format(fid_score_teacher) # NOTE: This is bad as expected, since the teacher doesn't denoise

    log_for_0("Initial compilation, this might take some minutes...")

    ######################################################################
    #                           Training Loop                            #
    ######################################################################
    timer.reset()
    min_fid = float("inf")
    for epoch in range(epoch_offset, training_config.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))
        
        step = epoch * steps_per_epoch
        
        if (
            epoch == 0
            or (epoch + 1) % training_config.eval_per_epoch == 0
        ):
            log_for_0("Sample epoch {}...".format(epoch))
            vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager, ema=False)
            vis = jax.device_get(vis)
            vis = float_to_uint8(vis)
            # vis = make_grid_visualization(vis)
            # logger.log_image(step + 1, {"vis_sample": vis[0]})
            for i in range(7):
                logger.log_image(step + 1, {f"vis_sample_{i}": vis[i]})
            vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager, ema=True)
            vis = jax.device_get(vis)
            vis = float_to_uint8(vis)
            for i in range(7):
                logger.log_image(step + 1, {f"vis_sample_ema_{i}": vis[i]})
        
        # training
        train_metrics = MyMetrics(reduction="last")
        for n_batch, batch in enumerate(train_loader):
            batch = prepare_batch_data(batch)
            state, metrics = p_train_step(state, batch)
            train_metrics.update(metrics)

            if epoch == epoch_offset and n_batch == 0:
                logging.info(
                    "p_train_step compiled in {}s".format(timer.elapse_with_reset())
                )
                logging.info("Initial compilation completed. Reset timer.")

            step = epoch * steps_per_epoch + n_batch
            ep = epoch + n_batch / steps_per_epoch
            if training_config.get("log_per_step"):
                if (step + 1) % training_config.log_per_step == 0:
                    # compute and log metrics
                    summary = train_metrics.compute_and_reset()
                    summary["steps_per_second"] = training_config.log_per_step / timer.elapse_with_reset()
                    summary.update({"ep": ep, "step": step})
                    logger.log_dict(step + 1, summary)

        # # Show training visualization
        # if (epoch + 1) % training_config.visualize_per_epoch == 0:
        #     vis = visualize_cifar_batch(vis)
        #     logger.log_image(step + 1, {"vis_train": vis[0]})

        # Show samples (eval)
        # if (
        #     epoch == 0
        #     or (epoch + 1) % training_config.eval_per_epoch == 0
        # ):
        #     log_for_0("Sample epoch {}...".format(epoch))
        #     vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager, ema=False)
        #     vis = jax.device_get(vis)
        #     vis = float_to_uint8(vis)
        #     # vis = make_grid_visualization(vis)
        #     # logger.log_image(step + 1, {"vis_sample": vis[0]})
        #     for i in range(7):
        #         logger.log_image(step + 1, {f"vis_sample_{i}": vis[i]})
        #     vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, latent_manager, ema=True)
        #     vis = jax.device_get(vis)
        #     vis = float_to_uint8(vis)
        #     for i in range(7):
        #         logger.log_image(step + 1, {f"vis_sample_ema_{i}": vis[i]})


        # save checkpoint
        if not config.save_by_fid:
            if (
                (epoch + 1) % training_config.checkpoint_per_epoch == 0
                or epoch == training_config.num_epochs
            ):
                state = sync_batch_stats(state)
                save_checkpoint(state, workdir)

        if config.fid.on_use and (
            (epoch + 1) % config.fid.fid_per_epoch == 0
            or (epoch + 1) == training_config.num_epochs
        ):
            fid_score_ema = fid_evaluator(state)
            if fid_score_ema < min_fid:
                log_for_0(f"!!! Best FID changed from {min_fid} to {fid_score_ema}")
                min_fid = fid_score_ema
                if config.save_by_fid:
                    state = sync_batch_stats(state)
                    save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
