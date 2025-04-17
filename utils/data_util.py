from absl import logging
import jax, os
import jax.numpy as jnp
from optax._src.alias import *
import numpy as np
import input_pipeline
from input_pipeline import prepare_batch_data

from diffusers.models import FlaxAutoencoderKL  # , AutoencoderKL
from functools import partial
import torch
from torchvision import transforms

NUM_CLASSES = 1000


class LatentDist(
    object
):  # https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/models/vae_flax.py#L689
    """
    Class of Gaussian distribution.

    Method:
        sample: Sample from the distribution.
    """

    def __init__(self, parameters, deterministic=False):
        """
        parameters: concatenated mean and std
        """
        # Last axis to account for channels-last
        self.mean, self.std = jnp.split(parameters, 2, axis=-1)
        self.deterministic = deterministic
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(self.mean)

    def sample(self, key):
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)


def create_cached_dataset(config, local_batch_size):
    train_loader, steps_per_epoch = input_pipeline.create_split(
        config.dataset,
        local_batch_size,
        split="train",
        use_flip=False,
        # split='val',
    )
    eval_loader, steps_per_eval = input_pipeline.create_split(
        config.dataset, local_batch_size, split="val", use_flip=False
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        f"pcuenq/sd-vae-ft-{config.vae}-flax"
    )

    os.mkdir(config.dataset.cached_root)
    os.mkdir(config.dataset.cached_root + "/train")
    os.mkdir(config.dataset.cached_root + "/val")
    logging.info("Start dataset cache")
    for epoch in range(1):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        logging.info("epoch {}...".format(epoch))

        def func(m, p, b):
            latent = m.apply(
                {"params": p},
                jnp.transpose(b, (0, 3, 1, 2)),
                method=FlaxAutoencoderKL.encode,
            ).latent_dist
            return jnp.concatenate((latent.mean, latent.std), axis=-1)

        p_train_step = jax.pmap(
            partial(func, m=vae, p=vae_params),
            axis_name="batch",
        )
        for n_batch, batch in enumerate(train_loader):
            step = epoch * steps_per_epoch + n_batch
            batch_flip = (transforms.functional.hflip(batch[0]), batch[1])
            batch = prepare_batch_data(batch)
            batch_flip = prepare_batch_data(batch_flip)
            # from IPython import embed; embed()
            # batch = {'image': batch['image'].reshape(-1, batch['image'].shape[2], batch['image'].shape[3], batch['image'].shape[4]), 'label': batch['label'].reshape(-1)}
            # batch_flip = {'image': batch_flip['image'].reshape(-1, batch_flip['image'].shape[2], batch_flip['image'].shape[3], batch_flip['image'].shape[4]), 'label': batch_flip['label'].reshape(-1)}
            inputs = p_train_step(b=batch["image"]).reshape(-1, 32, 32, 8)
            inputs_flip = p_train_step(b=batch_flip["image"]).reshape(-1, 32, 32, 8)
            labels = batch["label"].reshape(
                -1,
            )
            for i in range(inputs.shape[0]):
                torch.save(
                    {
                        "noflip": torch.tensor(np.array(inputs[i])),
                        "flipped": torch.tensor(np.array(inputs_flip[i])),
                        "label": torch.tensor(np.array(labels[i])),
                    },
                    os.path.join(
                        config.dataset.cached_root,
                        "train",
                        f"{(n_batch*local_batch_size+i):08d}" + ".pt",
                    ),
                )
            if n_batch % 100 == 0:
                logging.info("Progress: " + str(n_batch) + " batches")
        for n_batch, batch in enumerate(eval_loader):
            step = epoch * steps_per_epoch + n_batch
            batch_flip = (transforms.functional.hflip(batch[0]), batch[1])
            batch = prepare_batch_data(batch)
            batch_flip = prepare_batch_data(batch_flip)
            inputs = p_train_step(b=batch["image"]).reshape(-1, 32, 32, 8)
            inputs_flip = p_train_step(b=batch_flip["image"]).reshape(-1, 32, 32, 8)
            labels = batch["label"].reshape(
                -1,
            )
            for i in range(inputs.shape[0]):
                torch.save(
                    {
                        "noflip": torch.tensor(np.array(inputs[i])),
                        "flipped": torch.tensor(np.array(inputs_flip[i])),
                        "label": torch.tensor(np.array(labels[i])),
                    },
                    os.path.join(
                        config.dataset.cached_root,
                        "val",
                        f"{(n_batch*local_batch_size+i):08d}" + ".pt",
                    ),
                )
            if n_batch % 100 == 0:
                logging.info("Progress: " + str(n_batch) + " batches")