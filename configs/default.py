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

# Copyright 2021 The Flax Authors.
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
"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'InvalidModel'
    # model.image_size = 32
    # model.out_channels = 3
    # model.base_width = 128
    # model.n_T = 18  # inference steps
    # model.dropout = 0.13
    # model.use_aug_label = False # default: no aug
    # model.net_type = "ncsnppedm"

    # Augmentation
    config.aug = aug = ml_collections.ConfigDict()
    aug.use_edm_aug = False # default: no aug

    # Consistency training
    config.ct = ct = ml_collections.ConfigDict()
    ct.start_ema = 0.9
    ct.start_scales = 2
    ct.end_scales = 150

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.name = "imgnet_latent"
    dataset.root = "/kmh-nfs-ssd-eu-mount/data/vae_cached_muvar_imagenet"
    # dataset.cached_root = '/kmh-nfs-us-mount/data/vae_cached_muvar_imagenet'
    dataset.num_workers = 4
    dataset.prefetch_factor = 2
    dataset.pin_memory = False
    dataset.cache = False
    dataset.image_size = 256
    dataset.num_classes = 1000
    dataset.vae = "mse"

    # Eval fid
    config.fid = fid = ml_collections.ConfigDict()
    fid.num_samples = 50000
    fid.fid_per_epoch = 100
    fid.on_use = True
    fid.device_batch_size = 2
    fid.cache_ref = (
       "/kmh-nfs-ssd-eu-mount/data/cached/zhh/imgnet_256_train_jax_stats_20250205.npz"
    )
    fid.label_cond = True

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.optimizer = "adamw"
    training.learning_rate = 0.001
    training.lr_schedule = "const"  # 'cosine'/'cos', 'const'
    training.weight_decay = 0.0
    training.adam_b1 = 0.9
    training.adam_b2 = 0.999
    training.warmup_epochs = 200
    training.momentum = 0.9
    training.batch_size = 512
    training.shuffle_buffer_size = 16 * 1024
    training.prefetch = 10
    training.num_epochs = 4000
    training.log_per_step = 100
    training.log_per_epoch = -1
    training.eval_per_epoch = 100
    training.visualize_per_epoch = 100
    training.checkpoint_per_epoch = 200
    training.steps_per_eval = -1
    training.half_precision = False
    training.seed = 0  # init random seed
    training.wandb = True
    training.noise_level = 0.0
    training.label_drop_rate = 0.1
    
    # others
    # config.wandb = True
    config.load_from = ""
    config.pretrain = ""
    config.just_evaluate = False
    config.just_prior = False
    config.save_by_fid = False
    config.wandb_notes = ""
    return config


def metrics():
    return [
        "train_loss",
        "eval_loss",
        "train_accuracy",
        "eval_accuracy",
        "steps_per_second",
        "train_learning_rate",
    ]
