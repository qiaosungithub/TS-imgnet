from absl import logging
import torchvision
import torch
import numpy as np
import jax.numpy as jnp
import os
import jax

from utils.vis_util import float_to_uint8
from utils.logging_utils import log_for_0

def get_samples_from_dir(samples_dir, config):
    # e.g.: samples_dir = '/kmh-nfs-us-mount/logs/kaiminghe/results-edm/edm-cifar10-32x32-uncond-vp'
    ds = torchvision.datasets.ImageFolder(
        samples_dir, transform=torchvision.transforms.PILToTensor()
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=100, shuffle=False, drop_last=False, num_workers=12
    )
    samples_all = []
    for x in dl:
        samples_all.append(x[0].numpy().transpose(0, 2, 3, 1))
    samples_all = np.concatenate(samples_all)
    samples_all = samples_all[: config.fid.num_samples]
    return samples_all


def generate_samples_for_fid_eval(
    state, config, p_sample_step, run_p_sample_step, ema=True
):
    num_steps = np.ceil(
        config.fid.num_samples / config.fid.device_batch_size / jax.device_count()
    ).astype(int)
    log_for_0(f"expected sampling steps: {num_steps}")
    samples_all = []
    valid_count = 0
    nan_count = 0
    step = 0

    while valid_count < config.fid.num_samples:

        sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(
            jax.local_device_count()
        )
        sample_idx = jax.device_count() * step + sample_idx
        
        logging.info(f"Sampling step {step}...")
        samples = run_p_sample_step(
            p_sample_step, state, sample_idx=sample_idx, ema=ema
        )

        # 检查 NaN 并过滤
        nan_mask = jnp.isnan(samples).any(axis=(1, 2, 3)) # 检查每张图片是否有 NaN
        valid_samples = samples[~nan_mask] # 过滤掉含有 NaN 的图片
        nan_count += nan_mask.sum() # 记录 NaN 图片的数量
        valid_count += valid_samples.shape[0]

        valid_samples = float_to_uint8(valid_samples)

        samples_all.append(valid_samples)
        step += 1

        assert nan_count < valid_count, f"over half of image generated has nan!"

    samples_all = np.concatenate(samples_all, axis=0)
    samples_all = samples_all[: config.fid.num_samples]

    log_for_0(f"Total NaN images: {nan_count}")
    log_for_0(f"nan ratio: {nan_count / config.fid.num_samples}")
    return samples_all
