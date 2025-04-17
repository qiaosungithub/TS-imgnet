import jax
import logging as sys_logging
from absl import logging
import wandb
import numpy as np
import os
from PIL import Image

def log_for_0(*args, logging_fn=logging.info, additional_judge=True, **kwargs):
    if jax.process_index() == 0 and additional_judge:
        logging_fn(*args, **kwargs)


class ExcludeInfo(sys_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        # print('zhh ijijijijiji',record.pathname)
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > sys_logging.INFO
        return True


exclude_files = [
    "orbax/checkpoint/async_checkpointer.py",
    "orbax/checkpoint/multihost/utils.py",
    "orbax/checkpoint/future.py",
    "orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py",
    "orbax/checkpoint/type_handlers.py",
    "orbax/checkpoint/metadata/checkpoint.py",
]
file_filter = ExcludeInfo(exclude_files)


def supress_checkpt_info():
    logging.get_absl_handler().addFilter(file_filter)


class GoodLogger:

    def __init__(self, workdir, use_wandb=False):
        if jax.process_index() != 0:
            return
        self.use_wandb = use_wandb
        self.workdir = workdir
        if use_wandb:
            if wandb.run is None:
                raise RuntimeError(
                    "Failed to initialize wandb. Please check your wandb login. Also make sure to initialize the logger after creating the wandb run."
                )
        else:
            assert os.path.exists(workdir), f"workdir {workdir} does not exist"
            os.makedirs(os.path.join(workdir, 'zhh_images'), exist_ok=True)


    def log_dict(self, step, dict_obj):
        # [200] ep=0.159073, steps_per_second=6.76798, train_accuracy=0.00585938, train_loss=6.71379, train_lr=0.0127258, train_step=199
        if jax.process_index() != 0:
            return
        log_str = f"[{step}]"
        for k, v in dict_obj.items():
            log_str += f" {k}={v:.5f}," if isinstance(v, float) else f" {k}={v},"
        log_str = log_str.strip(",")
        logging.info(log_str)
        if self.use_wandb:
            wandb.log(dict_obj, step=step)

    def log_image(self, step, image_dict):
        if jax.process_index() != 0:
            return

        def reduce_arr_func(v):
            assert isinstance(v, np.ndarray), "Invalid image type {}".format(type(v))
            assert v.dtype == np.uint8, "Invalid image dtype {}".format(v.dtype)
            assert (
                v.ndim == 3
                and 3 in [v.shape[0], v.shape[2]]
            ), "Invalid image shape {}".format(v.shape)
            if v.shape[0] == 3:
                v = v.transpose((1, 2, 0))
            return v

        if self.use_wandb:
            wandb.log({
                k: wandb.Image(reduce_arr_func(v)) for k, v in image_dict.items()
            }, step=step)
        else:
            log_for_0(f"Saving images locally, at step {step}")
            for k, v in image_dict.items():
                v = Image.fromarray(reduce_arr_func(v))
                v.save(os.path.join(self.workdir, 'zhh_images', f"step{step:06d}_{k}.png"))

    # destructor
    def __del__(self):
        if jax.process_index() != 0:
            return
        if self.use_wandb:
            wandb.finish()
