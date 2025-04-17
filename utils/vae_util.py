# ZHH: we abstract all VAE utils here
import torch, jax, os

# import torch.nn as nn
# import torch.nn.functional as F
import torch.utils.data

from diffusers.models import FlaxAutoencoderKL

from utils.data_util import LatentDist
from flax import jax_utils

import jax.numpy as jnp
import numpy as np

from functools import partial
from utils.logging_utils import log_for_0


class LatentManager:

    def __init__(self, vae_type, decode_batch_size):
        # init VAE
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            f"pcuenq/sd-vae-ft-{vae_type}-flax"
        )
        self.vae = vae
        self.vae_params = vae_params

        self.batch_size = decode_batch_size
        self.latent_size = 32  # only support 32 for now

        # create decode function
        self.decode_fn = self.get_decode_fn()

    def get_decode_fn(self):

        def dist_prepare_batch_data(batch):
            # reshape (host_batch_size, 3, height, width) to
            # (local_devices, device_batch_size, height, width, 3)
            local_device_count = jax.local_device_count()

            return_dict = {}
            for k, v in batch.items():
                v = v.reshape((local_device_count, -1) + v.shape[1:])
                return_dict[k] = v
            return return_dict

        log_for_0("Compiling vae.apply...")

        z_dummy = jnp.ones(
            (
                jax.local_device_count(),
                self.batch_size * jax.process_count(),
                4,
                self.latent_size,
                self.latent_size,
            )
        )
        # p_batch = dist_prepare_batch_data(dict(z=z_dummy))

        p_vae_variable = jax_utils.replicate({"params": self.vae_params})
        p_decod_func = partial(self.vae.apply, method=FlaxAutoencoderKL.decode)
        p_decod_func = jax.pmap(p_decod_func, axis_name="batch")

        lowered = p_decod_func.lower(p_vae_variable, z_dummy)
        compiled_decod_func = lowered.compile()
        Bflops = compiled_decod_func.cost_analysis()[0]["flops"] / 1e9
        log_for_0("Compiling VAE decoder done")
        log_for_0(f"FLOPs (1e9): {Bflops}")

        def call_p_compiled_model_func(x, p_func, var):
            x = dist_prepare_batch_data(dict(x=x))["x"]
            x = p_func(var, x)
            x = x.sample
            x = x.reshape((-1,) + x.shape[2:])
            return dict(sample=x)

        call_compiled_decod_func = partial(
            call_p_compiled_model_func, p_func=compiled_decod_func, var=p_vae_variable
        )

        return call_compiled_decod_func

    def encode(self, images, rng):
        return (
            self.vae.apply(
                {"params": self.vae_params}, images, method=FlaxAutoencoderKL.encode
            ).latent_dist.sample(key=rng)
            * 0.18215
        )

    def cached_encode(self, cached_value, rng):
        # cached_value: contain mean and std
        return LatentDist(cached_value).sample(key=rng) * 0.18215

    def decode(self, latents):
        return self.decode_fn(latents / 0.18215)["sample"]


class LatentDataset(torch.utils.data.Dataset):

    def __init__(self, root, use_flip=False):
        self.root = root
        self.file_list = [file for file in os.listdir(root) if file.endswith(".pt")]
        self.use_flip = use_flip

    def __len__(self):
        return len(self.file_list)

    def __repr__(self) -> str:
        # copied from some random pytorch code
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body.append(f"Use flip: {self.use_flip}")
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.file_list[idx])
        data = torch.load(file_path)
        if torch.rand(1).item() >= 0.5 and self.use_flip:
            return (data["flipped"].permute(2, 0, 1), data["label"])
        else:
            return (data["noflip"].permute(2, 0, 1), data["label"])


if __name__ == "__main__":
    val_dataset = LatentDataset(
        "/kmh-nfs-ssd-eu-mount/data/vae_cached_muvar_imagenet/val"
    )
    print(val_dataset, flush=True)
    x, y = val_dataset[0]
    print(x.shape, flush=True)
    print(y, flush=True)

    mnger = LatentManager("mse", 1)

    # assert False
    from input_pipeline import prepare_batch_data

    ret_dic = prepare_batch_data((x.unsqueeze(0), y.unsqueeze(0)), 1)
    x = ret_dic["image"][0]
    print("cached value:", x.shape, flush=True)  # [1, 32, 32, 8]
    x = mnger.cached_encode(x, jax.random.PRNGKey(0))
    print("latent:", x.shape, flush=True)  # [1, 32, 32, 4]
    img = mnger.decode(x.transpose(0, 3, 1, 2))
    print("img:", img.shape, flush=True)  # [1, 3, 256, 256]
    img = img[0].transpose(1, 2, 0)
    img = jnp.clip((img + 1) / 2, 0, 1)
    # save imagez
    from PIL import Image

    img = jax.device_get(img)
    print("img range: [{}, {}]".format(img.min(), img.max()), flush=True)
    img = Image.fromarray((np.array(img) * 255).astype("uint8"))
    img.save("test.png")
    log_for_0("Image saved")