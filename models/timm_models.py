# PatchEmbed = Attention = Mlp = NotADirectoryError

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from math import sqrt

bomb = lambda: exec("raise ValueError")


class PatchEmbed(nn.Module):
    input_size: int
    initial_patch_size: int
    in_channels: int
    hidden_size: int
    bias: bool = True

    def setup(self):
        self.patch_size = (self.initial_patch_size, self.initial_patch_size)
        self.img_size = self.input_size
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            self.img_size
        )

        self.flatten = True
        self.proj = nn.Conv(
            self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(in_axis=(0, 1, 2), out_axis=-1),
            bias_init=nn.initializers.zeros,
        )
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: int):
        assert self.patch_size
        img_size = (img_size, img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def __call__(self, x):
        B, H, W, C = x.shape  # (2, 32, 32, 4)
        assert C < 7, "likely you miss the transpose, get x.shape = {}".format(x.shape)
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), "input size does not match, input size is {}, but self.shape is {}".format(
            (H, W), self.img_size
        )
        x = self.proj(x)  # (B, H/p, W/p, hidden_c)
        x = x.reshape(B, -1, x.shape[3])  # NHWC -> NLC
        # x = self.norm(x) # no norm layer
        return x


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    norm_layer: nn.Module = bomb
    linear_layer: nn.Module = bomb

    def setup(self):
        num_heads = self.num_heads
        dim = self.dim
        qkv_bias = self.qkv_bias
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = self.linear_layer(dim, dim * 3, bias=qkv_bias)
        self.q_norm = lambda x: x
        self.k_norm = lambda x: x
        self.attn_drop = lambda x: x
        self.proj = self.linear_layer(dim, dim, bias=True)
        self.proj_drop = lambda x: x

    def __call__(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )  # 3, B, num_heads, N, head_dim
        # q, k, v = qkv.unbind(0)
        q, k, v = jnp.split(qkv, 3, axis=0)  # k.shape: (1, 2, 6, 64, 64)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(0, 1, 2, 4, 3)  # 1, B, num_heads, i, j
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (
            attn @ v
        )  # (1, 2, 6, 64, 64) ; (1, B, num_heads, i, j) @ (1, B, num_heads, j, head_dim)

        x = x[0].transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    in_features: int
    hidden_features: int
    act_layer: nn.Module = bomb
    drop: float = 0.0
    linear_layer: nn.Module = bomb

    def setup(self):
        in_features = self.in_features
        hidden_features = self.hidden_features
        act_layer = self.act_layer
        drop = self.drop
        out_features = in_features
        hidden_features = hidden_features or in_features
        bias = (True, True)
        drop_probs = (drop, drop)
        assert drop < 1e-3, NotImplementedError()
        linear_layer = self.linear_layer

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = lambda x: x
        self.norm = lambda x: x
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = lambda x: x

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
