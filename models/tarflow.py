from absl import logging
from typing import Any
# from collections.abc import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp
# import numpy as np

from functools import partial

from utils.logging_utils import log_for_0

ModuleDef = Any
print = lambda *args, **kwargs : None

def get_map_fn(map_method, teacher_nblocks, self_nblocks):
    map_fn = None
    log_for_0(f"Use map method: {map_method}, teacher is {teacher_nblocks} blocks, student is {self_nblocks} blocks")
    if map_method == "skip":
        # method: skip
        assert teacher_nblocks % self_nblocks == 0, f"{teacher_nblocks} % {self_nblocks} != 0"
        ratio = teacher_nblocks // self_nblocks
        map_fn = lambda idx: idx * ratio
    elif map_method == "extreme":
        assert self_nblocks % 2 == 0, f"self_nblocks must be even, but got {self_nblocks}"
        half_layers = self_nblocks // 2
        map_fn = lambda idx: (
            idx if idx < half_layers else teacher_nblocks - self_nblocks + idx
        )
    elif map_method == "extreme2":
        assert self_nblocks % 2 == 0, f"self_nblocks must be even, but got {self_nblocks}"
        assert teacher_nblocks >= 2 * self_nblocks, f"teacher_nblocks must be larger than 2 * self_nblocks, but got {teacher_nblocks} and {self_nblocks}"
        half_layers = self_nblocks // 2
        map_fn = lambda idx: (
            idx * 2 if idx < half_layers else teacher_nblocks + 1 - self_nblocks * 2 + idx * 2
        )
    elif map_method == "last":
        map_fn = lambda idx: teacher_nblocks - self_nblocks + idx
    else:
        raise ValueError(f"Unknown load_pretrain_method: {map_method}")
    return map_fn

def edm_ema_scales_schedules(step, config, steps_per_epoch):
    # ema_halflife_kimg = 500  # from edm
    # ema_halflife_kimg = 50000  # from flow
    ema_halflife_kimg = config.training.get("ema_halflife_kimg", 50000)
    ema_halflife_nimg = ema_halflife_kimg * 1000

    ema_rampup_ratio = 0.05
    ema_halflife_nimg = jnp.minimum(
        ema_halflife_nimg, step * config.training.batch_size * ema_rampup_ratio
    )

    ema_beta = 0.5 ** (config.training.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
    scales = jnp.ones((1,), dtype=jnp.int32)
    return ema_beta, scales

# Pytorch-like initialization
torch_linear = partial(nn.Dense, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)

def safe_split(rng):
    if rng is None:
        return None, None
    return jax.random.split(rng)

class PermutationFlip:
    """Permutation and flip for images."""

    # flip: bool = True
    def __init__(self, flip: bool = True):
        self.flip = flip

    # @nn.compact
    def __call__(self, x: jnp.ndarray, inverse: bool = False):
        # x = jnp.take(x, self.permutation, axis=-1)
        if self.flip:
            x = jnp.flip(x, axis=1)
        return x
    

class Attention(nn.Module):
    """Attention mechanism."""

    in_channels: int
    num_heads: int
    dropout: float = 0.0
    dtype: Any = jnp.float32
    
    def setup(self):
        self.head_dim = self.in_channels // self.num_heads
        assert self.head_dim * self.num_heads == self.in_channels, "in_channels must be divisible by num_heads"

        # self.scale = self.head_dim ** -0.5 # useless

        self.qkv = torch_linear(features=3 * self.in_channels, dtype=self.dtype)
        self.proj = torch_linear(features=self.in_channels, dtype=self.dtype)
        self.norm = nn.LayerNorm(dtype=self.dtype)
        self.projdrop = nn.Dropout(rate=self.dropout)
        
        # self.k_cache = self.variable('cache', 'k_cache', lambda: {'cond': [], 'uncond': []})
        # self.v_cache = self.variable('cache', 'v_cache', lambda: {'cond': [], 'uncond': []})
        
    def __call__(self, 
                 x: jnp.ndarray, 
                 mask, 
                 k_cache: dict | None = None, 
                 v_cache: dict | None = None, 
                 temp: float = 1.0, 
                 which_cache: str = 'cond', 
                 train: bool = True,
                 rng = None):
        # x: [B, T, C]
        # mask: [B, T, T]
        # temp: scalar
        # which_cache: str
        # train: bool
        
        # print('\t\tattention get: ', k_cache and {k: len(v) and f'{len(v)}x{v[0].shape}' for k, v in k_cache.items()})
        
        B, T, C = x.shape
        x = self.norm(x.astype(self.dtype))
        qkv = self.qkv(x) # [B, T, 3 * C]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim) / temp
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)
        # [B, T, H, D]
        
        if not train:
            # k_cache[which_cache].append(k)
            # v_cache[which_cache].append(v)
            # k = jnp.concatenate(k_cache[which_cache], axis=1)
            # v = jnp.concatenate(v_cache[which_cache], axis=1)
            k_idx = k_cache["idx"]
            v_idx = v_cache["idx"]
            k_cache[which_cache] = jax.lax.dynamic_update_slice(k_cache[which_cache], k, (0, k_idx, 0, 0))
            v_cache[which_cache] = jax.lax.dynamic_update_slice(v_cache[which_cache], v, (0, v_idx, 0, 0))
            k = k_cache[which_cache]
            v = v_cache[which_cache]
            # k = jax.lax.dynamic_slice(k_cache[which_cache], (0, 0, 0, 0), (B, k_idx, self.num_heads, self.head_dim))
            # v = jax.lax.dynamic_slice(v_cache[which_cache], (0, 0, 0, 0), (B, v_idx, self.num_heads, self.head_dim))
            # print('k and v shape:', k.shape, v.shape)
            assert mask is not None, 'this KV cache impl. is different and require attn mask'
            # print('mask shape:', mask.shape)
            # assert k.shape[1] == T and v.shape[1] == T, f'k shape: {k.shape}, v shape: {v.shape}, T: {T}'
        else:
            assert k_cache is None and v_cache is None, "k_cache and v_cache must be None during training"
        
        # append
        # print('T:', T)
        # q = q[:, :T]
        # k = k[:, :T]
        # v = v[:, :T]
        # print('qkv shape:', q.shape, k.shape, v.shape)
        rng, rng_used = safe_split(rng)
        attn = nn.dot_product_attention(query=q, key=k, value=v, mask=mask, dropout_rng=rng_used, deterministic=not train, precision=None, dropout_rate=self.dropout)
        del rng_used
        attn = attn.reshape(B, T, C)
        attn = self.proj(attn) # [B, T, C]
        rng, rng_used = safe_split(rng)
        attn = self.projdrop(attn, deterministic=not train, rng=rng_used)
        del rng_used
        return attn, k_cache, v_cache

class MLP(nn.Module):
    """MLP."""

    # in_channels: int
    out_channels: int
    dropout: float = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        self.norm = nn.LayerNorm(dtype=self.dtype)
        # self.net = nn.Sequential(
        #     [torch_linear(features=4*self.out_channels, dtype=self.dtype),
        #     partial(nn.gelu, approximate=False),
        #     torch_linear(features=self.out_channels, dtype=self.dtype)]
        # )
        self.net1 = torch_linear(features=4*self.out_channels, dtype=self.dtype)
        self.net2 = torch_linear(features=self.out_channels, dtype=self.dtype)
        self.drop = nn.Dropout(rate=self.dropout)

    def __call__(self, x: jnp.ndarray, train: bool = True, rng = None):
        # print("mlp:", x.shape, self.out_channels)
        x = self.norm(x.astype(self.dtype))
        # return self.net(x)
        x = self.net1(x)
        rng, rng_used = safe_split(rng)
        x = self.drop(x, deterministic=not train, rng=rng_used)
        del rng_used
        x = jax.nn.gelu(x, approximate=False)
        x = self.net2(x)
        rng, rng_used = safe_split(rng)
        x = self.drop(x, deterministic=not train, rng=rng_used)
        del rng_used
        return x
    
class AttentionBlock(nn.Module):
    """Attention block."""

    num_heads: int
    channels: int
    dropout: float = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        self.attn = Attention(in_channels=self.channels, num_heads=self.num_heads, dtype=self.dtype, dropout=self.dropout)
        self.mlp = MLP(out_channels=self.channels, dtype=self.dtype, dropout=self.dropout)

    def __call__(self, 
                 x: jnp.ndarray, 
                 k_cache: dict | None = None,
                 v_cache: dict | None = None,
                 mask: jnp.ndarray | None = None,
                 temp: float = 1.0, 
                 which_cache: str = 'cond', 
                 train: bool = True,
                 rng = None):
        rng, rng_used = safe_split(rng)
        attn, k_cache, v_cache = self.attn(x, mask, k_cache=k_cache, v_cache=v_cache, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
        del rng_used
        x = x + attn
        rng, rng_used = safe_split(rng)
        x = x + self.mlp(x, train=train, rng=rng_used)
        del rng_used
        return x, k_cache, v_cache

class MetaBlock(nn.Module):
    """Meta block."""

    in_channels: int
    channels: int
    num_patches: int
    num_layers: int
    num_heads: int
    num_classes: int
    permutation: PermutationFlip
    debug: bool = False
    dropout: float = 0.0
    dtype: Any = jnp.float32
    
    def setup(self):
        self.proj_in = torch_linear(features=self.channels, dtype=self.dtype)
        self.pos_emebdding = self.param('pos_embedding', 
                                        nn.initializers.normal(stddev=0.02), 
                                        (1, self.num_patches, self.channels))
        
        self.class_embedding = None
        if self.num_classes > 0:
            self.class_embedding = self.param('class_embedding',
                                    nn.initializers.normal(stddev=0.02),
                                    (self.num_classes, 1, self.channels))
        
        self.blocks = [AttentionBlock(num_heads=self.num_heads, channels=self.channels, dtype=self.dtype, dropout=self.dropout) for _ in range(self.num_layers)]
        
        self.proj_out = nn.Dense(features=2*self.in_channels, dtype=self.dtype, 
            kernel_init=nn.initializers.zeros if not self.debug else nn.initializers.normal(stddev=0.01),
        bias_init=nn.initializers.zeros) 
        self.attn_mask = lambda: jnp.tril(jnp.ones((self.num_patches, self.num_patches), dtype=jnp.bool))
        
    def forward(self, 
                x: jnp.ndarray, 
                y: jnp.ndarray | None = None,
                temp: float = 1.0, 
                which_cache: str = 'cond', 
                train: bool = True,
                rng = None):
        """
        Args:
            x: [B, T, C]
            y: [B]
            temp: scalar
            which_cache: str
            train: bool
        """
        # print(x.shape, y.shape)
        x_in = self.permutation(x)
        x = self.proj_in(x_in)
        x = x + self.permutation(self.pos_emebdding)
        # x = self.permutation(x) # [B, T, C]
        
        if self.class_embedding is not None:
            if y is not None:
                # print(y.shape, self.class_embedding.shape)
                # if (y < 0).any():
                #     trivial_class = self.class_embedding.mean(axis=1)
                #     mask = (y < 0).astype(jnp.float32)
                #     class_embed =  (1 - mask) * self.class_embedding[y] + mask * trivial_class
                # else:
                mask = (y < 0).astype(jnp.float32).reshape(-1, 1, 1)
                # print(mask.shape, self.class_embedding[y].shape, self.class_embedding.mean(axis=0).shape)
                class_embed = (1 - mask) * self.class_embedding[y] + mask * self.class_embedding.mean(axis=0)
                # class_embed = self.class_embedding[y]
                # print(x.shape, class_embed.shape)
                x = x + class_embed
            else:
                x = x + self.class_embedding.mean(axis=0)
        
        for block in self.blocks:
            rng, rng_used = safe_split(rng)
            # print('\tblock')
            x, _, _ = block(x, mask=self.attn_mask(), temp=temp, which_cache=which_cache, train=train, rng=rng_used)
            del rng_used
        
        x = self.proj_out(x) # [B, T, 2*C]
        x = jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        alpha, mu = jnp.split(x, 2, axis=-1)
        x_new = (x_in - mu) * jnp.exp(-alpha) # [B, T, C_in]
        x_new = self.permutation(x_new, inverse=True)
        return x_new, - alpha.mean(axis=(1, 2))

    def reverse_step(
        self, 
        x: jnp.ndarray, 
        i: int,
        y: jnp.ndarray | None = None, 
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        temp: float = 1.0, 
        which_cache: str = 'cond', 
        train: bool = False):
        # NOTE: train=True is not supported
        """
        Args:
            x: [B, T, C]
            y: [B]
            temp: scalar
            which_cache: str
            train: bool
        """
        B, T, C = x.shape
        # x_in = x[:, i:i+1]
        x_in = jax.lax.dynamic_slice(x, (0, i, 0), (x.shape[0], 1, x.shape[2]))
        assert x_in.shape == (B, 1, C)
        x = self.proj_in(x_in)
        pos_embed = self.permutation(self.pos_emebdding)
        assert pos_embed.shape == (1, T, x.shape[-1]), f'{pos_embed.shape}, {(1, T, x.shape[-1])}'
        x = x + jax.lax.dynamic_slice(pos_embed, (0, i, 0), (pos_embed.shape[0], 1, pos_embed.shape[2]))
        
        if self.num_classes > 0:
            if y is not None:
                class_embed = self.class_embedding[y]
            else:
                class_embed = self.class_embedding.mean(axis=0)
            x = x + class_embed
        
        for bi, block in enumerate(self.blocks):
            # print('\tblock')
            mask = self.attn_mask()
            mask = jax.lax.dynamic_slice(mask, (i, 0), (1, mask.shape[1]))
            # print('mask:', mask)
            x, k_cache[bi], v_cache[bi] = block(x, k_cache=k_cache[bi], v_cache=v_cache[bi], temp=temp, which_cache=which_cache, train=train, mask=mask)
        
        x = self.proj_out(x) # [B, 1, 2*C]
        mu, alpha = jnp.split(x, 2, axis=-1)
        return mu, alpha, k_cache, v_cache # [B, C_in]
    
    def __call__(self, 
            x: jnp.ndarray, 
            y: jnp.ndarray | None = None,
            temp: float = 1.0, 
            which_cache: str = 'cond', 
            train: bool = True,
            rng = None):
        return self.forward(x, y, temp, which_cache, train, rng)
    
def reverse_block(
    params,
    block: MetaBlock, 
    x: jnp.ndarray, 
    y: jnp.ndarray | None = None,
    temp: float = 1.0, 
    which_cache: str = 'cond', 
    guidance: float = 0,
    train: bool = False):
    # NOTE: train=True is not supported
    """
    Args:
        x: [B, T, C]
        y: [B]
        temp: scalar
        which_cache: str
        train: bool
    """
    # x_in = x
    x = block.permutation(x)
    # x = self.proj_in(x)
    # x = x + self.pos_emebdding
    # self.clear_cache()
    B, T, C = x.shape
    num_heads = block.num_heads
    head_dim = block.channels // num_heads
    k_cache, v_cache = [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), "idx": 0} for _ in range(block.num_layers)], [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), "idx": 0} for _ in range(block.num_layers)]
    
    # B, T, C = x.shape
    assert T == block.num_patches
    # TODO: to for-i-loop
    
    # for i in range(T-1):
    #     # print('step:', i)
    #     alpha_i, mu_i, k_cache, v_cache = self.reverse_step(x, i, y, k_cache, v_cache, temp, which_cache="cond", train=train)
    #     # TODO: impl. guidance here.
        
    #     # print("mu and alpha mean: ", mu_i.mean(), alpha_i.mean())
    #     val = x[:, i+1] * jnp.exp(alpha_i[:, 0].astype(jnp.float32)) + mu_i[:, 0]
    #     B, T, C = x.shape
    #     assert val.shape == (B, C) 
    #     # print('val range:', val.min(), val.max())
    #     x = x.at[:, i+1].set(val)
    #     # print('x mean:', x.mean())
    
    def step_fn(i, vals):
        x_step, k_cache_step, v_cache_step = vals
        for obj in k_cache_step + v_cache_step:
            obj["idx"] = i
        # print('i:', i) # tracedarray
        # alpha_i, mu_i, k_cache_step, v_cache_step = block.reverse_step(x_step, i, y, k_cache_step, v_cache_step, temp, which_cache="cond", train=train)
        alpha_i_cond, mu_i_cond, k_cache_step, v_cache_step = block.apply(params, x_step, i, y, k_cache_step, v_cache_step, temp, "cond", train, method=block.reverse_step)
        
        alpha_i_uncond, mu_i_uncond, k_cache_step, v_cache_step = block.apply(params, x_step, i, None, k_cache_step, v_cache_step, temp, "uncond", train, method=block.reverse_step)
        
        alpha_i = (1 + guidance * i / (T - 1)) * alpha_i_cond - guidance * i / (T - 1) * alpha_i_uncond
        mu_i = (1 + guidance * i / (T - 1)) * mu_i_cond - guidance * i / (T - 1) * mu_i_uncond
        
        x_sliced = jax.lax.dynamic_slice(x_step, (0, i+1, 0), (x_step.shape[0], 1, x_step.shape[2]))
        assert x_sliced.shape == (B, 1, C)
        val = x_sliced[:, 0] * jnp.exp(alpha_i[:, 0].astype(jnp.float32)) + mu_i[:, 0]
        # x_step = x_step.at[:, i+1].set(val)
        assert val.shape == (B, C)
        x_step = jax.lax.dynamic_update_slice(x_step, val.reshape(B, 1, C), (0, i+1, 0))
        return x_step, k_cache_step, v_cache_step
        
        # val = x_step[:, i+1] * jnp.exp(alpha_i[:, 0].astype(jnp.float32)) + mu_i[:, 0]
        # B, T, C = x_step.shape
        # assert val.shape == (B, C)
        # x_step = x_step.at[:, i+1].set(val)
        # return x_step, k_cache_step, v_cache_step
    
    x, _, _ = jax.lax.fori_loop(0, T-1, step_fn, (x, k_cache, v_cache))
    # for i in range(T-1):
    #     x, k_cache, v_cache = step_fn(i, (x, k_cache, v_cache))
        # print('k_cache:',k_cache[0]["cond"])
    
    return block.permutation(x, inverse=True)

class NormalizingFlow(nn.Module):
    """Normalizing flow."""
    
    img_size: int
    out_channels: int
    channels: int
    patch_size: int
    num_layers: int
    num_heads: int
    num_blocks: int
    # permutation: PermutationFlip = PermutationFlip()
    num_classes: int = 0
    teacher_nblocks: int = None
    load_pretrain_method: str = "skip"
    dtype: Any = jnp.float32
    dropout: float = 0.0
    debug: bool = False
    prior_norm: float = 1.0
    
    def setup(self):
        patch_num = self.img_size // self.patch_size
        self.num_patches = patch_num ** 2
        self.in_channels = self.out_channels * self.patch_size * self.patch_size
        assert self.patch_size * patch_num == self.img_size, "img_size must be divisible by num_patches"
        
        judge = lambda idx: idx % 2 == 1
        
        if self.teacher_nblocks is not None:
            log_for_0(f"teacher_nblocks: {self.teacher_nblocks}")
            map_fn = get_map_fn(self.load_pretrain_method, self.teacher_nblocks, self.num_blocks)
            judge = lambda idx: map_fn(idx) % 2 == 1
        
        log_for_0(f"judge result: {[judge(i) for i in range(self.num_blocks)]}")
        self.blocks = [
            MetaBlock(
                in_channels=self.in_channels, 
                channels=self.channels, 
                num_patches=self.num_patches, 
                num_layers=self.num_layers, 
                num_heads=self.num_heads, 
                num_classes=self.num_classes, 
                # permutation=self.permutation
                permutation=PermutationFlip(judge(i)),
                dropout=self.dropout,
                debug=self.debug,
            ) for i in range(self.num_blocks)
        ]
        
        self.prior_shape = (self.num_patches, self.in_channels)
        
    def patchify(self, x):
        B, H, W, C = x.shape
        assert C == self.out_channels, "input must be RGB image"
        x = x.reshape(B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.num_patches, self.patch_size * self.patch_size * C)
        return x
    
    def unpatchify(self, x):
        B, T, C = x.shape
        # assert self.patch_size * self.patch_num == self.img_size, "img_size must be divisible by num_patches"
        x = x.reshape(B, self.img_size // self.patch_size, self.img_size // self.patch_size, self.patch_size, self.patch_size, self.out_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.img_size, self.img_size, self.out_channels)
        return x
        
    def __call__(self, 
                x: jnp.ndarray, 
                y: jnp.ndarray | None = None, 
                temp: float = 1.0, 
                which_cache: str = 'cond', 
                train: bool = True,
        ):
        """
        Args:
            x: [B, H, W, C]
            y: [B]
            temp: scalar
            which_cache: str
            train: bool
        """
        rng = self.make_rng('dropout')
        # total_x = [x]
        x = self.patchify(x)  # [B, T, C]

        tot_logdet = 0.0
        for block in self.blocks:
            rng, rng_used = safe_split(rng)
            x, logdet = block.forward(x, y, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
            del rng_used
            # total_x.append(self.unpatchify(x))
            # print("x and alpha shape: ", x.shape, alpha.shape)
            tot_logdet = tot_logdet + logdet
            # print("alpha:", jnp.mean(alpha, axis=(1, 2)))
        
        log_prior = 0.5 * (x ** 2).mean() * (self.prior_norm ** -2.0)
        
        # TODO: impl. update prior
        
        loss = - tot_logdet.mean() + log_prior
        # x = self.unpatchify(x)
        
        # img = jnp.stack(total_x, axis=1)
        
        loss_dict = {'loss': loss, 'log_det': tot_logdet.mean(), 'log_prior': log_prior}
        
        return loss, loss_dict, x
    
def reverse(params,
            nf: NormalizingFlow,
            x: jnp.ndarray,
            y: jnp.ndarray | None = None,
            temp: float = 1.0,
            guidance: float = 0,
            which_cache: str = 'cond',
            train: bool = False):
    # x = self.patchify(x)
    # for block in nf.blocks[::-1]:
    print('param keys:', params['params'].keys())
    # for block_param in params['params']['blocks'][::-1]:
    patch_num = nf.img_size // nf.patch_size
    num_patches = patch_num ** 2
    in_channels = nf.out_channels * nf.patch_size * nf.patch_size

    judge = lambda idx: idx % 2 == 1
    
    if nf.teacher_nblocks is not None:
        log_for_0(f"teacher_nblocks: {nf.teacher_nblocks}")
        map_fn = get_map_fn(nf.load_pretrain_method, nf.teacher_nblocks, nf.num_blocks)
        judge = lambda idx: map_fn(idx) % 2 == 1
    
    for i in range(nf.num_blocks-1,-1,-1):
        block_param = params['params'][f'blocks_{i}']
        # x = block.reverse(x, y, temp=temp, which_cache=which_cache, train=train)
        x = reverse_block({"params": block_param}, MetaBlock(
                in_channels=in_channels, 
                channels=nf.channels, 
                num_patches=num_patches, 
                num_layers=nf.num_layers, 
                num_heads=nf.num_heads, 
                num_classes=nf.num_classes, 
                # permutation=PermutationFlip(i % 2 == 1),
                permutation=PermutationFlip(judge(i)),
                debug=nf.debug,
            ), x, y, temp=temp, which_cache=which_cache, train=train, guidance=guidance)
        # print("mean during layer:", x.mean())
    x = nf.unpatchify(x)
    return x

# # move this out from model for JAX compilation
def generate(params, model: NormalizingFlow, rng, n_sample, noise_level, guidance, temperature=1.0, label_cond=True):
    """
    Generate samples from the model
    """
    patch_num = model.img_size // model.patch_size
    num_patches = patch_num ** 2
    in_channels = model.out_channels * model.patch_size * model.patch_size
    x_shape = (n_sample, num_patches, in_channels)
    rng_used, rng = jax.random.split(rng, 2)
    rng_used_2, rng = jax.random.split(rng, 2)
    rng_used_3, rng = jax.random.split(rng, 2)
    z = jax.random.normal(rng_used, x_shape, dtype=model.dtype) * model.prior_norm
    if label_cond:
        y = jax.random.randint(rng_used_2, (n_sample,), 0, model.num_classes)
    else:
        y = None
    
    # x = model.apply(params, z, y, method=model.reverse)
    if label_cond:
        x = reverse(params, model, z, y, guidance=guidance, temp=1.0)
    else:
        x = reverse(params, model, z, y, guidance=guidance, temp=temperature)
    
    if noise_level == 0: return x
    def nabla_log_prob(x):
        loss, _, _ = model.apply(params, x, y, rngs={"dropout": rng_used_3}) # no train
        return loss
    
    nabla_log_prob = jax.grad(nabla_log_prob)
    grad_dir = nabla_log_prob(x) * jnp.prod(jnp.array(x.shape))
    
    x = x - (noise_level ** 2) * grad_dir # use score-based model to denoise
    
    return x

NF_Debug = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=4, patch_size=2, num_layers=1, num_heads=1, num_blocks=1, debug=True,
)

NF_Base = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=768, patch_size=4, num_layers=4, num_heads=12, num_blocks=12,
)

NF_Small = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=4, num_layers=4, num_heads=6, num_blocks=12,
)

NF_Small_p2 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=4, num_heads=6, num_blocks=12,
) # 91M

NF_Small_p2_b6 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=4, num_heads=6, num_blocks=6,
)

NF_Small_p2_b2 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=4, num_heads=6, num_blocks=2,
)

# sqa add
NF_Small_p2_b8_l8 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=8, num_heads=6, num_blocks=8,
) # 118M

# for b4l8, 59M. DiT-B is 130M.

NF_Small_p2_b16_l4 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=4, num_heads=6, num_blocks=16,
) # 121M

NF_Small_p2_b4_l16 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=2, num_layers=16, num_heads=6, num_blocks=4,
) # 115M

NF_Small_p4_b8_l8 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=4, num_layers=8, num_heads=6, num_blocks=8,
) # 117M

NF_2x_p2_b4_l4 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=768, patch_size=2, num_layers=4, num_heads=12, num_blocks=4,
) # 117M

NF_2x_p2_b8_l4 = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=768, patch_size=2, num_layers=4, num_heads=12, num_blocks=8,
) # 234M

NF_Default = partial(
    NormalizingFlow, img_size=32, out_channels=4, channels=384, patch_size=4, num_layers=8, num_heads=6, num_blocks=8,
)

if __name__== "__main__":
    print = __builtins__.print
    jax.config.update("jax_default_matmul_precision", "float32")
    # jax.config.update("jax_disable_jit", True)
    model = NormalizingFlow(
        img_size=6,
        out_channels=3,
        channels=4,
        patch_size=2,
        num_layers=4,
        num_heads=2,
        num_blocks=4,
        num_classes=10,
        debug=True,
    )
    # model = NormalizingFlow(
    #     img_size=6,
    #     out_channels=3,
    #     channels=4,
    #     patch_size=2,
    #     num_layers=4,
    #     num_heads=2,
    #     num_blocks=4,
    #     num_classes=10,
    #     debug=True,
    # )
    model = NormalizingFlow(
        img_size=6,
        out_channels=3,
        channels=4,
        patch_size=2,
        num_layers=1,
        num_heads=2,
        num_blocks=1,
        num_classes=10,
        debug=True,
    )
    variables = model.init(jax.random.PRNGKey(0), jnp.ones((2, 6, 6, 3),dtype=jnp.float32), jnp.ones((2,),dtype=jnp.int32))
    x = jax.random.normal(jax.random.PRNGKey(0), (7, 6, 6, 3))
    print(x.mean(axis=(1, 2, 3)))
    # y = np.random.randint(0, 10, (7,))
    y = jax.random.randint(jax.random.PRNGKey(0), (7,), 0, 10)
    loss, _, z = model.apply(variables, x, y, rngs={"dropout": jax.random.PRNGKey(0)})
    print('latent shape:', z.shape) # should be: (7, 9, 12)
    print('is z nan?', jnp.isnan(z).any())
    back = reverse(variables, model, z, y)
    loss, _, backagain = model.apply(variables, back, y, rngs={"dropout": jax.random.PRNGKey(0)})
    # print(back.mean(axis=(1, 2, 3)))
    print('back.shape', back.shape) # should be: (7, 6, 6, 3)
    print('check x==back', ((x-back) ** 2).mean(axis=(1, 2, 3)))
    print('check z==backagain', ((z-backagain) ** 2).mean(axis=(1, 2)))
    
    print(jnp.allclose(x, back, atol=1e-5))

    # test generate
    x_gen = generate(variables, model, jax.random.PRNGKey(0), 7)
    print(x_gen.shape) # should be: (7, 6, 6, 3)