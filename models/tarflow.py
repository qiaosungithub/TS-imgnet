from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from utils.logging_utils import log_for_0

def safe_split(rng):
    if rng is None:
        return None, None
    else:
        return jax.random.split(rng)

ModuleDef = Any
print = lambda *args, **kwargs : None

### TEACHER STUDENT MODE ###
# T and S are in the same order (T fixed); apply L2 loss (on z) on each block separately.
SAME_ORDER_L2_EACH_BLOCK = 0 
# T and S are in the same order (T fixed); apply L2 loss (on mu and alpha) on each block separately.
SAME_ORDER_L2_MA_EACH_BLOCK = 1 
# T and S are in reverse order (T fixed); apply L2 loss (on z) on each block separately.
REV_ORDER_L2_EACH_BLOCK = 2
# T and S are in reverse order (T fixed); apply L2 loss (on mu and alpha) on each block separately.
REV_ORDER_L2_MA_EACH_BLOCK = 3
############################

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

class PermutationFlip:
    """Permutation and flip for images."""

    def __init__(self, flip: bool = True):
        self.flip = flip

    def __call__(self, x: jnp.ndarray, inverse: bool = False):
        if self.flip:
            x = jnp.flip(x, axis=1)
        return x

# random permutation is not supported for now. Please refer to lyy's NF github repo

class Attention(nn.Module):
    """Attention mechanism."""

    in_channels: int
    num_heads: int
    dropout: float = 0.0
    dtype: Any = jnp.float32
    
    def setup(self):
        self.head_dim = self.in_channels // self.num_heads
        assert self.head_dim * self.num_heads == self.in_channels, "in_channels must be divisible by num_heads"

        self.qkv = torch_linear(features=3 * self.in_channels, dtype=self.dtype)
        self.proj = torch_linear(features=self.in_channels, dtype=self.dtype)
        self.norm = nn.LayerNorm(dtype=self.dtype)
        self.projdrop = nn.Dropout(rate=self.dropout)
        
    def __call__(self, 
                 x: jnp.ndarray, 
                 mask, 
                 k_cache: dict | None = None, 
                 v_cache: dict | None = None, 
                 temp: float = 1.0, 
                 which_cache: str = 'cond', 
                 train: bool = True,
                 rng = None):
        
        B, T, C = x.shape
        x = self.norm(x.astype(self.dtype))
        qkv = self.qkv(x) # [B, T, 3 * C]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim) / temp
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)
        # [B, T, H, D]
        
        if not train and k_cache is not None and v_cache is not None:
            k_idx = k_cache["idx"]
            v_idx = v_cache["idx"]
            k_cache[which_cache] = jax.lax.dynamic_update_slice(k_cache[which_cache], k, (0, k_idx, 0, 0))
            v_cache[which_cache] = jax.lax.dynamic_update_slice(v_cache[which_cache], v, (0, v_idx, 0, 0))
            k = k_cache[which_cache]
            v = v_cache[which_cache]
            assert mask is not None, 'this KV cache impl. is different and require attn mask'
        else:
            assert k_cache is None and v_cache is None, "k_cache and v_cache must be None during training"
  
        rng, rng_used = safe_split(rng)
        
        attn = nn.dot_product_attention(query=q, key=k, value=v, mask=mask, dropout_rng=rng_used, deterministic=not train, precision=None, dropout_rate=self.dropout, broadcast_dropout=False, dtype=self.dtype)
        del rng_used
        attn = attn.reshape(B, T, C)
        attn = self.proj(attn) # [B, T, C]
        rng, rng_used = safe_split(rng)
        attn = self.projdrop(attn, rng=rng_used, deterministic=not train)
        return attn, k_cache, v_cache

class MLP(nn.Module):
    """MLP."""

    # in_channels: int
    out_channels: int
    dropout: float = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        self.norm = nn.LayerNorm(dtype=self.dtype)
        self.net1 = torch_linear(features=4*self.out_channels, dtype=self.dtype)
        self.net2 = torch_linear(features=self.out_channels, dtype=self.dtype)
        self.drop = nn.Dropout(rate=self.dropout)

    def __call__(self, x: jnp.ndarray, train: bool = True, rng = None):
        # print("mlp:", x.shape, self.out_channels)
        x = self.norm(x.astype(self.dtype))
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
        return x, k_cache, v_cache
    
# next cell prediction is not supported for now. Please refer to lyy's github repo.

class MetaBlock(nn.Module):
    """Meta block."""

    in_channels: int
    channels: int
    num_patches: int
    num_layers: int
    num_heads: int
    num_classes: int
    permutation: PermutationFlip
    dropout: float = 0.0
    debug: bool = False
    mode: int = SAME_ORDER_L2_EACH_BLOCK

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

    def forward_flatten(self,
                        x: jnp.ndarray,
                        y: jnp.ndarray | None = None,
                        temp: float = 1.0,
                        which_cache: str = 'cond',
                        train: bool = True,
                        rng = None):
        # for student use
        x_proj = self.proj_in(x)
        # permutation = PermutationConfig.from_dict(self.permutation).build_permutation()
        pos_embed = self.permutation(self.pos_emebdding)
        # pos_embed = apply_cell_trans(pos_embed, self.cell_size)
        x_proj = x_proj + pos_embed
        if self.class_embedding is not None:
            if y is not None:
                mask = (y < 0).astype(jnp.float32).reshape(-1, 1, 1)
                class_embed = (1 - mask) * self.class_embedding[y] + mask * self.class_embedding.mean(axis=0)
                x_proj = x_proj + class_embed
            else:
                x_proj = x_proj + self.class_embedding.mean(axis=0)
                
        for block in self.blocks:
            rng, rng_used = safe_split(rng)
            x_proj, _, _ = block(x_proj, mask=self.attn_mask(), temp=temp, which_cache=which_cache, train=train, rng=rng_used)
            del rng_used
            
        x_proj = self.proj_out(x_proj) # [B, T, 2*C]
        x_proj = jnp.concatenate([jnp.zeros_like(x_proj[:, :self.square_cell_size]), x_proj[:, :-self.square_cell_size]], axis=1)
        alpha, mu = jnp.split(x_proj, 2, axis=-1)
        if self.mode in [SAME_ORDER_L2_EACH_BLOCK, SAME_ORDER_L2_MA_EACH_BLOCK]:
            x_new = (x - mu) * jnp.exp(-alpha) # [B, T, C_in]
        elif self.mode in [REV_ORDER_L2_EACH_BLOCK, REV_ORDER_L2_MA_EACH_BLOCK]:
            x_new = x * jnp.exp(alpha) + mu
        return x_new
        
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
        x_in = self.permutation(x)
        x = self.proj_in(x_in)
        x = x + self.permutation(self.pos_emebdding)
        # x = self.permutation(x) # [B, T, C]
        
        if self.class_embedding is not None:
            if y is not None:
                mask = (y < 0).astype(jnp.float32).reshape(-1, 1, 1)
                class_embed = (1 - mask) * self.class_embedding[y] + mask * self.class_embedding.mean(axis=0)
                x = x + class_embed
            else:
                x = x + self.class_embedding.mean(axis=0)
        
        for block in self.blocks:
            rng, rng_used = safe_split(rng)
            x, _, _ = block(x, mask=self.attn_mask(), temp=temp, which_cache=which_cache, train=train, rng=rng_used)
            del rng_used
        
        x = self.proj_out(x) # [B, T, 2*C]
        x = jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        alpha, mu = jnp.split(x, 2, axis=-1)
        if self.mode in [SAME_ORDER_L2_EACH_BLOCK, SAME_ORDER_L2_MA_EACH_BLOCK]:
            x_new = (x_in - mu) * jnp.exp(-alpha) # [B, T, C_in]
            log_jacob = - alpha.mean(axis=(1, 2))
        elif self.mode in [REV_ORDER_L2_EACH_BLOCK, REV_ORDER_L2_MA_EACH_BLOCK]:
            x_new = x_in * jnp.exp(alpha) + mu
            log_jacob = alpha.mean(axis=(1, 2))
        x_new = self.permutation(x_new, inverse=True)
        return x_new, log_jacob, alpha, mu

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
            mask = self.attn_mask()
            mask = jax.lax.dynamic_slice(mask, (i, 0), (1, mask.shape[1]))
            x, k_cache[bi], v_cache[bi] = block(x, k_cache=k_cache[bi], v_cache=v_cache[bi], temp=temp, which_cache=which_cache, train=train, mask=mask)
        
        x = self.proj_out(x) # [B, 1, 2*C]
        mu, alpha = jnp.split(x, 2, axis=-1)
        return mu, alpha, k_cache, v_cache # [B, C_in]

    def reverse(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray | None = None,
        temp: float = 1.0,
        which_cache: str = 'cond',
        train: bool = False,
    ):
        x = self.permutation(x)
        B, T, C = x.shape
        num_heads = self.num_heads
        head_dim = self.channels // num_heads
        k_cache, v_cache = [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), "idx": 0} for _ in range(self.num_layers)], [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), "idx": 0} for _ in range(self.num_layers)]
        
        
        def step_fn(i, vals):
            x_step, log_jacob, k_cache_step, v_cache_step = vals
            for obj in k_cache_step + v_cache_step:
                obj["idx"] = i
                
            alpha_i, mu_i, k_cache_step, v_cache_step = self.reverse_step(x_step, i, y, k_cache_step, v_cache_step, temp, which_cache, train)
            
            x_sliced = jax.lax.dynamic_slice(x_step, (0, (i+1), 0), (x_step.shape[0], 1, x_step.shape[2]))
            assert x_sliced.shape == (B, 1, C)
            if self.mode in [SAME_ORDER_L2_EACH_BLOCK, SAME_ORDER_L2_MA_EACH_BLOCK]:
                val = x_sliced * jnp.exp(alpha_i.astype(jnp.float32)) + mu_i
                log_jacob += alpha_i.mean(axis=(1, 2))
            elif self.mode in [REV_ORDER_L2_EACH_BLOCK, REV_ORDER_L2_MA_EACH_BLOCK]:
                val = (x_sliced - mu_i) * jnp.exp(-alpha_i.astype(jnp.float32))
                log_jacob -= alpha_i.mean(axis=(1, 2))
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            x_step = jax.lax.dynamic_update_slice(x_step, val, (0, (i+1), 0))
            return x_step, log_jacob, k_cache_step, v_cache_step
        
        tot_log_jacob = jnp.zeros((B,), dtype=jnp.float32)
        x, tot_log_jacob, _, _ = jax.lax.fori_loop(0, T - 1, step_fn, (x, tot_log_jacob, k_cache, v_cache))
        
        x = self.permutation(x, inverse=True)
        
        return x, tot_log_jacob
    
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
    x = block.permutation(x)
    B, T, C = x.shape
    num_heads = block.num_heads
    head_dim = block.channels // num_heads
    k_cache, v_cache = [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=jnp.nan), "idx": 0} for _ in range(block.num_layers)], [{'cond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), 'uncond': jnp.full((B, T, num_heads, head_dim), fill_value=1e8), "idx": 0} for _ in range(block.num_layers)]
    
    assert T == block.num_patches
    
    def step_fn(i, vals):
        x_step, k_cache_step, v_cache_step = vals
        for obj in k_cache_step + v_cache_step:
            obj["idx"] = i
        alpha_i_cond, mu_i_cond, k_cache_step, v_cache_step = block.apply(params, x_step, i, y, k_cache_step, v_cache_step, temp, "cond", train, method=block.reverse_step)
        
        alpha_i_uncond, mu_i_uncond, k_cache_step, v_cache_step = block.apply(params, x_step, i, None, k_cache_step, v_cache_step, temp, "uncond", train, method=block.reverse_step)
        
        alpha_i = (1 + guidance * i / (T - 1)) * alpha_i_cond - guidance * i / (T - 1) * alpha_i_uncond
        mu_i = (1 + guidance * i / (T - 1)) * mu_i_cond - guidance * i / (T - 1) * mu_i_uncond
        
        x_sliced = jax.lax.dynamic_slice(x_step, (0, i+1, 0), (x_step.shape[0], 1, x_step.shape[2]))
        assert x_sliced.shape == (B, 1, C)
        if block.mode in [SAME_ORDER_L2_EACH_BLOCK, SAME_ORDER_L2_MA_EACH_BLOCK]:
            val = x_sliced * jnp.exp(alpha_i.astype(jnp.float32)) + mu_i
        elif block.mode in [REV_ORDER_L2_EACH_BLOCK, REV_ORDER_L2_MA_EACH_BLOCK]:
            val = (x_sliced - mu_i) * jnp.exp(-alpha_i.astype(jnp.float32))
        else:
            raise ValueError(f"Unknown mode: {block.mode}")
        x_step = jax.lax.dynamic_update_slice(x_step, val, (0, i+1, 0))
        return x_step, k_cache_step, v_cache_step
    
    x, _, _ = jax.lax.fori_loop(0, T-1, step_fn, (x, k_cache, v_cache))
    return block.permutation(x, inverse=True)

def reverse_block_student(
    params,
    block: MetaBlock, 
    x: jnp.ndarray, 
    y: jnp.ndarray | None = None,
    temp: float = 1.0, 
    which_cache: str = 'cond', 
    guidance: float = 0,
    train: bool = False):
    
    x = block.permutation(x)
    B, T, C = x.shape
    num_heads = block.num_heads
    # head_dim = block.channels // num_heads
    
    assert T == block.num_patches
    scs = block.cell_size ** 2
    assert T % scs == 0, f"{T} % {scs} != 0"
    
    x_cond = block.apply(params, x, y, temp, which_cache, train, method=block.forward_flatten)
    x_uncond = block.apply(params, x, None, temp, which_cache, train, method=block.forward_flatten)
    
    x = (1 + guidance) * x_cond - guidance * x_uncond # simple guidance. TODO: more complex?
    
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
    mode: int = SAME_ORDER_L2_EACH_BLOCK
    debug: bool = False
    prior_norm: float = 1.0 # not supported for now
    
    def setup(self):
        assert self.prior_norm == 1.0, f"prior_norm is not supported for now, but got {self.prior_norm}"
        self.patch_num = patch_num = self.img_size // self.patch_size
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
                mode=self.mode,
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
    
    def forward_on_each_block(self,
                xs: jnp.ndarray,
                y: jnp.ndarray | None = None,
                temp: float = 1.0,
                which_cache: str = 'cond',
                train: bool = True,
                rng = None,
        ):
            # used for student traning. We now use forward_with_sg instead.
            raise DeprecationWarning
            N = xs.shape[0]
            assert N == self.num_blocks
            zs = jnp.zeros_like(xs)
            alphas = jnp.zeros_like(xs)
            mus = jnp.zeros_like(xs)
            
            # xs_new = xs
            # rng, rng_used = safe_split(rng)
            # z_temp, _, alpha, mu = self.blocks[0].forward(xs_new[0], y, temp=temp, which_cache=which_cache, train=train, rng=rng)
            # xs_new = xs_new.at[0].set(z_temp)
            
            for i in range(N):
                rng, rng_used = safe_split(rng)
                z, _, alpha, mu = self.blocks[i].forward(xs[i], y, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
                zs = zs.at[i].set(z)
                alphas = alphas.at[i].set(alpha)
                mus = mus.at[i].set(mu)
                del rng_used
            
            return zs, alphas, mus
    
    def forward_with_sg(self,
                x: jnp.ndarray,
                y: jnp.ndarray | None = None,
                temp: float = 1.0,
                which_cache: str = 'cond',
                train: bool = True,
                rng = None,
        ):
            # used for student traning.
            # assert N == self.num_blocks
            B, T, C = x.shape
            
            zs = jnp.zeros((self.num_blocks, B, T, C), dtype=self.dtype)
            current_x = x
            
            for i in range(self.num_blocks):
                rng, rng_used = safe_split(rng)
                next_x, _, _, _ = self.blocks[i].forward(current_x, y, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
                zs = zs.at[i].set(next_x)
                current_x = jax.lax.stop_gradient(next_x)
                del rng_used
            
            return zs
    
    def forward_flatten(self, 
                x: jnp.ndarray, 
                y: jnp.ndarray | None = None, 
                temp: float = 1.0, 
                which_cache: str = 'cond', 
                train: bool = True,
                rng = None,
        ):
            B, T, C = x.shape
            xs = jnp.zeros((self.num_blocks+1, B, T, C), dtype=self.dtype)
            alphas = jnp.zeros((self.num_blocks, B, T, C), dtype=self.dtype)
            mus = jnp.zeros((self.num_blocks, B, T, C), dtype=self.dtype)
            xs = xs.at[0].set(x)

            tot_logdet = 0.0
            i = 0
            for block in self.blocks:
                rng, rng_used = safe_split(rng)
                x, logdet, alpha, mu = block.forward(x, y, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
                alphas = alphas.at[i].set(alpha)
                mus = mus.at[i].set(mu)
                i += 1
                xs = xs.at[i].set(x)
                tot_logdet = tot_logdet + logdet
                del rng_used
            
            log_prior = 0.5 * (x ** 2).mean()
            loss = - tot_logdet.mean() + log_prior
            loss_dict = {'loss': loss, 'log_det': tot_logdet.mean(), 'log_prior': log_prior}
            
            return loss, loss_dict, xs, alphas, mus
        
    def reverse(self,
                x: jnp.ndarray,
                y: jnp.ndarray | None = None,
                temp: float = 1.0,
                which_cache: str = 'cond',
                train: bool = False,
        ):
        x = self.patchify(x)
        tot_log_jacob = 0.0
        for i in range(self.num_blocks-1,-1,-1):
            x, log_jacob = self.blocks[i].reverse(x, y, temp=temp, which_cache=which_cache, train=train)
            tot_log_jacob += log_jacob
            
        log_prior = 0.5 * (x ** 2).mean()
        loss = - tot_log_jacob.mean() + log_prior
        loss_dict = {'loss': loss, 'log_det': tot_log_jacob.mean(), 'log_prior': log_prior}
        return loss, loss_dict, x
        
    def __call__(self, 
                x: jnp.ndarray, 
                y: jnp.ndarray | None = None, 
                temp: float = 1.0, 
                which_cache: str = 'cond', 
                train: bool = True,
                rng = jax.random.PRNGKey(0),
        ):
        """
        Args:
            x: [B, H, W, C]
            y: [B]
            temp: scalar
            which_cache: str
            train: bool
        """
        # total_x = [x]
        x = self.patchify(x)  # [B, T, C]
        B, T, C = x.shape
        xs = jnp.zeros((self.num_blocks+1, B, T, C), dtype=self.dtype)
        alphas = jnp.zeros((self.num_blocks, B, T, C), dtype=self.dtype)
        mus = jnp.zeros((self.num_blocks, B, T, C), dtype=self.dtype)
        xs = xs.at[0].set(x)

        tot_logdet = 0.0
        i = 0
        for block in self.blocks:
            rng, rng_used = safe_split(rng)
            x, logdet, alpha, mu = block.forward(x, y, temp=temp, which_cache=which_cache, train=train, rng=rng_used)
            alphas = alphas.at[i].set(alpha)
            mus = mus.at[i].set(mu)
            i += 1
            xs = xs.at[i].set(x)
            tot_logdet = tot_logdet + logdet
            del rng_used
        
        log_prior = 0.5 * (x ** 2).mean() * (self.prior_norm ** -2.0)
        
        # TODO: impl. update prior
        
        loss = - tot_logdet.mean() + log_prior
        
        loss_dict = {'loss': loss, 'log_det': tot_logdet.mean(), 'log_prior': log_prior}
        
        return loss, loss_dict, xs, alphas, mus
    
class TeacherStudent(nn.Module):
    """Normalizing flow, teacher-student model."""
    
    img_size: int
    out_channels: int
    channels: int
    patch_size: int
    num_layers: int
    num_heads: int
    num_blocks: int
    # perms: list[PermutationConfig]
    num_classes: int = 0
    dtype: Any = jnp.float32
    cell_size: int = 1
    teacher_dropout: float = 0.0
    student_dropout: float = 0.0
    mode: int = SAME_ORDER_L2_EACH_BLOCK
    debug: bool = False
    
    def setup(self):
        self.teacher = NormalizingFlow(
            img_size=self.img_size,
            out_channels=self.out_channels,
            channels=self.channels,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            # perms=self.perms,
            num_classes=self.num_classes,
            dtype=self.dtype,
            dropout=self.teacher_dropout,
            cell_size=self.cell_size,
            debug=self.debug,
        )
        if self.mode in [SAME_ORDER_L2_EACH_BLOCK, SAME_ORDER_L2_MA_EACH_BLOCK]:
            raise NotImplementedError("SAME_ORDER_L2_EACH_BLOCK and SAME_ORDER_L2_MA_EACH_BLOCK are not implemented")
            self.student = NormalizingFlow(
                img_size=self.img_size,
                out_channels=self.out_channels,
                channels=self.channels,
                patch_size=self.patch_size,
                num_layers=int(self.num_layers),
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                perms=self.perms,
                num_classes=self.num_classes,
                dtype=self.dtype,
                cell_size=self.cell_size,
                dropout=self.student_dropout,
                mode=self.mode,
                debug=self.debug,
            )
        elif self.mode in [REV_ORDER_L2_EACH_BLOCK, REV_ORDER_L2_MA_EACH_BLOCK]:
            self.student = NormalizingFlow(
                img_size=self.img_size,
                out_channels=self.out_channels,
                channels=self.channels,
                patch_size=self.patch_size,
                num_layers=int(self.num_layers),
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                perms=self.perms[::-1],
                num_classes=self.num_classes,
                dtype=self.dtype,
                cell_size=self.cell_size,
                dropout=self.student_dropout,
                mode=self.mode,
                debug=self.debug,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
    def patchify(self, x):
        B, H, W, C = x.shape
        assert C == self.out_channels, "input must be RGB image"
        x = x.reshape(B, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        patch_num = self.img_size // self.patch_size
        num_patches = patch_num ** 2
        x = x.reshape(B, num_patches, self.patch_size * self.patch_size * C)
        return x
        
    def unpatchify(self, x):
        B, T, C = x.shape
        x = x.reshape(B, self.img_size // self.patch_size, self.img_size // self.patch_size, self.patch_size, self.patch_size, self.out_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.img_size, self.img_size, self.out_channels)
        return x
    
    def calc_student_reverse(self, x, y, temp: float = 1.0, which_cache: str = 'cond', train: bool = False):
        _, _, z = self.student.reverse(x, y, temp=temp, which_cache=which_cache, train=train)
        return z
    
    def calc_student_forward(self, x, y, temp: float = 1.0, which_cache: str = 'cond', train: bool = False):
        _, _, zs, _, _ = self.student(x, y, temp=temp, which_cache=which_cache, train=train)
        return zs[-1]
    
    def calc_teacher_forward(self, x, y, temp: float = 1.0, which_cache: str = 'cond', train: bool = True):
        loss, _, _, _, _ = self.teacher(x, y, temp=temp, which_cache=which_cache, train=train)
        return loss
    
    def __call__(self, 
                x: jnp.ndarray, 
                y: jnp.ndarray | None = None, 
                # zs: jnp.ndarray | None = None,
                temp: float = 1.0, 
                which_cache: str = 'cond', 
                train: bool = True,
        ):
        
        rng = self.make_rng('dropout')
        rng, rng_used = safe_split(rng)
        rng, rng_used_2 = safe_split(rng)
        rng, rng_used_3 = safe_split(rng)
                
        loss_1, loss_dict_1, zs, _, _ = self.teacher(x, y, temp=temp, which_cache=which_cache, train=False, rng=rng_used)
        del rng_used
        loss_dict = {}
        
        if self.mode in [SAME_ORDER_L2_EACH_BLOCK, SAME_ORDER_L2_MA_EACH_BLOCK]:
            raise NotImplementedError("SAME_ORDER_L2_EACH_BLOCK and SAME_ORDER_L2_MA_EACH_BLOCK are not implemented")
            zs = jax.lax.stop_gradient(zs)
            alphas_t = jax.lax.stop_gradient(alphas_t)
            mus_t = jax.lax.stop_gradient(mus_t)

            loss_2, loss_dict_2, zs_s, alphas_s, mus_s = self.student(x, y, temp=temp, which_cache=which_cache, train=train, rng=rng_used_2)
            xs, alphas, mus = self.student.forward_on_each_block(zs[:-1], y, temp=temp, which_cache=which_cache, train=train, rng=rng_used_3)
            
            loss_dict = {
                'loss_student': loss_2,
                'loss_student_prior': loss_dict_2['log_prior'],
                'loss_student_logdet': loss_dict_2['log_det'],
            }
            
            if self.mode == SAME_ORDER_L2_EACH_BLOCK:
                losses = jnp.mean((xs - zs[1:]) ** 2, axis=(1, 2, 3))
                loss = jnp.sum(losses)
                
                losses = losses.reshape(2, -1)
                block_losses = jnp.sum(losses, axis=1)
                loss_dict["front_distill_loss"] = block_losses[0]
                loss_dict["back_distill_loss"] = block_losses[1]
            
            if self.mode == SAME_ORDER_L2_MA_EACH_BLOCK:
                loss_alpha = jnp.mean((alphas - alphas_t) ** 2, axis=(1, 2, 3))
                loss_alpha = jnp.sum(loss_alpha)
                loss_mu = jnp.mean((mus - mus_t) ** 2, axis=(1, 2, 3))
                loss_mu = jnp.sum(loss_mu)
                loss = loss_mu + loss_alpha
                loss_dict["loss_mu"] = loss_mu
                loss_dict["loss_alpha"] = loss_alpha
        
        elif self.mode in [REV_ORDER_L2_EACH_BLOCK, REV_ORDER_L2_MA_EACH_BLOCK]:
            zs = jax.lax.stop_gradient(zs)
            zs = jnp.flip(zs, axis=0)
            
            # xs, alphas, mus = self.student.forward_on_each_block(zs[:-1], y, temp=temp, which_cache=which_cache, train=train, rng=rng_used_3)
            xs = self.student.forward_with_sg(zs[0], y, temp=temp, which_cache=which_cache, train=train, rng=rng_used_3)
            
            if self.mode == REV_ORDER_L2_MA_EACH_BLOCK:
                raise NotImplementedError("REV_ORDER_L2_MA_EACH_BLOCK is not encouraged by Kaiming")
                loss_alpha = jnp.mean((alphas - alphas_t) ** 2, axis=(1, 2, 3))
                loss_alpha = jnp.sum(loss_alpha)
                loss_mu = jnp.mean((mus - mus_t) ** 2, axis=(1, 2, 3))
                loss_mu = jnp.sum(loss_mu)
                
                loss = loss_alpha + loss_mu
                loss_dict['loss_mu'] = loss_mu
                loss_dict['loss_alpha'] = loss_alpha
            
            if self.mode == REV_ORDER_L2_EACH_BLOCK:
                losses = jnp.mean((xs - zs[1:]) ** 2, axis=(1, 2, 3))
                norm_x = jnp.mean(xs ** 2, axis=(1, 2, 3))
                norm_z = jnp.mean(zs[1:] ** 2, axis=(1, 2, 3))
                norm_x = jax.lax.stop_gradient(norm_x)
                norm_z = jax.lax.stop_gradient(norm_z)
                
                # block 0 means first noise to image
                for i in range(len(losses)):
                    loss_dict[f"block_{i}"] = losses[i]
                for i in range(len(norm_x)):
                    loss_dict[f"norm_x_{i}"] = norm_x[i]
                for i in range(len(norm_z)):
                    loss_dict[f"norm_z_{i}"] = norm_z[i]
                    
                # losses /= (norm_x + norm_z)
                # losses *= jnp.mean(norm_x + norm_z)
                loss = jnp.sum(losses)
            
        loss_dict['loss'] = loss
        
        return loss, loss_dict, zs

    
def reverse(params,
            nf: TeacherStudent,
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
        block_param = params['params']['teacher'][f'blocks_{i}']
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

def reverse_student(params,
            nf: TeacherStudent,
            x: jnp.ndarray,
            y: jnp.ndarray | None = None,
            temp: float = 1.0,
            guidance: float = 0,
            which_cache: str = 'cond',
            train: bool = False):
    print('param keys:', params['params'].keys())
    # for block_param in params['params']['blocks'][::-1]:
    patch_num = nf.img_size // nf.patch_size
    num_patches = patch_num ** 2
    in_channels = nf.out_channels * nf.patch_size * nf.patch_size
    if nf.mode in [SAME_ORDER_L2_EACH_BLOCK, SAME_ORDER_L2_MA_EACH_BLOCK]:
        raise NotImplementedError("SAME_ORDER_L2_EACH_BLOCK and SAME_ORDER_L2_MA_EACH_BLOCK are not implemented")
        for i in range(nf.num_blocks-1,-1,-1):
            block_param = params['params']['student'][f'blocks_{i}']
            x = reverse_block({"params": block_param}, MetaBlock(
                    in_channels=in_channels, 
                    channels=nf.channels, 
                    num_patches=num_patches, 
                    num_layers=int(nf.num_layers), 
                    num_heads=nf.num_heads, 
                    num_classes=nf.num_classes, 
                    permutation=nf.perms[i],
                    cell_size=nf.cell_size,
                    debug=nf.debug,
                ), x, y, temp=temp, which_cache=which_cache, train=train, guidance=guidance)
    elif nf.mode in [REV_ORDER_L2_EACH_BLOCK, REV_ORDER_L2_MA_EACH_BLOCK]:
        for i in range(nf.num_blocks):
            block_param = params['params']['student'][f'blocks_{i}']
            x = reverse_block_student({"params": block_param}, MetaBlock(
                    in_channels=in_channels, 
                    channels=nf.channels, 
                    num_patches=num_patches, 
                    num_layers=int(nf.num_layers), 
                    num_heads=nf.num_heads, 
                    num_classes=nf.num_classes, 
                    permutation=nf.perms[nf.num_blocks-1-i],
                    cell_size=nf.cell_size,
                    mode=nf.mode,
                    debug=nf.debug,
                ), x, y, temp=temp, which_cache=which_cache, train=train, guidance=guidance)
        # print("mean during layer:", x.mean())
    x = nf.unpatchify(x)
    return x

# move this out from model for JAX compilation
def generate(params, model: TeacherStudent, rng, n_sample, noise_level, guidance, temperature=1.0, label_cond=True, denoise=True):
    """
    Generate samples from the model
    """
    patch_num = model.img_size // model.patch_size
    num_patches = patch_num ** 2
    in_channels = model.out_channels * model.patch_size * model.patch_size
    x_shape = (n_sample, num_patches, in_channels)
    rng_used, rng = jax.random.split(rng, 2)
    rng_used_2, rng = jax.random.split(rng, 2)
    z = jax.random.normal(rng_used, x_shape, dtype=model.dtype) * model.prior_norm
    if label_cond:
        y = jax.random.randint(rng_used_2, (n_sample,), 0, model.num_classes)
    else:
        y = None
    
    if label_cond:
        x = reverse(params, model, z, y, guidance=guidance, temp=1.0)
    else:
        x = reverse(params, model, z, y, guidance=guidance, temp=temperature)
    
    if noise_level == 0 or not denoise: return x
    def nabla_log_prob(x):
        loss = model.apply(params, x, y, method=model.calc_teacher_forward)
        return loss
    
    nabla_log_prob = jax.grad(nabla_log_prob)
    grad_dir = nabla_log_prob(x) * jnp.prod(jnp.array(x.shape))
    
    x = x - (noise_level ** 2) * grad_dir # use score-based model to denoise
    
    return x

def generate_student(params, model: TeacherStudent, rng, n_sample, noise_level, guidance, temperature=1.0, label_cond=True, mult_prior=1.0, denoise=True):
    """
    Generate samples from the model
    """
    patch_num = model.img_size // model.patch_size
    num_patches = patch_num ** 2
    in_channels = model.out_channels * model.patch_size * model.patch_size
    x_shape = (n_sample, num_patches, in_channels)
    rng_used, rng = jax.random.split(rng, 2)
    rng_used_2, rng = jax.random.split(rng, 2)
    z = jax.random.normal(rng_used, x_shape, dtype=model.dtype)
    
    z *= mult_prior
    
    if label_cond:
        y = jax.random.randint(rng_used_2, (n_sample,), 0, model.num_classes)
    else:
        y = None
    
    if label_cond:
        x = reverse_student(params, model, z, y, guidance=guidance, temp=1.0)
    else:
        x = reverse_student(params, model, z, y, guidance=guidance, temp=temperature)
        
    # denoising
    def nabla_log_prob(x):
        loss = model.apply(params, x, y, method=model.calc_teacher_forward) # note that I use teacher here.
        # TODO: impl. student score model
        return loss
    
    if denoise:
        nabla_log_prob = jax.grad(nabla_log_prob)
        grad_dir = nabla_log_prob(x) * jnp.prod(jnp.array(x.shape))
        
        x = x - (noise_level ** 2) * grad_dir # use score-based model to denoise
        
    return x

def generate_prior(params, model: TeacherStudent, rng, n_sample, noise_level, guidance, temperature=1.0, label_cond=True):
    patch_num = model.img_size // model.patch_size
    num_patches = patch_num ** 2
    in_channels = model.out_channels * model.patch_size * model.patch_size
    x_shape = (n_sample, num_patches, in_channels)
    rng_used, rng = jax.random.split(rng, 2)
    rng_used_2, rng = jax.random.split(rng, 2)
    z = jax.random.normal(rng_used, x_shape, dtype=model.dtype)
    if label_cond:
        y = jax.random.randint(rng_used_2, (n_sample,), 0, model.num_classes)
    else:
        y = None
    
    if label_cond:
        x = reverse(params, model, z, y, guidance=guidance, temp=1.0)
    else:
        x = reverse(params, model, z, y, guidance=guidance, temp=temperature)
    
    z = model.apply(params, x, method=model.patchify)
    
    for i in range(model.num_blocks-1,-1,-1):
        block_param = params['params']['student'][f'blocks_{i}']
        z = reverse_block({"params": block_param}, MetaBlock(
                in_channels=in_channels, 
                channels=model.channels, 
                num_patches=num_patches, 
                num_layers=int(model.num_layers), 
                num_heads=model.num_heads, 
                num_classes=model.num_classes, 
                permutation=PermutationFlip((model.num_blocks-1-i)%2==1),
                cell_size=model.cell_size,
                debug=model.debug,
                mode=model.mode,
        ), z, y)
        
    z = model.unpatchify(z)
    # x_recover = model.apply(params, z, y, method=model.calc_student_forward)
    # x_recover = model.unpatchify(x_recover)
    return x, z
    

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