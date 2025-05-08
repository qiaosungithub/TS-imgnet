from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from functools import partial

# Pytorch-like initialization
torch_linear = partial(nn.Dense, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)

def safe_split(rng):
    if rng is None:
        return None, None
    else:
        return jax.random.split(rng)

class PermutationFlip:
    """Permutation and flip for images."""

    def __init__(self, flip: bool = True):
        self.flip = flip

    def __call__(self, x: jnp.ndarray, inverse: bool = False):
        # inverse: legacy, not used.
        if self.flip:
            x = jnp.flip(x, axis=1)
        return x

def sqa_attention(q, k, v, mask, train, dropout_module=None, dropout_rng=None, dtype=jnp.float32):
    """
    Compute the attention weights and output.
    sanity check with nn.dot_product_attention
    qkv shape: [B, T, num_head, head_dim]
    """
    if train: assert dropout_rng is not None

    head_dim = q.shape[-1]
    q = q / jnp.sqrt(head_dim).astype(dtype)
    dot_product = jnp.einsum('...qhd, ...khd->...hqk', q, k)

    # deal with mask
    big_neg = jnp.finfo(dtype).min # to avoid -inf
    dot_product = jnp.where(mask, dot_product, big_neg)
    
    attn_weights = nn.softmax(dot_product, axis=-1).astype(dtype)
    
    # Apply dropout on attn_weights
    if train:
        attn_weights = dropout_module(attn_weights, rng=dropout_rng, deterministic=False)
    
    # Compute the output
    output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, v)
    
    return output

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
        self.attndrop = nn.Dropout(rate=self.dropout)
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
        
        # attn = nn.dot_product_attention(query=q, key=k, value=v, mask=mask, dropout_rng=rng_used, deterministic=not train, precision=None, dropout_rate=self.dropout, broadcast_dropout=False, dtype=self.dtype)
        attn = sqa_attention(q, k, v, mask=mask, train=train, dropout_module=self.attndrop, dropout_rng=rng_used, dtype=self.dtype)
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
    mode: str = "same" # options: same, reverse

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
        raise DeprecationWarning
        # for student use. image only forward. The input has been permuted.
        x_proj = self.proj_in(x)
        pos_embed = self.permutation(self.pos_emebdding)
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
        x_proj = jnp.concatenate([jnp.zeros_like(x_proj[:, :1]), x_proj[:, :-1]], axis=1)
        alpha, mu = jnp.split(x_proj, 2, axis=-1) 
        assert self.mode == REV_ORDER_L2_EACH_BLOCK
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
        forward with mu, alpha, jacobian. The input will be permuted in this function.
        foward: *exp(alpha) + mu
        """
        x_in = self.permutation(x)
        x = self.proj_in(x_in)
        x = x + self.permutation(self.pos_emebdding) # [B, T, C]
        
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
        if self.mode == "same":
            x_new = (x_in - mu) * jnp.exp(-alpha) # [B, T, C_in]
            log_jacob = - alpha.mean(axis=(1, 2))
        elif self.mode == "reverse":
            x_new = x_in * jnp.exp(alpha) + mu
            log_jacob = alpha.mean(axis=(1, 2))
        else: raise NotImplementedError(f"Unknown mode: {self.mode}")
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
        train: bool = False,
    ):
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
        """
        reverse: (x-mu) * exp(-alpha)
        """
        raise LookupError("没用到？")
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
            assert self.mode == REV_ORDER_L2_EACH_BLOCK
            val = (x_sliced - mu_i) * jnp.exp(-alpha_i.astype(jnp.float32))
            log_jacob -= alpha_i.mean(axis=(1, 2))
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