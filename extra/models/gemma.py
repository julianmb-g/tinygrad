from tinygrad.tensor import Tensor
import math

class GemmaRMSNorm:
  def __init__(self, dim: int, eps: float = 1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x: Tensor) -> Tensor:
    # Gemma's RMSNorm multiplies by (1 + weight)
    normed = x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
    return normed * (1 + self.weight)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)

def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
  x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
  x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]
  f_cos, f_sin = freqs_cis[..., 0], freqs_cis[..., 1]
  ro = x0 * f_cos - x1 * f_sin
  co = x0 * f_sin + x1 * f_cos
  return Tensor.stack(ro, co, dim=-1).flatten(3)

class GemmaMLP:
  def __init__(self, hidden_dim: int, ff_dim: int):
    self.gate_proj = Tensor.empty(hidden_dim, ff_dim)
    self.up_proj = Tensor.empty(hidden_dim, ff_dim)
    self.down_proj = Tensor.empty(ff_dim, hidden_dim)

  def __call__(self, x: Tensor) -> Tensor:
    gate = x.matmul(self.gate_proj)
    up = x.matmul(self.up_proj)
    gelu = gate.gelu()
    hidden = gelu * up
    return hidden.matmul(self.down_proj)

class GemmaAttention:
  def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.q_proj = Tensor.empty(hidden_dim, num_heads * head_dim)
    self.k_proj = Tensor.empty(hidden_dim, num_kv_heads * head_dim)
    self.v_proj = Tensor.empty(hidden_dim, num_kv_heads * head_dim)
    self.o_proj = Tensor.empty(num_heads * head_dim, hidden_dim)

  def __call__(self, x: Tensor, freqs_cis: Tensor, mask: Tensor|None = None) -> Tensor:
    bsz, seq_len, _ = x.shape
    q = x.matmul(self.q_proj).reshape(bsz, seq_len, self.num_heads, self.head_dim)
    k = x.matmul(self.k_proj).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
    v = x.matmul(self.v_proj).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)

    q = apply_rotary_emb(q, freqs_cis)
    k = apply_rotary_emb(k, freqs_cis)

    if self.num_heads != self.num_kv_heads:
      num_repeats = self.num_heads // self.num_kv_heads
      k = k.unsqueeze(3).expand(bsz, seq_len, self.num_kv_heads, num_repeats, self.head_dim).flatten(2, 3)
      v = v.unsqueeze(3).expand(bsz, seq_len, self.num_kv_heads, num_repeats, self.head_dim).flatten(2, 3)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask)
    attn = attn.transpose(1, 2).flatten(2)
    return attn.matmul(self.o_proj)
