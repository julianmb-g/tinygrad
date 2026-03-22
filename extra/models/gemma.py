from tinygrad.tensor import Tensor

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
