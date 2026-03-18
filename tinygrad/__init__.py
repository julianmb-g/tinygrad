import os

if int(os.getenv("TYPED", "0")):
  from typeguard import install_import_hook
  install_import_hook(__name__)
from tinygrad.engine.jit import TinyJit  # noqa: F401
from tinygrad.function import function  # noqa: F401
from tinygrad.tensor import Tensor  # noqa: F401
from tinygrad.uop.ops import UOp

Variable = UOp.variable
from tinygrad.device import Device  # noqa: F401
from tinygrad.dtype import dtypes  # noqa: F401
from tinygrad.helpers import Context, GlobalCounters, fetch, getenv  # noqa: F401
