from .qwen35_scalar_fs import (
    ScalarFutureSeedConfig,
    apply_scalar_future_seed,
    freeze_except_future_seed,
    get_future_seed_runtime_stats,
    install_qwen35_upstream_compat_fixes,
    list_future_seed_parameters,
)
from .qwen3next_scalar_fs import apply_qwen3next_scalar_future_seed

__all__ = [
    "ScalarFutureSeedConfig",
    "apply_scalar_future_seed",
    "apply_qwen3next_scalar_future_seed",
    "freeze_except_future_seed",
    "get_future_seed_runtime_stats",
    "install_qwen35_upstream_compat_fixes",
    "list_future_seed_parameters",
]
