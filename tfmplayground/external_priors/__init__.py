"""Interfaces to external prior libraries (TabICL, TICL, TabPFN v1)."""

from .base import PriorDataLoader, PriorDumpDataLoader

__all__ = [
    "PriorDataLoader",
    "PriorDumpDataLoader",
]

# The generator wrappers need the heavy external prior libraries (install the
# `external-priors` extra); keep base importable without them.
try:
    from .tabicl import TabICLPriorDataLoader  # noqa: F401

    __all__.append("TabICLPriorDataLoader")
except ImportError:
    pass

try:
    from .tabpfn import TabPFNPriorDataLoader, build_tabpfn_prior  # noqa: F401

    __all__ += ["TabPFNPriorDataLoader", "build_tabpfn_prior"]
except ImportError:
    pass

try:
    from .ticl import TICLPriorDataLoader, build_ticl_prior  # noqa: F401

    __all__ += ["TICLPriorDataLoader", "build_ticl_prior"]
except ImportError:
    pass
