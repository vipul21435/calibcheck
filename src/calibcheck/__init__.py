"""calibcheck: measure how well stated confidence matches observed accuracy."""

from .binning import assign_bins, equal_mass_edges, equal_width_edges, make_edges
from .io import Items, read_items
from .metrics import (
    BinStats,
    BrierDecomposition,
    bin_stats,
    brier_score,
    ece,
    mce,
    murphy_decomposition,
    squared_calibration_error,
    summary,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "BinStats",
    "BrierDecomposition",
    "Items",
    "__version__",
    "assign_bins",
    "bin_stats",
    "brier_score",
    "ece",
    "equal_mass_edges",
    "equal_width_edges",
    "make_edges",
    "mce",
    "murphy_decomposition",
    "read_items",
    "squared_calibration_error",
    "summary",
]
