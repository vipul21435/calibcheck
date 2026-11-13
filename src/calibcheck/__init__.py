"""calibcheck: measure how well stated confidence matches observed accuracy."""

from .binning import assign_bins, equal_mass_edges, equal_width_edges, make_edges
from .io import Items, read_items
from .metrics import (
    BinStats,
    BrierDecomposition,
    RiskCoverage,
    accuracy_at_coverage,
    aurc,
    auroc,
    bin_stats,
    brier_score,
    ece,
    log_loss,
    mce,
    murphy_decomposition,
    risk_coverage,
    squared_calibration_error,
    summary,
)
from .multiclass import (
    classwise_ece,
    multiclass_brier,
    multiclass_log_loss,
    softmax,
    top_label,
)
from .recalibrate import (
    HistogramBinning,
    IsotonicRegression,
    PlattScaling,
    TemperatureScaling,
    apply_temperature,
    cross_fit,
    cross_fit_temperature,
    fit_apply,
    fit_temperature,
)
from .resample import Consistency, bootstrap, consistency_resampling
from .stated import digit_preference, parse_stated

__version__ = "0.1.0"

__all__ = [
    "BinStats",
    "BrierDecomposition",
    "Consistency",
    "HistogramBinning",
    "IsotonicRegression",
    "Items",
    "PlattScaling",
    "RiskCoverage",
    "TemperatureScaling",
    "__version__",
    "accuracy_at_coverage",
    "apply_temperature",
    "assign_bins",
    "aurc",
    "auroc",
    "bin_stats",
    "bootstrap",
    "brier_score",
    "classwise_ece",
    "consistency_resampling",
    "cross_fit",
    "cross_fit_temperature",
    "digit_preference",
    "ece",
    "equal_mass_edges",
    "equal_width_edges",
    "fit_apply",
    "fit_temperature",
    "log_loss",
    "make_edges",
    "mce",
    "multiclass_brier",
    "multiclass_log_loss",
    "murphy_decomposition",
    "parse_stated",
    "read_items",
    "risk_coverage",
    "softmax",
    "squared_calibration_error",
    "summary",
    "top_label",
]
