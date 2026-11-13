# Changelog

## 0.1.0 - 2026-11-13

First release.

- Metrics for a confidence and a correctness per item: ECE (plug-in and debiased) and MCE with
  equal-width or equal-mass bins, squared and L2 calibration error (plug-in and debiased),
  Brier score with reliability, resolution, uncertainty and a within-bin term, log loss,
  AUROC, the risk-coverage curve with AURC, and accuracy at a fixed coverage.
- Metrics for probability vectors or logits: top-label metrics, classwise ECE, log loss and
  Brier score of the whole vector.
- Intervals: percentile bootstrap over items, or over whole groups, for every metric.
- Consistency resampling: bands for the reliability diagram and a test of calibration.
- Recalibration scored on held-out folds or a separate file: temperature scaling (on a
  confidence or on a vector), Platt scaling, isotonic regression, histogram binning.
- Stated confidence: "85%", "0.85" and phrases from a configurable map; digit preference.
- Commands `metrics`, `diagram`, `recalibrate` and `compare`, with Markdown and JSON output.
- Results on the per-question MMLU details of two models from the archived Open LLM
  Leaderboard, with the script that fetches them.
