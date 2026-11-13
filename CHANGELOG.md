# Changelog

## Unreleased

Fixes after review.

- The recalibration table names its loss column "top-label log loss" and, for vectors, prints
  the log loss of the whole vector before and after temperature scaling. Its JSON form lists
  the fitted temperature or Platt parameters of each fold.
- `compare` includes the classwise ECE and the whole-vector log loss and Brier score when
  both files hold vectors.
- A temperature search that ends on an edge of its range [0.01, 100] raises a warning and
  returns that edge.
- Infinite entries in a probability or logit vector are rejected with file and line.
- Under `--scale auto` a bare stated value strictly between 1 and 10 is rejected as ambiguous.
- New example `examples/false_alarm_rate.py` and a test for the false-alarm rate of the
  consistency test on continuous confidences.

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
