# calibcheck

Measure how well a model's stated confidence matches how often it is right.

**Status: early scaffold (0.1.0.dev0).** The pieces under "What works so far" are implemented
and tested. Intervals, recalibration, diagrams, model comparison and results on real data are
not there yet; they are listed under "Not implemented yet".

## Why this is hard

The usual estimator of calibration error is biased upward and depends on the binning. With a
finite number of items per bin, a perfectly calibrated model still shows a gap between accuracy
and mean confidence in every bin, so a small non-zero number can come from the estimator
rather than from the model. Recalibration has a matching trap: a map fitted and scored on the
same items always looks good.

## Install

```
pip install git+https://github.com/vipul21435/calibcheck
```

Python 3.10 or later. The only runtime dependency is numpy.

## Quickstart

The input is a JSONL or CSV file with one item per line: a confidence in [0, 1] and whether
the answer was correct.

```
{"id": "q001", "confidence": 0.7, "correct": true}
{"id": "q002", "confidence": 0.9, "correct": true}
```

`examples/example.jsonl` holds 400 simulated items. They are not measurements of any model;
`examples/make_example.py` draws them from a process that is overconfident at 0.7 and above
and slightly underconfident below.

```
$ calibcheck metrics examples/example.jsonl --binning mass
# Calibration report: examples/example.jsonl

400 items, 7 equal-mass bins.

| metric | value |
|---|---|
| accuracy | 0.7150 |
| mean confidence | 0.8300 |
| ECE, plug-in | 0.1210 |
| MCE | 0.2439 |
| squared calibration error, plug-in | 0.0183 |
| squared calibration error, debiased | 0.0146 |
| L2 calibration error, plug-in | 0.1352 |
| L2 calibration error, debiased | 0.1207 |
| Brier score | 0.2158 |
| reliability | 0.0183 |
| resolution | 0.0062 |
| uncertainty | 0.2038 |
| within-bin term | 0.0000 |

| bin | items | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.000, 0.600) | 17 | 0.5000 | 0.4706 | -0.0294 |
| [0.600, 0.700) | 28 | 0.6000 | 0.5357 | -0.0643 |
| [0.700, 0.800) | 44 | 0.7000 | 0.7273 | 0.0273 |
| [0.800, 0.900) | 108 | 0.8000 | 0.7037 | -0.0963 |
| [0.900, 0.950) | 108 | 0.9000 | 0.7500 | -0.1500 |
| [0.950, 1.000) | 54 | 0.9500 | 0.7963 | -0.1537 |
| [1.000, 1.000] | 41 | 1.0000 | 0.7561 | -0.2439 |
```

Because the example is simulated, the error of the process behind it is known:
`python examples/make_example.py` prints a squared calibration error of 0.0087 and an ECE of
0.0833. On these 400 draws the plug-in estimate of the squared error is 0.0183 and the
debiased estimate is 0.0146. One sample of this size is noisy either way; the test suite
checks the bias of both estimators over thousands of simulated samples instead.

Options of `calibcheck metrics`: `--bins N` (default 15), `--binning width` (default) or
`mass`, `--confidence FIELD` and `--correct FIELD` to name the fields, `--format json`,
`--output PATH`.

From Python:

```python
from calibcheck import summary

confidence = [0.9, 0.9, 0.9, 0.9, 0.6, 0.6]
correct = [1, 1, 1, 0, 1, 0]
out = summary(confidence, correct, n_bins=10)
print(round(out["ece"], 4), round(out["sq_ce_plugin"], 4), round(out["sq_ce_debiased"], 4))
# 0.1333 0.0183 -0.1067
```

Six items give a plug-in ECE of 0.13 and a debiased squared error below zero: the observed
gaps are smaller than the sampling variance estimated for bins this small, so these six items
are no evidence of miscalibration.

## What works so far

- Equal-width bins and equal-mass bins. Equal-mass bins keep tied confidences together, so
  the result does not depend on the order of the items.
- ECE (plug-in) and MCE.
- Squared calibration error, plug-in and debiased, and its square root.
- Brier score with the reliability, resolution and uncertainty terms.
- A loader for JSONL and CSV that reports problems with file name and line number.
- `calibcheck metrics` with Markdown and JSON output.

## How it works

Item `i` has confidence `c_i` and outcome `y_i` (1 if correct). Bin `b` holds `n_b` of the
`n` items, with mean confidence `conf_b` and accuracy `acc_b`.

**Bins.** Equal-width bin `i` of `B` is `[i/B, (i+1)/B)`, the last one closed at 1. Guo et
al. (2017) write `((m-1)/M, m/M]`; the two conventions differ only for values exactly on an
edge, which is common for stated confidences such as 0.8 or 0.9. Edges are computed as the
correctly rounded `i/B`, so 0.3 always falls in the bin that starts at 0.3. For equal-mass
bins the sorted confidences are cut at positions `k*n // B` and each edge is the value found
at a cut; duplicate edges collapse, so there can be fewer bins than requested, and none is
empty.

**ECE and MCE** (Guo et al., 2017, who use 15 bins, the default here):

```
ECE = sum_b (n_b / n) * |acc_b - conf_b|        MCE = max_b |acc_b - conf_b|
```

**Squared calibration error** (Kumar, Liang and Ma, 2019, Definitions 5.1 and 5.2):

```
plug-in   = sum_b (n_b / n) * (conf_b - acc_b)^2
debiased  = sum_b (n_b / n) * [ (conf_b - acc_b)^2 - acc_b * (1 - acc_b) / (n_b - 1) ]
```

`acc_b` is a mean of `n_b` outcomes, so the squared gap overshoots the true one by the
variance of that mean; `acc_b * (1 - acc_b) / (n_b - 1)` is an unbiased estimate of that
variance. The paper states the estimator for a model whose output is constant within a bin.
Here `conf_b` is the bin's mean confidence, so the within-bin spread of the confidences is
ignored; it is small for narrow bins. The debiased value can be negative. The L2 calibration
error is its square root, clipped at zero first. A bin with a single item has no variance
estimate and adds nothing to the debiased value; the report says how many such bins there are.

**Brier score** (Brier, 1950) is `mean((c_i - y_i)^2)`. Murphy (1973) partitions it as

```
Brier = reliability - resolution + uncertainty
reliability = sum_b (n_b / n) * (conf_b - acc_b)^2
resolution  = sum_b (n_b / n) * (acc_b - mean(y))^2
uncertainty = mean(y) * (1 - mean(y))
```

which is exact when the confidence is constant within each bin. Otherwise the report adds a
within-bin term, `Brier - (reliability - resolution + uncertainty)`, equal to the within-bin
variance of the confidences minus twice their within-bin covariance with the outcomes. The
reliability term is the plug-in squared calibration error.

References:

- G. W. Brier (1950). Verification of forecasts expressed in terms of probability. Monthly
  Weather Review 78(1), 1-3.
- A. H. Murphy (1973). A new vector partition of the probability score. Journal of Applied
  Meteorology 12(4), 595-600.
- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger (2017). On calibration of modern neural
  networks. https://arxiv.org/abs/1706.04599
- A. Kumar, P. Liang, T. Ma (2019). Verified uncertainty calibration. NeurIPS 2019.
  https://arxiv.org/abs/1909.10155

## Not implemented yet

- A debiased variant of the ECE itself (only the squared error has one so far).
- Classwise and top-label variants from full probability vectors, log loss, AUROC of
  confidence for correctness, the risk-coverage curve, AURC and accuracy at fixed coverage.
- Bootstrap intervals (clustered when a group column is given), consistency bands for
  reliability diagrams and a test of calibration against them.
- Recalibration (temperature scaling, Platt scaling, isotonic regression, histogram binning)
  fitted on one split and scored on another.
- Parsing of stated confidences such as "85%" or word scales, and digit preference.
- The `diagram`, `recalibrate` and `compare` commands.
- Results on real public data. Every number in this README comes from the simulated example.

## Limitations

- With equal-mass bins the edges come from the same items that are then scored; the
  unbiasedness argument for the debiased estimator assumes bins fixed in advance.
- All binned metrics change with the number of bins. Report the binning with the number.

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha.

Built by Vipul Raj Jha · https://vipul21435.github.io
