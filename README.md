# UR2CUTE

Using Repetitively 2 CNNs for Unsteady Timeseries Estimation. UR2CUTE is a dual-stage, PyTorch powered model dedicated to intermittent demand forecasting. A classifier estimates demand occurrence while a regressor predicts magnitude, allowing the library to focus on the sparse structure typical of slow-moving inventory.

## What's New in 2.0

Version 2.0.0 keeps the two-CNN hurdle design but overhauls how the networks see the data. **This is a breaking release.**

- **Multichannel temporal windows** replace the old flat lag vector. Each CNN now receives `(channels, window)` where the window axis is genuine time: target history, optional covariates, and engineered intermittency channels (time-since-last-demand, causal rolling mean).
- **Dilated causal convolutions (TCN-style)** so the convolutions slide over consecutive time steps and cover long inter-demand gaps cheaply.
- **Better training objectives:** `BCEWithLogitsLoss` with per-step `pos_weight` for the classifier (heavy zero-class imbalance) and a Huber (SmoothL1) loss for the regressor (robust to spikes).
- **Smarter scaling:** per-channel min-max plus an automatic `log1p` transform on non-negative targets.

Migration notes:

- **Models saved with 1.x cannot be loaded in 2.0** — retrain and re-save.
- The default `n_steps_lag` changed from `3` to `12` (it is now a lookback window, not a count of lag columns). Set it explicitly to pin behaviour.

## What's New in 2.2

Better out-of-the-box defaults, informed by benchmarking on real intermittent data:

- **`threshold` now defaults to `"balanced"`** (was `0.5`). It tunes the occurrence
  cutoff on the validation split to drive forecast bias toward zero — the previous
  fixed/`"auto"` cutoffs systematically under-forecast. Adds no training cost.
- **`verbose` now defaults to `False`** (was `True`), which suits fitting many series
  in a loop. Pass `verbose=True` to restore per-epoch logging.

Both are behaviour changes for code relying on the old defaults; set the parameters
explicitly to pin previous behaviour.

## Overview

Intermittent demand is dominated by long zero stretches punctuated by irregular spikes. Traditional statistical models struggle to capture both the timing and the size of those bursts. UR2CUTE tackles the problem with a hurdle-style architecture:

1. A dilated-causal CNN classifier predicts the probability of non-zero demand for each step in the forecast horizon.
2. A dilated-causal CNN regressor estimates the corresponding quantities.
3. Final forecasts combine the two outputs through an adaptive threshold so the regressor only contributes when demand is likely.

Both CNNs consume the same multichannel temporal window — target history, optional covariates, and engineered intermittency channels — so the convolutions slide over a genuine time axis and capture long inter-demand gaps cheaply.

The estimator follows the scikit-learn API, includes thorough input validation, and automatically selects CPU or GPU devices.

## Features

- Pure PyTorch implementation with GPU support when available.
- Direct multi-step forecasting: predicts the entire horizon in a single forward pass.
- Multichannel temporal windows: target history, optional external covariates, and engineered intermittency channels (time-since-last-demand, causal rolling mean).
- Dilated causal convolutions (TCN-style) so each CNN slides over a genuine time axis and cheaply covers long inter-demand gaps.
- Customizable hyperparameters (epochs, batch size, independent learning rates, dropout).
- Occurrence threshold modes: a fixed value, `"auto"` (fraction of zeros), or `"balanced"` (tuned on validation to drive the forecast bias toward zero).
- Early stopping that restores the best weights, kept in memory during training.
- Reproducible results through explicit random seed management.
- Model persistence through `save_model` and `load_model`.
- Complete type hints and packaged type information (`py.typed`).

## Dependencies

- Python 3.7 or newer
- PyTorch 1.7+
- NumPy 1.20+ (for `sliding_window_view`)
- pandas
- scikit-learn

## Installation

### From PyPI

```bash
pip install UR2CUTE
```

### From Source

```bash
git clone https://github.com/FH-Prevail/UR2CUTE_torch.git
cd UR2CUTE_torch
pip install -e .

# Optional extras
pip install -e ".[dev]"
pip install -e ".[test]"
pip install -e ".[docs]"
```

### Verify Installation

```python
from UR2CUTE import UR2CUTE
print(UR2CUTE.__module__)
```

## Quick Start

```python
import pandas as pd
import torch
from UR2CUTE import UR2CUTE

data = pd.DataFrame(
    {
        "date": pd.date_range("2023-01-01", periods=50, freq="W"),
        "target": [0, 5, 0, 0, 12, 0, 0, 0, 7, 0] * 5,
        "promo": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0] * 5,
        "price": [10.0, 9.5, 9.5, 9.5, 10.0, 10.0, 10.0, 9.8, 9.8, 9.8] * 5,
    }
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UR2CUTE(
    n_steps_lag=12,
    forecast_horizon=4,
    external_features=["promo", "price"],
    threshold="balanced",
)
model.fit(data, target_col="target")
print(model.predict(data))
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `n_steps_lag` | Length of the lookback window (past time steps each CNN sees). | 12 |
| `forecast_horizon` | Number of future periods predicted per call. | 8 |
| `external_features` | Optional list of column names used as exogenous inputs. | `None` |
| `epochs` | Training epochs for both CNN models. | 100 |
| `batch_size` | Training batch size. | 32 |
| `threshold` | Occurrence cutoff: a float, `"auto"` (fraction of zeros), or `"balanced"` (tuned on validation to minimize forecast bias). | `"balanced"` |
| `patience` | Early stopping patience (epochs). | 10 |
| `random_seed` | Global random seed applied to NumPy, Python, and PyTorch. | 42 |
| `classification_lr` | Learning rate for the classifier. | 0.0021 |
| `regression_lr` | Learning rate for the regressor. | 0.0021 |
| `dropout_classification` | Dropout applied inside the classifier. | 0.4 |
| `dropout_regression` | Dropout applied inside the regressor. | 0.2 |
| `regressor_nonzero_only` | When `True`, the regressor is trained only on sequences where the forecast horizon contains at least one non-zero value. Set to `False` to train the regressor on all sequences. | `True` |
| `verbose` | Enables progress output and early-stopping logs. | `False` |

## Usage Patterns

### Auto Threshold

```python
model = UR2CUTE(threshold="auto")
model.fit(df, "target")
print(model.threshold_)
```

### External Features

```python
covariates = ["promotion", "price", "weekday"]
model = UR2CUTE(external_features=covariates)
model.fit(df, "target")
```

### Progress Logging

Training is silent by default; pass `verbose=True` for per-epoch loss and
early-stopping logs.

```python
model = UR2CUTE(verbose=True)
model.fit(df, "target")
```

### Model Persistence

```python
trained = UR2CUTE().fit(train_df, "target")
trained.save_model("production_model.pkl")

loaded = UR2CUTE.load_model("production_model.pkl")
preds = loaded.predict(new_df)
```

## How It Works

1. **Preprocessing** – validates the input frame, builds the multichannel feature matrix (target + covariates + engineered intermittency channels), slides it into multi-step temporal windows, and splits chronologically into train and validation partitions.
2. **Scaling** – fits per-channel min-max scaling on the training windows only, and applies a `log1p` transform to non-negative targets to tame spikes before scaling, preventing validation leakage.
3. **Classification Stage** – trains a dilated-causal CNN with `BCEWithLogitsLoss` and per-step `pos_weight` to estimate the probability of demand for each future horizon step under heavy zero-class imbalance.
4. **Regression Stage** – trains a dilated-causal CNN regressor with a Huber (SmoothL1) loss, robust to demand spikes. By default (`regressor_nonzero_only=True`) only sequences where the horizon contains at least one non-zero value are used, keeping the regressor focused on demand magnitude; if no such sequences exist it falls back to the full dataset. Set `regressor_nonzero_only=False` to train on all sequences instead.
5. **Inference** – transforms the latest observed window, runs both networks, rescales quantities, and zeros out forecasts whose occurrence probability falls below the stored threshold.

## Performance

Internal benchmarks show UR2CUTE outperforming Croston, AutoARIMA, Prophet, gradient boosted trees, and random forests on sparse demand series, especially in MAE% and RMSE%. Improvements stem from the dedicated occurrence model, the multichannel temporal window with engineered intermittency signals, and the dilated causal filters tuned to each dataset.

## Citation

```
@article{mirshahi2024intermittent,
  title={Intermittent Time Series Demand Forecasting Using Dual Convolutional Neural Networks},
  author={Mirshahi, Sina and Brandtner, Patrick and Kominkova Oplatkova, Zuzana},
  journal={MENDEL -- Soft Computing Journal},
  volume={30},
  number={1},
  year={2024},
  publisher={MENDEL Journal}
}
```

## License

UR2CUTE is released under the MIT License. See `LICENSE` for the full text.

## Contributors

- Sina Mirshahi
- Patrick Brandtner
- Zuzana Kominkova Oplatkova
- Taha Falatouri
- Mehran Naseri
- Farzaneh Darbanian

## Acknowledgments

This work was carried out at:

- Department of Informatics and Artificial Intelligence, Tomas Bata University
- Department for Logistics, University of Applied Sciences Upper Austria, Steyr
- Josef Ressel-Centre for Predictive Value Network Intelligence, Steyr
