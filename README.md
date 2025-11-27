# UR2CUTE

Using Repetitively 2 CNNs for Unsteady Timeseries Estimation. UR2CUTE is a dual-stage, PyTorch powered model dedicated to intermittent demand forecasting. A classifier estimates demand occurrence while a regressor predicts magnitude, allowing the library to focus on the sparse structure typical of slow-moving inventory.

## Overview

Intermittent demand is dominated by long zero stretches punctuated by irregular spikes. Traditional statistical models struggle to capture both the timing and the size of those bursts. UR2CUTE tackles the problem with a hurdle-style architecture:

1. A CNN classifier predicts the probability of non-zero demand for each step in the forecast horizon.
2. A CNN regressor estimates the corresponding quantities.
3. Final forecasts combine the two outputs through an adaptive threshold so the regressor only contributes when demand is likely.

The estimator follows the scikit-learn API, includes thorough input validation, and automatically selects CPU or GPU devices.

## Features

- Pure PyTorch implementation with GPU support when available.
- Direct multi-step forecasting: predicts the entire horizon in a single forward pass.
- Automatic lag generation plus optional external covariates.
- Customizable hyperparameters (epochs, batch size, independent learning rates, dropout).
- Auto-threshold mode that derives an occurrence cutoff from the training set.
- Early stopping with persistent checkpoints stored in a temporary directory.
- Reproducible results through explicit random seed management.
- Model persistence through `save_model` and `load_model`.
- Complete type hints and packaged type information (`py.typed`).

## Dependencies

- Python 3.7 or newer
- PyTorch 1.7+
- NumPy
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
    n_steps_lag=3,
    forecast_horizon=4,
    external_features=["promo", "price"],
    threshold="auto",
)
model.fit(data, target_col="target")
print(model.predict(data))
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `n_steps_lag` | Number of lag features to generate. | 3 |
| `forecast_horizon` | Number of future periods predicted per call. | 8 |
| `external_features` | Optional list of column names used as exogenous inputs. | `None` |
| `epochs` | Training epochs for both CNN models. | 100 |
| `batch_size` | Training batch size. | 32 |
| `threshold` | Manual probability threshold or `"auto"` to derive from training data. | 0.5 |
| `patience` | Early stopping patience (epochs). | 10 |
| `random_seed` | Global random seed applied to NumPy, Python, and PyTorch. | 42 |
| `classification_lr` | Learning rate for the classifier. | 0.0021 |
| `regression_lr` | Learning rate for the regressor. | 0.0021 |
| `dropout_classification` | Dropout applied inside the classifier. | 0.4 |
| `dropout_regression` | Dropout applied inside the regressor. | 0.2 |
| `verbose` | Enables progress output and early-stopping logs. | `True` |

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

### Silent Training

```python
model = UR2CUTE(verbose=False)
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

1. **Preprocessing** – validates the input frame, generates lag features, creates multi-step samples, and splits chronologically into train and validation partitions.
2. **Scaling** – fits separate MinMax scalers on the training set and applies them to validation and inference data, preventing validation leakage.
3. **Classification Stage** – trains a CNN with sigmoid output and BCE loss to estimate the probability of demand for each future horizon step.
4. **Regression Stage** – trains a CNN regressor with MSE loss on samples that exhibit demand; if no such sequences exist, the model safely falls back to the full dataset.
5. **Inference** – transforms the latest observed sequence, runs both networks, rescales quantities, and zeros out forecasts whose probability falls below the stored threshold.

## Performance

Internal benchmarks show UR2CUTE outperforming Croston, AutoARIMA, Prophet, gradient boosted trees, and random forests on sparse demand series, especially in MAE% and RMSE%. Improvements stem from the dedicated occurrence model, lagged covariates, and the ability to learn temporal filters tuned to each dataset.

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
