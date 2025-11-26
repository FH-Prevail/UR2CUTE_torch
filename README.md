# UR2CUTE

**Using Repetitively 2 CNNs for Unsteady Timeseries Estimation**

UR2CUTE is a specialized forecasting model designed for intermittent time series data. By employing a dual CNN approach with PyTorch, it effectively addresses the challenges of predicting both the occurrence and magnitude of demand in irregular time series patterns.

## 📋 Overview

Intermittent demand forecasting presents unique challenges due to irregular and unpredictable demand patterns, characterized by periods of zero demand followed by random non-zero demand. Traditional forecasting methods often perform poorly on such data.

UR2CUTE employs a two-step approach:
1. A CNN-based classification model predicts demand occurrence (zero vs. non-zero)
2. A CNN-based regression model estimates the magnitude of demand

This dual-phase approach significantly improves forecasting accuracy for intermittent demand, particularly in predicting periods of zero demand.

## 🔍 Features

- **PyTorch Implementation**: Uses PyTorch's efficient tensor operations and GPU acceleration
- **Two-Step Prediction Process**: Separate models for order occurrence and quantity prediction
- **Temporal Pattern Recognition**: CNNs effectively capture temporal patterns in intermittent data
- **Lag Feature Generation**: Automatically creates lagged features to capture historical dependencies
- **Custom Loss Functions**: Optimized loss functions for each prediction task
- **Sklearn Compatibility**: Follows scikit-learn API conventions for easy integration
- **Direct Multi-Step Forecasting**: Predicts multiple future time steps in one pass
- **Automatic Device Selection**: Utilizes GPU acceleration when available
- **Model Persistence**: Save and load trained models with `save_model()` and `load_model()`
- **Auto Threshold**: Automatically compute optimal threshold from training data
- **Production Ready**: Comprehensive input validation and error handling
- **Type Hints**: Full type annotation support for better IDE integration

## 📦 Dependencies

- Python 3.7+
- PyTorch 1.7+
- NumPy
- pandas
- scikit-learn

## 🛠️ Installation

### From PyPI (Recommended)

```bash
pip install UR2CUTE
```

### From Source

```bash
# Clone the repository
git clone https://github.com/FH-Prevail/UR2CUTE_torch.git
cd UR2CUTE_torch

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev]"  # For development
pip install -e ".[test]"  # For testing
pip install -e ".[docs]"  # For documentation
```

### Verify Installation

```python
from UR2CUTE import UR2CUTE
print(UR2CUTE.__module__)  # Should print 'UR2CUTE.model'
```

## 📊 Quick Start

```python
import pandas as pd
import torch
from UR2CUTE import UR2CUTE

# Load time series data
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=50, freq='W'),
    'target': [0, 5, 0, 0, 12, 0, 0, 0, 7, 0, ...],  # Intermittent data
    'feat1': [...],  # Optional external features
    'feat2': [...]   # Optional external features
})

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = UR2CUTE(
    n_steps_lag=3,
    forecast_horizon=4,
    external_features=['feat1', 'feat2']
)

# Fit model
model.fit(df, target_col='target')

# Make predictions for the next forecast_horizon steps
predictions = model.predict(df)
print("Predicted values:", predictions)
```

## 🔧 Parameters

| Parameter                | Description                                                                                                                                   | Default |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `n_steps_lag`            | Number of lag features to generate                                                                                                            | 3       |
| `forecast_horizon`       | Number of future steps to predict                                                                                                             | 8       |
| `external_features`      | List of column names for external features                                                                                                    | None    |
| `epochs`                 | Training epochs for both CNN models                                                                                                           | 100     |
| `batch_size`             | Batch size for training                                                                                                                       | 32      |
| `threshold`              | Probability threshold for classifying zero vs. non-zero. Can be a float or `"auto"` to automatically compute the threshold from training data | 0.5     |
| `patience`               | Patience for EarlyStopping                                                                                                                    | 10      |
| `random_seed`            | Random seed for reproducibility                                                                                                               | 42      |
| `classification_lr`      | Learning rate for classification model                                                                                                        | 0.0021  |
| `regression_lr`          | Learning rate for regression model                                                                                                            | 0.0021  |
| `dropout_classification` | Dropout rate for classification model                                                                                                         | 0.4     |
| `dropout_regression`     | Dropout rate for regression model                                                                                                             | 0.2     |
| `verbose`                | Whether to print training progress                                                                                                            | True    |

## 📝 Methods

### fit(df, target_col)

Fits the UR2CUTE model on the time series data.

**Parameters:**
- `df` (pandas.DataFrame): Time series data with at least the target column. Must be sorted by time.
- `target_col` (str): The name of the column to forecast.

**Returns:**
- The fitted UR2CUTE model instance.

### predict(df)

Predicts the next `forecast_horizon` steps from the input DataFrame.

**Parameters:**
- `df` (pandas.DataFrame): Time series data with the same columns as used in fit(). Must be sorted by time.

**Returns:**
- numpy.ndarray: The predictions for each step in the horizon.

### save_model(path)

Save the trained model to disk for later use.

**Parameters:**
- `path` (str): Path where the model should be saved (e.g., 'my_model.pkl')

**Example:**
```python
model.fit(df, 'target')
model.save_model('ur2cute_model.pkl')
```

### load_model(path) [classmethod]

Load a previously saved model from disk.

**Parameters:**
- `path` (str): Path to the saved model file

**Returns:**
- UR2CUTE: A loaded model instance ready for prediction

**Example:**
```python
model = UR2CUTE.load_model('ur2cute_model.pkl')
predictions = model.predict(new_data)
```

## 🚀 Advanced Usage

### Auto Threshold

Let the model automatically compute the optimal threshold based on your training data:

```python
model = UR2CUTE(
    n_steps_lag=3,
    forecast_horizon=4,
    threshold='auto'  # Automatically computed from training data
)
model.fit(df, 'target')
# The computed threshold is stored in model.threshold_
print(f"Computed threshold: {model.threshold_}")
```

### Silent Training

Disable training output for production environments:

```python
model = UR2CUTE(
    n_steps_lag=3,
    forecast_horizon=4,
    verbose=False  # No training output
)
model.fit(df, 'target')
```

### Model Persistence Workflow

Save and deploy models in production:

```python
# Training phase
model = UR2CUTE(n_steps_lag=3, forecast_horizon=4)
model.fit(train_df, 'target')
model.save_model('production_model.pkl')

# Deployment phase (different script/server)
from UR2CUTE import UR2CUTE
model = UR2CUTE.load_model('production_model.pkl')
predictions = model.predict(new_data)
```

### Using External Features

Include additional predictive features:

```python
df = pd.DataFrame({
    'target': [...],
    'promotion': [0, 1, 0, 1, ...],  # Promotional events
    'price': [10.5, 9.99, 10.5, ...],  # Price changes
    'day_of_week': [1, 2, 3, 4, ...]  # Temporal features
})

model = UR2CUTE(
    n_steps_lag=3,
    forecast_horizon=4,
    external_features=['promotion', 'price', 'day_of_week']
)
model.fit(df, 'target')
```

## 🔍 How It Works

1. **Data Preprocessing**:
   - Aggregates demand data (e.g., daily to weekly)
   - Generates lag features to capture historical patterns
   - Validates input data for quality and completeness

2. **Model Architecture**:
   - **Classification Model**: CNN architecture with convolutional layers, max pooling, and dropout
   - **Regression Model**: Similar CNN architecture optimized for quantity prediction

3. **Training Process**:
   - Models are trained using PyTorch's DataLoader for efficient batch processing
   - Early stopping prevents overfitting by monitoring validation loss
   - Uses Adam optimizer with customizable learning rates
   - Temporary checkpoints are automatically cleaned up

4. **Prediction Process**:
   - Classification model predicts if demand will occur (probability > threshold)
   - Regression model predicts the magnitude of demand
   - Final prediction combines both models' outputs

## 🏆 Performance

UR2CUTE outperforms traditional forecasting techniques including:
- Croston's method
- XGBoost
- Random Forest
- ETR
- Prophet
- AutoARIMA

Particularly for predicting intermittent demand, UR2CUTE shows significant improvements in:
- Mean Absolute Error % (MAE%)
- Root Mean Square Error % (RMSE%)
- R-squared values

## 📚 Citation

If you use UR2CUTE in your research, please cite:

```
@article{mirshahi2024intermittent,
  title={Intermittent Time Series Demand Forecasting Using Dual Convolutional Neural Networks},
  author={Mirshahi, Sina and Brandtner, Patrick and Kom{\'i}nkov{\'a} Oplatkov{\'a}, Zuzana},
  journal={MENDEL — Soft Computing Journal},
  volume={30},
  number={1},
  year={2024},
  publisher={MENDEL Journal}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Sina Mirshahi
- Patrick Brandtner
- Zuzana Komínková Oplatková
- Taha Falatouri
- Mehran Naseri
- Farzaneh Darbanian

## 🙏 Acknowledgments

This research was conducted at:
- Department of Informatics and Artificial Intelligence, Tomas Bata
- Department for Logistics, University of Applied Sciences Upper Austria, Steyr
- Josef Ressel-Centre for Predictive Value Network Intelligence, Steyr