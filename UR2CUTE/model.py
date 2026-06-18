import os
import random
import pickle
import copy
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from numpy.lib.stride_tricks import sliding_window_view

from sklearn.base import BaseEstimator


# Internal column names for engineered intermittent-demand channels.
_TSL_COL = "__time_since_last__"
_ROLLMEAN_COL = "__rolling_mean__"


def _build_feature_frame(df, target_col, external_features, window):
    """
    Build the multichannel feature matrix that feeds the CNNs.

    Returns an array of shape (T, C) where the columns (channels) are, in order:
        [target, *external_features, time-since-last-nonzero, rolling-mean]

    The target and external features are passed through as raw time series; the
    last two channels are causal intermittent-demand features computed only from
    information available up to and including each row (no look-ahead leakage).
    """
    external_features = external_features or []
    demand = df[target_col].to_numpy(dtype=float)

    # Time since the most recent non-zero demand (a core intermittency signal).
    tsl = np.empty(len(demand), dtype=float)
    last_nonzero = -1
    for i, value in enumerate(demand):
        if value > 0:
            tsl[i] = 0.0
            last_nonzero = i
        else:
            tsl[i] = (i - last_nonzero) if last_nonzero >= 0 else (i + 1)

    # Causal rolling mean of demand over the lookback window.
    roll_mean = (
        pd.Series(demand).rolling(window, min_periods=1).mean().to_numpy(dtype=float)
    )

    channels = [demand]
    for feature in external_features:
        channels.append(df[feature].to_numpy(dtype=float))
    channels.append(tsl)
    channels.append(roll_mean)

    return np.column_stack(channels)


def _window_data(feature_frame, target_raw, window, forecast_horizon):
    """
    Turn the (T, C) feature frame into CNN-ready temporal windows.

    For each origin time t with a full window of history behind it and a full
    forecast horizon ahead of it:
      - X is the (C, window) slice covering rows [t-window+1 .. t]; the last axis
        is genuine time, so a 1-D convolution slides over consecutive steps.
      - y is the next forecast_horizon raw target values (rows t+1 .. t+horizon).
    """
    n_rows, n_channels = feature_frame.shape
    if n_rows < window + forecast_horizon:
        return (
            np.empty((0, n_channels, window), dtype=np.float32),
            np.empty((0, forecast_horizon), dtype=np.float32),
        )

    # (n_rows - window + 1, C, window); windows[i, c, w] = feature_frame[i + w, c]
    windows = sliding_window_view(feature_frame, window_shape=window, axis=0)
    n_usable = n_rows - window - forecast_horizon + 1
    X = windows[:n_usable]

    # For origin i (last history row = i + window - 1), targets start at i + window.
    horizon_windows = sliding_window_view(target_raw, window_shape=forecast_horizon)
    y = horizon_windows[window : window + n_usable]

    return X.astype(np.float32), y.astype(np.float32)


def _split_train_validation_sequences(X_all, y_all, forecast_horizon, val_fraction=0.1):
    """
    Split windows chronologically without letting train and validation targets overlap.

    Validation windows are taken from the tail of the series, while an embargo of
    `forecast_horizon - 1` windows is left between train and validation samples.
    If the dataset is too small to support a leakage-free split, fall back to
    using the full dataset for both train and validation.
    """
    n_sequences = X_all.shape[0]
    if n_sequences < 2:
        return X_all, y_all, X_all.copy(), y_all.copy()

    n_val = max(1, int(np.ceil(n_sequences * val_fraction)))
    val_start = n_sequences - n_val
    train_end = val_start - forecast_horizon + 1

    if train_end <= 0 or val_start >= n_sequences:
        return X_all, y_all, X_all.copy(), y_all.copy()

    X_train = X_all[:train_end]
    y_train = y_all[:train_end]
    X_val = X_all[val_start:]
    y_val = y_all[val_start:]

    if len(X_train) == 0 or len(X_val) == 0:
        return X_all, y_all, X_all.copy(), y_all.copy()

    return X_train, y_train, X_val, y_val


class _CausalConv1d(nn.Module):
    """
    Dilated causal 1-D convolution: left-pads the input so output step t only
    depends on inputs at steps <= t (no leakage from the future).
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class CNNClassifier(nn.Module):
    """
    Dilated-causal CNN that predicts zero vs. nonzero demand for each horizon step.

    Outputs raw logits (no sigmoid) for use with BCEWithLogitsLoss; apply a sigmoid
    at inference to recover probabilities.
    """
    def __init__(self, n_channels, forecast_horizon, dropout_rate=0.4):
        super().__init__()
        width = 64
        self.block1 = _CausalConv1d(n_channels, width, kernel_size=3, dilation=1)
        self.block2 = _CausalConv1d(width, width, kernel_size=3, dilation=2)
        self.block3 = _CausalConv1d(width, width, kernel_size=3, dilation=4)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(width, 32)
        self.fc2 = nn.Linear(32, forecast_horizon)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, channels, window)
        x = self.relu(self.block1(x))
        x = self.relu(self.block2(x))
        x = self.relu(self.block3(x))
        x = self.pool(x).squeeze(-1)  # global temporal pooling -> (batch, width)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # logits


class CNNRegressor(nn.Module):
    """
    Dilated-causal CNN that predicts demand magnitude for each horizon step.
    """
    def __init__(self, n_channels, forecast_horizon, dropout_rate=0.2):
        super().__init__()
        width = 32
        self.block1 = _CausalConv1d(n_channels, width, kernel_size=3, dilation=1)
        self.block2 = _CausalConv1d(width, width, kernel_size=3, dilation=2)
        self.block3 = _CausalConv1d(width, width, kernel_size=3, dilation=4)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(width, 46)
        self.fc2 = nn.Linear(46, forecast_horizon)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, channels, window)
        x = self.relu(self.block1(x))
        x = self.relu(self.block2(x))
        x = self.relu(self.block3(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class EarlyStopping:
    """
    PyTorch implementation of early stopping
    """
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_state_dict = None
        self.early_stop = False
        self.val_loss_min = np.inf  # Changed from np.Inf to np.inf for NumPy 2.0 compatibility
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.best_state_dict = {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in model.state_dict().items()
        }
        self.val_loss_min = val_loss

    def restore_best_weights(self, model, device):
        if self.best_state_dict is None:
            raise RuntimeError("Early stopping did not capture any model weights.")

        state_dict = {
            key: value.to(device) if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in self.best_state_dict.items()
        }
        model.load_state_dict(state_dict)


class UR2CUTE(BaseEstimator):
    """
    UR2CUTE: Using Repetitively 2 CNNs for Unsteady Timeseries Estimation (two-step/hurdle approach).

    This estimator does direct multi-step forecasting with two dilated-causal CNNs:
      - A CNN classifier predicting zero vs. nonzero demand for each future step.
      - A CNN regressor predicting the quantity (by default trained only on windows
        whose horizon contains at least one nonzero step).

    Both CNNs consume the same multichannel temporal window: the target history plus
    any external covariates and two engineered intermittency channels (time since the
    last nonzero demand and a causal rolling mean). Because the window's last axis is
    genuine time, the dilated causal convolutions slide over consecutive time steps and
    cover long inter-demand gaps cheaply.

    Parameters
    ----------
    n_steps_lag : int
        Length of the lookback window (number of past time steps each CNN sees).
    forecast_horizon : int
        Number of future steps to predict in one pass.
    external_features : list of str or None
        Column names for external covariates (if any).
    epochs : int
        Training epochs for both CNN models.
    batch_size : int
        Batch size for training.
    threshold : float, "auto", or "balanced"
        Probability threshold for classifying zero vs. nonzero demand.
        Default "balanced" tunes the cutoff on the validation split to drive the
        forecast bias toward zero. "auto" uses the proportion of zeros in the
        training data (simple, but tends to over-gate and under-forecast). A float
        is used as a fixed cutoff.
    patience : int
        Patience for EarlyStopping.
    random_seed : int
        Random seed for reproducibility.
    classification_lr : float
        Learning rate for classification model.
    regression_lr : float
        Learning rate for regression model.
    dropout_classification : float
        Dropout rate for the classification model.
    dropout_regression : float
        Dropout rate for the regression model.
    regressor_nonzero_only : bool
        If True (default), the regressor is trained only on sequences where the horizon
        contains at least one non-zero value. Set to False to train the regressor on all
        sequences regardless of demand occurrence.
    verbose : bool
        Whether to print training progress. Default is False (quiet), which suits
        fitting many series in a loop.
    """

    def __init__(
        self,
        n_steps_lag=12,
        forecast_horizon=8,
        external_features=None,
        epochs=100,
        batch_size=32,
        threshold="balanced",
        patience=10,
        random_seed=42,
        classification_lr=0.0021,
        regression_lr=0.0021,
        dropout_classification=0.4,
        dropout_regression=0.2,
        regressor_nonzero_only=True,
        verbose=False
    ):
        self.n_steps_lag = n_steps_lag
        self.forecast_horizon = forecast_horizon
        self.external_features = external_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.patience = patience
        self.random_seed = random_seed
        self.classification_lr = classification_lr
        self.regression_lr = regression_lr
        self.dropout_classification = dropout_classification
        self.dropout_regression = dropout_regression
        self.regressor_nonzero_only = regressor_nonzero_only
        self.verbose = verbose

        # Models will be created in fit()
        self.classifier_ = None
        self.regressor_ = None
        # Per-channel input scaling (fit on training windows only)
        self.channel_min_ = None
        self.channel_max_ = None
        # Target scaling (on the optional log1p scale)
        self.y_min_ = None
        self.y_max_ = None
        self.use_log_target_ = None
        # Fitted dims
        self.n_channels_ = None
        # Fitted threshold (resolved auto/balanced/fixed cutoff)
        self.threshold_ = None

    @property
    def device(self):
        """
        Lazily-resolved compute device (cuda if available, else cpu).

        Kept out of __init__ so that __init__ only assigns the estimator's
        hyperparameters, per the sklearn convention required by clone()/get_params.
        """
        if getattr(self, "_device", None) is None:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self._device

    def _set_random_seeds(self):
        """
        Force reproducible behavior by setting seeds.
        Note: On GPU, some ops may still be non-deterministic.
        """
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _make_loader(self, X, y):
        """Build a reproducible, seeded DataLoader for the given window tensors."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

    def _scale_X(self, X):
        """Per-channel min-max scaling of (samples, channels, window) windows."""
        value_range = self.channel_max_ - self.channel_min_
        value_range = np.where(value_range == 0, 1.0, value_range)
        return (X - self.channel_min_[None, :, None]) / value_range[None, :, None]

    def _scale_y(self, y):
        """Optional log1p transform followed by min-max scaling of the target."""
        t = np.log1p(y) if self.use_log_target_ else y
        value_range = self.y_max_ - self.y_min_
        if value_range == 0:
            value_range = 1.0
        return (t - self.y_min_) / value_range

    def _inverse_scale_y(self, y_scaled):
        """Invert _scale_y back to the original target scale."""
        value_range = self.y_max_ - self.y_min_
        if value_range == 0:
            value_range = 1.0
        t = y_scaled * value_range + self.y_min_
        return np.expm1(t) if self.use_log_target_ else t

    def _regressor_magnitude(self, X):
        """Back-transformed regression magnitudes for windows X."""
        xt = torch.FloatTensor(X).to(self.device)
        self.regressor_.eval()
        with torch.no_grad():
            scaled = np.clip(self.regressor_(xt).cpu().numpy(), 0.0, 1.2)
        return self._inverse_scale_y(scaled)

    def _tune_threshold_balanced(self, X_val, y_val_raw):
        """Pick the occurrence threshold that drives validation bias toward zero
        (ties broken by lower MAE). Cheap: the threshold only gates predict()."""
        xt = torch.FloatTensor(X_val).to(self.device)
        self.classifier_.eval()
        with torch.no_grad():
            prob = torch.sigmoid(self.classifier_(xt)).cpu().numpy()
        qty = self._regressor_magnitude(X_val)

        best_t, best_obj, best_mae = 0.5, np.inf, np.inf
        for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
            preds = np.where(prob > t, np.round(qty), 0.0)
            preds = np.clip(preds, 0.0, None)
            err = preds - y_val_raw
            obj = abs(float(np.mean(err)))
            mae = float(np.mean(np.abs(err)))
            if obj < best_obj - 1e-9 or (abs(obj - best_obj) <= 1e-9 and mae < best_mae):
                best_t, best_obj, best_mae = float(t), obj, mae
        return round(best_t, 2)

    def _train_classifier(self, X_train, y_train, X_val, y_val):
        """
        Train the classification model with BCEWithLogitsLoss (with per-step
        pos_weight to counter the heavy zero-class imbalance).
        """
        train_loader = self._make_loader(X_train, y_train)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Initialize model
        self.classifier_ = CNNClassifier(
            n_channels=self.n_channels_,
            forecast_horizon=self.forecast_horizon,
            dropout_rate=self.dropout_classification
        ).to(self.device)

        # Per-horizon-step positive weight = (# negatives) / (# positives).
        positives = y_train.sum(axis=0)
        negatives = y_train.shape[0] - positives
        pos_weight = negatives / np.clip(positives, 1.0, None)
        pos_weight_tensor = torch.FloatTensor(pos_weight).to(self.device)

        optimizer = optim.Adam(self.classifier_.parameters(), lr=self.classification_lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose)

        for epoch in range(self.epochs):
            self.classifier_.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.classifier_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            self.classifier_.eval()
            with torch.no_grad():
                val_logits = self.classifier_(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()

                # Report accuracy with the same threshold inference will use.
                val_probs = torch.sigmoid(val_logits)
                predicted = (val_probs > self.threshold_).float()
                correct = (predicted == y_val_tensor).float().sum()
                accuracy = correct / (y_val_tensor.size(0) * y_val_tensor.size(1))

            if self.verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')

            early_stopping(val_loss, self.classifier_)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        early_stopping.restore_best_weights(self.classifier_, self.device)

    def _train_regressor(self, X_train, y_train, X_val, y_val):
        """
        Train the regression model with a Huber (SmoothL1) loss, which is more
        robust to the demand spikes typical of intermittent series than plain MSE.
        """
        train_loader = self._make_loader(X_train, y_train)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Initialize model
        self.regressor_ = CNNRegressor(
            n_channels=self.n_channels_,
            forecast_horizon=self.forecast_horizon,
            dropout_rate=self.dropout_regression
        ).to(self.device)

        optimizer = optim.Adam(self.regressor_.parameters(), lr=self.regression_lr)
        criterion = nn.SmoothL1Loss()

        early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose)

        for epoch in range(self.epochs):
            self.regressor_.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.regressor_(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            self.regressor_.eval()
            with torch.no_grad():
                val_outputs = self.regressor_(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if self.verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            early_stopping(val_loss, self.regressor_)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        early_stopping.restore_best_weights(self.regressor_, self.device)

    def _validate_input_data(self, df: pd.DataFrame, target_col: str) -> None:
        """
        Validate input data for fit method.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to validate
        target_col : str
            Name of the target column

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Check minimum data length
        min_required_length = self.n_steps_lag + self.forecast_horizon + 1
        if len(df) < min_required_length:
            raise ValueError(
                f"Insufficient data: need at least {min_required_length} rows "
                f"(n_steps_lag={self.n_steps_lag} + forecast_horizon={self.forecast_horizon} + 1), "
                f"but got {len(df)} rows"
            )

        # Check if target column exists
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Check if external features exist
        if self.external_features:
            missing_features = [f for f in self.external_features if f not in df.columns]
            if missing_features:
                raise ValueError(
                    f"External features {missing_features} not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

        # Check for NaN values in target column
        if df[target_col].isna().any():
            raise ValueError(
                f"Target column '{target_col}' contains NaN values. "
                f"Please handle missing values before fitting."
            )

        # Check for NaN values in external features
        if self.external_features:
            for feature in self.external_features:
                if df[feature].isna().any():
                    raise ValueError(
                        f"External feature '{feature}' contains NaN values. "
                        f"Please handle missing values before fitting."
                    )

    def _validate_predict_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data for predict method.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to validate

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Need at least a full lookback window to forecast from.
        min_required_length = self.n_steps_lag
        if len(df) < min_required_length:
            raise ValueError(
                f"Insufficient data for prediction: need at least {min_required_length} rows "
                f"(n_steps_lag={self.n_steps_lag}), but got {len(df)} rows"
            )

        # Check if target column exists
        if self.target_col_ not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col_}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Check if external features exist
        if self.external_features:
            missing_features = [f for f in self.external_features if f not in df.columns]
            if missing_features:
                raise ValueError(
                    f"External features {missing_features} not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

        # Check for NaN values in the last n_steps_lag rows of target column
        # (these are the ones we'll use for the lookback window).
        target_values = df[self.target_col_].tail(self.n_steps_lag)
        if target_values.isna().any():
            raise ValueError(
                f"Target column '{self.target_col_}' contains NaN values in the last "
                f"{self.n_steps_lag} rows needed for the lookback window. "
                f"Please handle missing values before prediction."
            )

        # Check for NaN values in the last window rows of external features
        if self.external_features:
            window_rows = df[self.external_features].tail(self.n_steps_lag)
            if window_rows.isna().any().any():
                nan_features = window_rows.columns[window_rows.isna().any()].tolist()
                raise ValueError(
                    f"External features {nan_features} contain NaN values in the last "
                    f"{self.n_steps_lag} rows. Please handle missing values before prediction."
                )

    def fit(self, df: pd.DataFrame, target_col: str) -> 'UR2CUTE':
        """
        Fit the UR2CUTE model on a time-series dataframe `df`.

        Expected columns:
          - `target_col`: The main target to forecast.
          - If external_features is not empty, those columns must exist in df.

        Parameters
        ----------
        df : pd.DataFrame
            Time-series data with at least the target column, sorted by time.
        target_col : str
            The name of the column to forecast.

        Returns
        -------
        self : UR2CUTE
            Fitted estimator.

        Raises
        ------
        ValueError
            If input data validation fails.
        """
        # Validate input data
        self._validate_input_data(df, target_col)

        self._set_random_seeds()
        self.target_col_ = target_col

        # 1) Build the multichannel feature frame and window it into temporal samples.
        feature_frame = _build_feature_frame(
            df, target_col, self.external_features, self.n_steps_lag
        )
        target_raw = df[target_col].to_numpy(dtype=float)
        X_all, y_all = _window_data(
            feature_frame, target_raw, self.n_steps_lag, self.forecast_horizon
        )
        n_sequences = X_all.shape[0]
        if n_sequences == 0:
            raise ValueError(
                "Not enough usable sequences after windowing. "
                "Increase the size of your dataset or reduce n_steps_lag/forecast_horizon."
            )

        # Keep the validation tail chronologically separate from train targets.
        X_train_raw, y_train, X_val_raw, y_val = _split_train_validation_sequences(
            X_all, y_all, self.forecast_horizon
        )

        # 2) Per-channel input scaling, fit on training windows only.
        self.channel_min_ = X_train_raw.min(axis=(0, 2))
        self.channel_max_ = X_train_raw.max(axis=(0, 2))
        X_train = self._scale_X(X_train_raw)
        X_val = self._scale_X(X_val_raw)
        self.n_channels_ = X_train.shape[1]

        # 3) Target scaling: log1p (only when non-negative) then min-max, train stats only.
        self.use_log_target_ = bool(y_train.min() >= 0)
        t_train = np.log1p(y_train) if self.use_log_target_ else y_train
        self.y_min_ = float(t_train.min())
        self.y_max_ = float(t_train.max())
        y_train_scaled = self._scale_y(y_train)
        y_val_scaled = self._scale_y(y_val)

        # Occurrence threshold.
        #   "auto"     -> fraction of zeros (simple, but tends to over-gate).
        #   "balanced" -> tuned on the validation set after both CNNs are trained
        #                 to drive the forecast bias toward zero (set below); use a
        #                 neutral placeholder for now so the classifier's reported
        #                 validation accuracy is well-defined.
        #   float      -> used as-is.
        thr_mode = self.threshold.lower() if isinstance(self.threshold, str) else None
        if thr_mode == "auto":
            self.threshold_ = round(np.mean(y_train == 0), 2)
            if self.verbose:
                print(f"Auto threshold set to: {self.threshold_}")
        elif thr_mode == "balanced":
            self.threshold_ = 0.5
        else:
            self.threshold_ = self.threshold

        # Classification target: zero vs. nonzero
        y_train_binary = (y_train > 0).astype(np.float32)
        y_val_binary = (y_val > 0).astype(np.float32)

        # --------------------------
        # Train Classification Model
        # --------------------------
        self._train_classifier(X_train, y_train_binary, X_val, y_val_binary)

        # -----------------------
        # Train Regression Model
        # When regressor_nonzero_only=True (default), train only on samples that have
        # at least one nonzero step in the horizon. Falls back to full dataset if no
        # such samples exist. When False, train on all samples.
        # -----------------------
        if self.regressor_nonzero_only:
            nonzero_mask_train = (y_train.sum(axis=1) > 0)
            nonzero_mask_val = (y_val.sum(axis=1) > 0)

            if not np.any(nonzero_mask_train):
                if self.verbose:
                    print(
                        "No non-zero horizons found in training data; "
                        "training regressor on the full dataset."
                    )
                X_train_reg = X_train
                y_train_reg = y_train_scaled
            else:
                X_train_reg = X_train[nonzero_mask_train]
                y_train_reg = y_train_scaled[nonzero_mask_train]

            if not np.any(nonzero_mask_val):
                X_val_reg = X_val
                y_val_reg = y_val_scaled
            else:
                X_val_reg = X_val[nonzero_mask_val]
                y_val_reg = y_val_scaled[nonzero_mask_val]
        else:
            X_train_reg = X_train
            y_train_reg = y_train_scaled
            X_val_reg = X_val
            y_val_reg = y_val_scaled

        self._train_regressor(X_train_reg, y_train_reg, X_val_reg, y_val_reg)

        # "balanced" threshold: tune on validation to push the forecast bias toward 0.
        if thr_mode == "balanced":
            self.threshold_ = self._tune_threshold_balanced(X_val, y_val)
            if self.verbose:
                print(f"Balanced threshold set to: {self.threshold_}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict the next self.forecast_horizon steps from the *last* window of `df`.

        Steps:
          1) Build the multichannel feature frame for df.
          2) Take the final lookback window as input.
          3) Predict occurrence probability (sigmoid of the classifier logits) per step.
          4) Predict quantity with the regressor, gated by the fitted threshold.

        Parameters
        ----------
        df : pd.DataFrame
            The time-series DataFrame (sorted by time). Must have the same columns as in fit().

        Returns
        -------
        forecast : np.ndarray of shape (forecast_horizon,)
            The integer predictions for each step in the horizon.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If input data validation fails.
        """
        # Check if model has been fitted
        if self.classifier_ is None or self.regressor_ is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        # Validate input data
        self._validate_predict_data(df)

        # Build the feature frame and take the final lookback window: (channels, window).
        feature_frame = _build_feature_frame(
            df, self.target_col_, self.external_features, self.n_steps_lag
        )
        window = feature_frame[-self.n_steps_lag:].T  # (channels, window)
        x_input = window[np.newaxis, :, :].astype(np.float32)
        x_input_scaled = self._scale_X(x_input)

        x_tensor = torch.FloatTensor(x_input_scaled).to(self.device)

        # Occurrence probabilities (sigmoid of logits) for each step.
        self.classifier_.eval()
        with torch.no_grad():
            order_logits = self.classifier_(x_tensor)[0]
            order_prob = torch.sigmoid(order_logits).cpu().numpy()

        # Quantity for each step.
        self.regressor_.eval()
        with torch.no_grad():
            quantity_pred_scaled = self.regressor_(x_tensor)[0].cpu().numpy()

        # The regression head is unbounded, so on large-magnitude or log1p-scaled
        # targets a runaway prediction can explode through the expm1 inverse. Clamp
        # to a small margin above the training target range before inverting: normal
        # forecasts (well inside [0, 1] scaled) are untouched, but absurd magnitudes
        # are prevented.
        quantity_pred_scaled = np.clip(quantity_pred_scaled, 0.0, 1.2)
        quantity_pred = self._inverse_scale_y(quantity_pred_scaled)

        # Combine using fitted threshold
        final_preds = []
        for prob, qty in zip(order_prob, quantity_pred):
            pred = qty if prob > self.threshold_ else 0
            final_preds.append(max(0, round(float(pred))))

        return np.array(final_preds)

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        This saves both PyTorch models, scaling parameters, and all fitted
        attributes needed for prediction.

        Parameters
        ----------
        path : str
            Path to save the model. Should end with .pkl or .pickle

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet or if saving fails.

        Examples
        --------
        >>> model = UR2CUTE()
        >>> model.fit(df, 'target')
        >>> model.save_model('ur2cute_model.pkl')
        """
        if self.classifier_ is None or self.regressor_ is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before save_model()."
            )

        try:
            # Move state dicts to CPU to keep the saved artifact loadable on CPU-only machines
            classifier_state_cpu = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.classifier_.state_dict().items()
            }
            regressor_state_cpu = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.regressor_.state_dict().items()
            }

            model_data = {
                # Hyperparameters
                'n_steps_lag': self.n_steps_lag,
                'forecast_horizon': self.forecast_horizon,
                'external_features': self.external_features,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'threshold': self.threshold,
                'patience': self.patience,
                'random_seed': self.random_seed,
                'classification_lr': self.classification_lr,
                'regression_lr': self.regression_lr,
                'dropout_classification': self.dropout_classification,
                'dropout_regression': self.dropout_regression,
                'regressor_nonzero_only': self.regressor_nonzero_only,
                'verbose': self.verbose,
                # Fitted attributes
                'target_col_': self.target_col_,
                'n_channels_': self.n_channels_,
                'threshold_': self.threshold_,
                'channel_min_': self.channel_min_,
                'channel_max_': self.channel_max_,
                'y_min_': self.y_min_,
                'y_max_': self.y_max_,
                'use_log_target_': self.use_log_target_,
                # Model state dicts
                'classifier_state_dict': classifier_state_cpu,
                'regressor_state_dict': regressor_state_cpu,
                # Device info
                'device_type': self.device.type
            }

            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")

    @classmethod
    def load_model(cls, path: str) -> 'UR2CUTE':
        """
        Load a trained model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file.

        Returns
        -------
        model : UR2CUTE
            The loaded model ready for prediction.

        Raises
        ------
        RuntimeError
            If loading fails.

        Examples
        --------
        >>> model = UR2CUTE.load_model('ur2cute_model.pkl')
        >>> predictions = model.predict(new_df)
        """
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            # Create instance with saved hyperparameters
            model = cls(
                n_steps_lag=model_data['n_steps_lag'],
                forecast_horizon=model_data['forecast_horizon'],
                external_features=model_data['external_features'],
                epochs=model_data['epochs'],
                batch_size=model_data['batch_size'],
                threshold=model_data['threshold'],
                patience=model_data['patience'],
                random_seed=model_data['random_seed'],
                classification_lr=model_data['classification_lr'],
                regression_lr=model_data['regression_lr'],
                dropout_classification=model_data['dropout_classification'],
                dropout_regression=model_data['dropout_regression'],
                # Default to True for backward compatibility with models saved before
                # these fields were persisted.
                regressor_nonzero_only=model_data.get('regressor_nonzero_only', True),
                verbose=model_data.get('verbose', True)
            )

            # Restore fitted attributes
            model.target_col_ = model_data['target_col_']
            model.n_channels_ = model_data['n_channels_']
            model.threshold_ = model_data['threshold_']
            model.channel_min_ = model_data['channel_min_']
            model.channel_max_ = model_data['channel_max_']
            model.y_min_ = model_data['y_min_']
            model.y_max_ = model_data['y_max_']
            model.use_log_target_ = model_data['use_log_target_']

            # Recreate models with correct architecture
            model.classifier_ = CNNClassifier(
                n_channels=model.n_channels_,
                forecast_horizon=model.forecast_horizon,
                dropout_rate=model.dropout_classification
            ).to(model.device)

            model.regressor_ = CNNRegressor(
                n_channels=model.n_channels_,
                forecast_horizon=model.forecast_horizon,
                dropout_rate=model.dropout_regression
            ).to(model.device)

            # Load model weights with device mapping for GPU<->CPU compatibility
            classifier_state = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                              for k, v in model_data['classifier_state_dict'].items()}
            regressor_state = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                             for k, v in model_data['regressor_state_dict'].items()}

            model.classifier_.load_state_dict(classifier_state)
            model.regressor_.load_state_dict(regressor_state)

            # Set models to eval mode
            model.classifier_.eval()
            model.regressor_.eval()

            return model

        except FileNotFoundError:
            raise RuntimeError(f"Model file not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
