import os
import random
import tempfile
import shutil
import pickle
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler


def _combined_loss(alpha=0.5):
    """
    Custom loss function combining MSE and MAE.
    """
    def loss(y_true, y_pred):
        mse = torch.mean(torch.square(y_true - y_pred))
        mae = torch.mean(torch.abs(y_true - y_pred))
        return alpha * mse + (1 - alpha) * mae
    return loss


def _generate_lag_features(df, column_name, n_lags=1):
    """
    Generate lag features for a given column in the dataframe.
    """
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f"{column_name}_Lag{i}"] = df[column_name].shift(i)
    return df


def _create_multistep_data(df, target_name, external_features, n_steps_lag, forecast_horizon):
    """
    Build multi-step training samples. For each possible row i (up to len(df)-forecast_horizon):
      - The input vector is: [external features] + [lag features from row i]
      - The target is the next forecast_horizon values of target_name (rows i+1 .. i+forecast_horizon).
    """
    X_list = []
    y_list = []
    for i in range(len(df) - forecast_horizon):
        # Lags from current row i
        lag_vals = df.iloc[i][[f"{target_name}_Lag{j}" for j in range(1, n_steps_lag + 1)]].values

        # External features from current row i (if any)
        ext_vals = df.iloc[i][external_features].values if external_features else []

        X_list.append(np.concatenate([ext_vals, lag_vals]))
        # Next forecast_horizon steps for the target
        y_seq = df.iloc[i+1 : i+forecast_horizon+1][target_name].values
        y_list.append(y_seq)
    return np.array(X_list), np.array(y_list)


class CNNClassifier(nn.Module):
    """
    PyTorch CNN model for classification (zero vs. nonzero)
    """
    def __init__(self, n_features, forecast_horizon, dropout_rate=0.4):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        # Calculate size after pooling correctly (handles odd n_features)
        # Conv1d with padding=1 preserves size, MaxPool1d with kernel=2 does floor division
        size_after_pool = n_features // 2
        flattened_size = 64 * size_after_pool

        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, forecast_horizon)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input shape: (batch, features, 1)
        x = x.permute(0, 2, 1)  # PyTorch expects (batch, channels, features)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class CNNRegressor(nn.Module):
    """
    PyTorch CNN model for regression
    """
    def __init__(self, n_features, forecast_horizon, dropout_rate=0.2):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()

        # Calculate size after pooling correctly (handles odd n_features)
        # Conv1d with padding=1 preserves size, MaxPool1d with kernel=2 does floor division
        size_after_pool = n_features // 2
        flattened_size = 32 * size_after_pool

        self.fc1 = nn.Linear(flattened_size, 46)
        self.fc2 = nn.Linear(46, forecast_horizon)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input shape: (batch, features, 1)
        x = x.permute(0, 2, 1)  # PyTorch expects (batch, channels, features)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EarlyStopping:
    """
    PyTorch implementation of early stopping
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # Changed from np.Inf to np.inf for NumPy 2.0 compatibility
        self.delta = delta
        self.path = path
        
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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class UR2CUTE(BaseEstimator):
    """
    UR2CUTE: Using Repetitively 2 CNNs for Unsteady Timeseries Estimation (two-step/hurdle approach).

    This estimator does direct multi-step forecasting with:
      - A CNN-based classification model to predict zero vs. nonzero for each future step.
      - A CNN-based regression model to predict the quantity (only trained on sequences that have
        at least one nonzero step in the horizon).

    Parameters
    ----------
    n_steps_lag : int
        Number of lag features to generate.
    forecast_horizon : int
        Number of future steps to predict in one pass.
    external_features : list of str or None
        Column names for external features (if any).
    epochs : int
        Training epochs for both CNN models.
    batch_size : int
        Batch size for training.
    threshold : float or "auto"
        Probability threshold for classifying zero vs. nonzero demand.
        If "auto", computes threshold based on proportion of zeros in training data.
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
    verbose : bool
        Whether to print training progress. Default is True.
    """

    def __init__(
        self,
        n_steps_lag=3,
        forecast_horizon=8,
        external_features=None,
        epochs=100,
        batch_size=32,
        threshold=0.5,
        patience=10,
        random_seed=42,
        classification_lr=0.0021,
        regression_lr=0.0021,
        dropout_classification=0.4,
        dropout_regression=0.2,
        verbose=True
    ):
        self.n_steps_lag = n_steps_lag
        self.forecast_horizon = forecast_horizon
        self.external_features = external_features if external_features is not None else []
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.patience = patience
        self.random_seed = random_seed
        self.classification_lr = classification_lr
        self.regression_lr = regression_lr
        self.dropout_classification = dropout_classification
        self.dropout_regression = dropout_regression
        self.verbose = verbose

        # Set device (cuda if available, else cpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Models will be created in fit()
        self.classifier_ = None
        self.regressor_ = None
        # Scalers
        self.scaler_X_ = None
        self.scaler_y_ = None
        # Fitted dims
        self.n_features_ = None
        # Fitted threshold (for auto threshold computation)
        self.threshold_ = None
        # Temporary directory for checkpoints
        self._temp_dir = None

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

    def _train_classifier(self, X_train, y_train, X_val, y_val):
        """
        Train the classification model
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self.classifier_ = CNNClassifier(
            n_features=self.n_features_,
            forecast_horizon=self.forecast_horizon,
            dropout_rate=self.dropout_classification
        ).to(self.device)

        # Initialize optimizer and loss
        optimizer = optim.Adam(self.classifier_.parameters(), lr=self.classification_lr)
        criterion = nn.BCELoss()

        # Early stopping with unique checkpoint path
        classifier_checkpoint_path = os.path.join(self._temp_dir, 'classifier_checkpoint.pt')
        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=self.verbose,
            path=classifier_checkpoint_path
        )

        # Training loop
        for epoch in range(self.epochs):
            self.classifier_.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.classifier_(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            self.classifier_.eval()
            with torch.no_grad():
                val_outputs = self.classifier_(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

                # Calculate accuracy
                predicted = (val_outputs > 0.5).float()
                correct = (predicted == y_val_tensor).float().sum()
                accuracy = correct / (y_val_tensor.size(0) * y_val_tensor.size(1))

            if self.verbose:
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')

            # Early stopping
            early_stopping(val_loss, self.classifier_)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        # Load the best model with error handling
        try:
            self.classifier_.load_state_dict(torch.load(classifier_checkpoint_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load classifier checkpoint: {e}")
        
    def _train_regressor(self, X_train, y_train, X_val, y_val):
        """
        Train the regression model
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self.regressor_ = CNNRegressor(
            n_features=self.n_features_,
            forecast_horizon=self.forecast_horizon,
            dropout_rate=self.dropout_regression
        ).to(self.device)

        # Initialize optimizer and loss
        optimizer = optim.Adam(self.regressor_.parameters(), lr=self.regression_lr)
        criterion = nn.MSELoss()

        # Early stopping with unique checkpoint path
        regressor_checkpoint_path = os.path.join(self._temp_dir, 'regressor_checkpoint.pt')
        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=self.verbose,
            path=regressor_checkpoint_path
        )

        # Training loop
        for epoch in range(self.epochs):
            self.regressor_.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
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

            # Early stopping
            early_stopping(val_loss, self.regressor_)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

        # Load the best model with error handling
        try:
            self.regressor_.load_state_dict(torch.load(regressor_checkpoint_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load regressor checkpoint: {e}")

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

        # Check minimum data length for prediction (need at least n_steps_lag rows)
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
        # (these are the ones we'll use for lag features)
        target_values = df[self.target_col_].tail(self.n_steps_lag)
        if target_values.isna().any():
            raise ValueError(
                f"Target column '{self.target_col_}' contains NaN values in the last "
                f"{self.n_steps_lag} rows needed for lag features. "
                f"Please handle missing values before prediction."
            )

        # Check for NaN values in the last row of external features
        if self.external_features:
            last_row = df[self.external_features].iloc[-1]
            if last_row.isna().any():
                nan_features = last_row[last_row.isna()].index.tolist()
                raise ValueError(
                    f"External features {nan_features} contain NaN values in the last row. "
                    f"Please handle missing values before prediction."
                )

    def fit(self, df: pd.DataFrame, target_col: str) -> 'UR2CUTE':
        """
        Fit the UR2CUTE model on a time-series dataframe `df`.

        Expected columns:
          - `target_col`: The main target to forecast.
          - If external_features is not empty, those columns must exist in df.
          - We'll generate lag features for `target_col`.

        Parameters
        ----------
        df : pd.DataFrame
            Time-series data with at least the target column. Must be sorted by time in advance
            (or you can ensure we do it here).
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
        RuntimeError
            If model training fails.
        """
        # Validate input data
        self._validate_input_data(df, target_col)

        self._set_random_seeds()
        self.target_col_ = target_col

        # Create temporary directory for checkpoints
        self._temp_dir = tempfile.mkdtemp()

        try:
            # 1) Generate lag features & drop NaNs (work on copy to avoid modifying input)
            df_copy = df.copy()
            df_lagged = _generate_lag_features(df_copy, target_col, n_lags=self.n_steps_lag)
            df_lagged.dropna(inplace=True)
            df_lagged.reset_index(drop=True, inplace=True)

            # 2) Create multi-step training data
            X_all, y_all = _create_multistep_data(
                df_lagged,
                target_col,
                self.external_features,
                self.n_steps_lag,
                self.forecast_horizon
            )
            # shape: X_all -> (samples, features), y_all -> (samples, forecast_horizon)
            n_sequences = X_all.shape[0]
            if n_sequences == 0:
                raise ValueError(
                    "Not enough usable sequences after generating lag features. "
                    "Increase the size of your dataset or reduce n_steps_lag/forecast_horizon."
                )

            # Time-based split for validation (10%)
            val_split_idx = int(n_sequences * 0.9)
            X_train_raw = X_all[:val_split_idx]
            y_train = y_all[:val_split_idx]
            X_val_raw = X_all[val_split_idx:]
            y_val = y_all[val_split_idx:]

            # Ensure both splits contain at least one sequence
            if len(X_train_raw) == 0:
                X_train_raw = X_all
                y_train = y_all
            if len(X_val_raw) == 0:
                X_val_raw = X_train_raw.copy()
                y_val = y_train.copy()

            # 3) Scale inputs using training data statistics only
            self.scaler_X_ = MinMaxScaler()
            X_train_scaled = self.scaler_X_.fit_transform(X_train_raw)
            X_val_scaled = self.scaler_X_.transform(X_val_raw)

            self.scaler_y_ = MinMaxScaler()
            y_train_flat = y_train.flatten().reshape(-1, 1)
            self.scaler_y_.fit(y_train_flat)
            y_train_scaled = self.scaler_y_.transform(y_train_flat).reshape(y_train.shape)
            y_val_scaled = self.scaler_y_.transform(y_val.flatten().reshape(-1, 1)).reshape(y_val.shape)

            # For CNN, we want (samples, features, 1)
            X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_val = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
            self.n_features_ = X_train.shape[1]

            # Auto threshold: if threshold is set to "auto", calculate it and store as threshold_
            if isinstance(self.threshold, str) and self.threshold.lower() == "auto":
                self.threshold_ = round(np.mean(y_train == 0), 2)
                if self.verbose:
                    print(f"Auto threshold set to: {self.threshold_}")
            else:
                # Use the provided threshold value
                self.threshold_ = self.threshold

            # Classification target: zero vs. nonzero
            y_train_binary = (y_train > 0).astype(float)  # shape: (samples, horizon)
            y_val_binary = (y_val > 0).astype(float)

            # --------------------------
            # Train Classification Model
            # --------------------------
            self._train_classifier(X_train, y_train_binary, X_val, y_val_binary)

            # -----------------------
            # Train Regression Model
            # Train only on samples that have at least one nonzero step in the horizon
            # OR you can filter for sum > 0, or for any > 0, etc.
            # We'll use sum > 0 here.
            # -----------------------
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

            self._train_regressor(X_train_reg, y_train_reg, X_val_reg, y_val_reg)

        finally:
            # Clean up temporary directory
            if self._temp_dir and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
                self._temp_dir = None

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict the next self.forecast_horizon steps from the *last* row of the input DataFrame.

        We'll:
          1) Generate lag features for df.
          2) Take the final row (post-lag) as input.
          3) Predict classification (zero vs. nonzero) for each horizon step.
          4) Predict regression quantity, but only if classification > threshold.

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
            If the model has not been fitted yet or if prediction fails.
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

        try:
            target_col = self.target_col_

            # Build lag features (work on copy to avoid modifying input)
            df_copy = df.copy()
            df_lagged = _generate_lag_features(df_copy, target_col, n_lags=self.n_steps_lag)
            df_lagged.dropna(inplace=True)

            # Take the final row to forecast from
            last_idx = df_lagged.index[-1]
            lag_vals = df_lagged.loc[last_idx, [f"{target_col}_Lag{j}" for j in range(1, self.n_steps_lag + 1)]].values

            if self.external_features:
                ext_vals = df_lagged.loc[last_idx, self.external_features].values
            else:
                ext_vals = []

            x_input = np.concatenate([ext_vals, lag_vals]).reshape(1, -1)
            x_input_scaled = self.scaler_X_.transform(x_input)
            x_input_reshaped = x_input_scaled.reshape((1, x_input_scaled.shape[1], 1))

            # Convert to PyTorch tensor
            x_tensor = torch.FloatTensor(x_input_reshaped).to(self.device)

            # Classification (probabilities for each step)
            self.classifier_.eval()
            with torch.no_grad():
                order_prob = self.classifier_(x_tensor)[0].cpu().numpy()

            # Regression (quantity for each step)
            self.regressor_.eval()
            with torch.no_grad():
                quantity_pred_scaled = self.regressor_(x_tensor)[0].cpu().numpy()

            quantity_pred = self.scaler_y_.inverse_transform(quantity_pred_scaled.reshape(-1, 1)).flatten()

            # Combine using fitted threshold
            final_preds = []
            for prob, qty in zip(order_prob, quantity_pred):
                pred = qty if prob > self.threshold_ else 0
                final_preds.append(max(0, round(pred)))

            return np.array(final_preds)

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def get_params(self, deep=True):
        """
        For sklearn compatibility: returns the hyperparameters as a dict.
        """
        return {
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
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """
        For sklearn compatibility: sets hyperparameters from a dict.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        This saves both PyTorch models, scalers, and all fitted attributes
        needed for prediction.

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
                'verbose': self.verbose,
                # Fitted attributes
                'target_col_': self.target_col_,
                'n_features_': self.n_features_,
                'threshold_': self.threshold_,
                # Model state dicts
                'classifier_state_dict': classifier_state_cpu,
                'regressor_state_dict': regressor_state_cpu,
                # Scalers
                'scaler_X_': self.scaler_X_,
                'scaler_y_': self.scaler_y_,
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
                verbose=model_data.get('verbose', True)  # Default to True for backward compatibility
            )

            # Restore fitted attributes
            model.target_col_ = model_data['target_col_']
            model.n_features_ = model_data['n_features_']
            model.threshold_ = model_data['threshold_']
            model.scaler_X_ = model_data['scaler_X_']
            model.scaler_y_ = model_data['scaler_y_']

            # Recreate models with correct architecture
            model.classifier_ = CNNClassifier(
                n_features=model.n_features_,
                forecast_horizon=model.forecast_horizon,
                dropout_rate=model.dropout_classification
            ).to(model.device)

            model.regressor_ = CNNRegressor(
                n_features=model.n_features_,
                forecast_horizon=model.forecast_horizon,
                dropout_rate=model.dropout_regression
            ).to(model.device)

            # Load model weights with device mapping for GPU→CPU compatibility
            # Convert state dicts to current device to handle GPU→CPU or CPU→GPU transfers
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
