import os
import random
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
        ext_vals = df.loc[i, external_features].values if external_features else []
        
        X_list.append(np.concatenate([ext_vals, lag_vals]))
        # Next forecast_horizon steps for the target
        y_seq = df.loc[i+1 : i+forecast_horizon, target_name].values
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
        
        # Calculate size after pooling
        flattened_size = 64 * (n_features // 2)
        
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
        
        # Calculate size after pooling
        flattened_size = 32 * (n_features // 2)
        
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
    threshold : float
        Probability threshold for classifying zero vs. nonzero demand.
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
        dropout_regression=0.2
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
        
        # Early stopping
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path='classifier_checkpoint.pt')
        
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
                
            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')
            
            # Early stopping
            early_stopping(val_loss, self.classifier_)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        # Load the best model
        self.classifier_.load_state_dict(torch.load('classifier_checkpoint.pt'))
        
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
        
        # Early stopping
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path='regressor_checkpoint.pt')
        
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
                
            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            early_stopping(val_loss, self.regressor_)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
        # Load the best model
        self.regressor_.load_state_dict(torch.load('regressor_checkpoint.pt'))

    def fit(self, df, target_col):
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
        """
        self._set_random_seeds()
        self.target_col_ = target_col

        # 1) Generate lag features & drop NaNs
        df_lagged = _generate_lag_features(df, target_col, n_lags=self.n_steps_lag)
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

        # 3) Scale inputs
        self.scaler_X_ = MinMaxScaler()
        X_scaled = self.scaler_X_.fit_transform(X_all)

        self.scaler_y_ = MinMaxScaler()
        y_flat = y_all.flatten().reshape(-1, 1)
        self.scaler_y_.fit(y_flat)
        y_scaled = self.scaler_y_.transform(y_flat).reshape(y_all.shape)

        # For CNN, we want (samples, features, 1)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        self.n_features_ = X_reshaped.shape[1]

        # 4) Time-based split for validation (10%)
        val_split_idx = int(len(X_reshaped) * 0.9)
        X_train = X_reshaped[:val_split_idx]
        y_train = y_all[:val_split_idx]
        X_val = X_reshaped[val_split_idx:]
        y_val = y_all[val_split_idx:]

        y_train_scaled = y_scaled[:val_split_idx]
        y_val_scaled = y_scaled[val_split_idx:]
        
        # Auto threshold: if threshold is set to "auto", calculate it based on the proportion of zeros in y_train
        if isinstance(self.threshold, str) and self.threshold.lower() == "auto":
            computed_threshold = round(np.mean(y_train == 0), 2)
            self.threshold = computed_threshold
            print(f"Auto threshold set to: {self.threshold}")

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

        X_train_reg = X_train[nonzero_mask_train]
        y_train_reg = y_train_scaled[nonzero_mask_train]

        X_val_reg = X_val[nonzero_mask_val]
        y_val_reg = y_val_scaled[nonzero_mask_val]

        self._train_regressor(X_train_reg, y_train_reg, X_val_reg, y_val_reg)

        return self

    def predict(self, df):
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
        """
        target_col = self.target_col_
        
        # Build lag features
        df_lagged = _generate_lag_features(df, target_col, n_lags=self.n_steps_lag)
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

        # Combine
        final_preds = []
        for prob, qty in zip(order_prob, quantity_pred):
            pred = qty if prob > self.threshold else 0
            final_preds.append(max(0, round(pred)))

        return np.array(final_preds)

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
            'dropout_regression': self.dropout_regression
        }

    def set_params(self, **params):
        """
        For sklearn compatibility: sets hyperparameters from a dict.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self