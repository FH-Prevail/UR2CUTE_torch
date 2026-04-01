import os
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from UR2CUTE import UR2CUTE
from UR2CUTE.model import (
    _create_multistep_data,
    _generate_lag_features,
    _split_train_validation_sequences,
)


def _workspace_artifact_path(filename):
    return os.path.join(os.getcwd(), filename)


def test_fit_and_predict_runs_end_to_end():
    rows = 30
    df = pd.DataFrame(
        {
            "target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 3,
            "promo": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0] * 3,
        }
    )
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=3,
        epochs=3,
        batch_size=8,
        patience=2,
        threshold="auto",
        verbose=False,
        external_features=["promo"],
    )

    model.fit(df, target_col="target")
    preds = model.predict(df)

    assert preds.shape == (model.forecast_horizon,)
    assert (preds >= 0).all()


def test_save_and_load_model():
    """Test that models can be saved and loaded (GPU->CPU compatibility)"""
    df = pd.DataFrame(
        {
            "target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 3,
            "promo": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0] * 3,
        }
    )
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=3,
        epochs=2,
        batch_size=8,
        patience=2,
        threshold=0.5,
        verbose=False,
        external_features=["promo"],
    )

    model.fit(df, target_col="target")
    original_preds = model.predict(df)

    # Save and load the model
    model_path = _workspace_artifact_path("test_model.pkl")
    model.save_model(model_path)
    loaded_model = UR2CUTE.load_model(model_path)

    # Verify loaded model produces same predictions
    loaded_preds = loaded_model.predict(df)
    assert loaded_preds.shape == original_preds.shape
    assert (loaded_preds == original_preds).all()


def test_save_model_forces_cpu_state_dicts():
    """Ensure saved artifacts stay portable to CPU-only environments."""
    df = pd.DataFrame(
        {
            "target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 2,
            "promo": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0] * 2,
        }
    )
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=3,
        epochs=1,
        batch_size=8,
        patience=1,
        threshold=0.5,
        verbose=False,
        external_features=["promo"],
    )
    model.fit(df, target_col="target")

    model_path = _workspace_artifact_path("portable_model.pkl")
    model.save_model(model_path)

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    for state_dict in (saved["classifier_state_dict"], saved["regressor_state_dict"]):
        for tensor in state_dict.values():
            if torch.is_tensor(tensor):
                assert tensor.device.type == "cpu"


def test_auto_threshold_sets_value():
    """Auto threshold should derive a probability cutoff from training data."""
    df = pd.DataFrame({"target": [0, 0, 0, 5, 6, 0, 4, 0, 0, 3]})
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=2,
        epochs=1,
        batch_size=4,
        patience=1,
        threshold="auto",
        verbose=False,
    )
    model.fit(df, target_col="target")
    assert isinstance(model.threshold_, float)
    assert 0.0 <= model.threshold_ <= 1.0


def test_predict_validation_empty_dataframe():
    """Test that predict() validates empty DataFrames"""
    df = pd.DataFrame({"target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 3})
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=3,
        epochs=2,
        verbose=False,
    )
    model.fit(df, target_col="target")

    # Try to predict with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        model.predict(empty_df)


def test_predict_validation_insufficient_data():
    """Test that predict() validates data length"""
    df = pd.DataFrame({"target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 3})
    model = UR2CUTE(
        n_steps_lag=5,
        forecast_horizon=3,
        epochs=2,
        verbose=False,
    )
    model.fit(df, target_col="target")

    # Try to predict with insufficient data
    short_df = pd.DataFrame({"target": [1, 2, 3]})  # Only 3 rows, need 5
    with pytest.raises(ValueError, match="Insufficient data for prediction"):
        model.predict(short_df)


def test_predict_validation_missing_target_column():
    """Test that predict() validates target column exists"""
    df = pd.DataFrame({"target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 3})
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=3,
        epochs=2,
        verbose=False,
    )
    model.fit(df, target_col="target")

    # Try to predict with wrong column name
    wrong_df = pd.DataFrame({"wrong_name": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0]})
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        model.predict(wrong_df)


def test_predict_validation_missing_external_features():
    """Test that predict() validates external features exist"""
    df = pd.DataFrame(
        {
            "target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 3,
            "promo": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0] * 3,
        }
    )
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=3,
        epochs=2,
        verbose=False,
        external_features=["promo"],
    )
    model.fit(df, target_col="target")

    # Try to predict without external features
    df_no_promo = pd.DataFrame({"target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0]})
    with pytest.raises(ValueError, match="External features.*not found"):
        model.predict(df_no_promo)


def test_predict_validation_nan_values():
    """Test that predict() validates NaN values in required columns"""
    df = pd.DataFrame({"target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 3})
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=3,
        epochs=2,
        verbose=False,
    )
    model.fit(df, target_col="target")

    # Try to predict with NaN in last rows (need NaN in last n_steps_lag rows)
    df_with_nan = pd.DataFrame({"target": [0, 0, 5, 0, 7, 0, 0, 3, float('nan'), 10]})
    with pytest.raises(ValueError, match="contains NaN values"):
        model.predict(df_with_nan)


def test_fit_and_predict_with_single_feature():
    """A one-feature model should train without collapsing at the pooling layer."""
    df = pd.DataFrame({"target": [0, 1, 0, 2, 0, 3, 0, 4, 0, 5]})
    model = UR2CUTE(
        n_steps_lag=1,
        forecast_horizon=2,
        epochs=2,
        batch_size=2,
        patience=1,
        verbose=False,
    )

    model.fit(df, target_col="target")
    preds = model.predict(df)

    assert preds.shape == (model.forecast_horizon,)
    assert (preds >= 0).all()


def test_regressor_nonzero_only_false():
    """Regressor trained on all sequences when regressor_nonzero_only=False."""
    df = pd.DataFrame(
        {
            "target": [0, 0, 5, 0, 7, 0, 0, 3, 0, 0] * 3,
            "promo": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0] * 3,
        }
    )
    model = UR2CUTE(
        n_steps_lag=2,
        forecast_horizon=3,
        epochs=3,
        batch_size=8,
        patience=2,
        threshold="auto",
        regressor_nonzero_only=False,
        verbose=False,
        external_features=["promo"],
    )
    model.fit(df, target_col="target")
    preds = model.predict(df)

    assert preds.shape == (model.forecast_horizon,)
    assert (preds >= 0).all()


def test_regressor_nonzero_only_default_is_true():
    """Default value of regressor_nonzero_only should be True."""
    model = UR2CUTE()
    assert model.regressor_nonzero_only is True


def test_validation_split_keeps_targets_disjoint():
    """Validation windows should not reuse target periods seen in training."""
    df = pd.DataFrame({"target": list(range(1, 21))})
    lagged = _generate_lag_features(df, "target", n_lags=2).dropna().reset_index(drop=True)
    _, y_all = _create_multistep_data(lagged, "target", [], 2, 4)

    _, y_train, _, y_val = _split_train_validation_sequences(
        np.zeros((len(y_all), 1)),
        y_all,
        forecast_horizon=4,
    )

    train_targets = set(y_train.flatten().tolist())
    val_targets = set(y_val.flatten().tolist())

    assert train_targets.isdisjoint(val_targets)
