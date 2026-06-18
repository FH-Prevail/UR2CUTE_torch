import os
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from UR2CUTE import UR2CUTE
from UR2CUTE.model import (
    _build_feature_frame,
    _window_data,
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
    feature_frame = _build_feature_frame(df, "target", [], 2)
    X_all, y_all = _window_data(
        feature_frame, df["target"].to_numpy(dtype=float), 2, 4
    )

    _, y_train, _, y_val = _split_train_validation_sequences(
        X_all,
        y_all,
        forecast_horizon=4,
    )

    train_targets = set(y_train.flatten().tolist())
    val_targets = set(y_val.flatten().tolist())

    assert train_targets.isdisjoint(val_targets)


def test_feature_frame_channel_layout():
    """Channels are [target, *external_features, time-since-last, rolling-mean]."""
    df = pd.DataFrame(
        {
            "target": [0, 0, 3, 0, 0, 5, 0],
            "promo": [0, 1, 0, 0, 1, 0, 0],
        }
    )
    frame = _build_feature_frame(df, "target", ["promo"], window=3)

    # 1 target + 1 external + 2 engineered channels
    assert frame.shape == (len(df), 4)
    # Channel 0 is the raw target.
    np.testing.assert_array_equal(frame[:, 0], df["target"].to_numpy(dtype=float))
    # Channel 1 is the external feature.
    np.testing.assert_array_equal(frame[:, 1], df["promo"].to_numpy(dtype=float))
    # Channel 2 is time-since-last-nonzero: resets to 0 on demand, counts up otherwise.
    np.testing.assert_array_equal(frame[:, 2], [1, 2, 0, 1, 2, 0, 1])


def test_window_shape_and_causality():
    """Windows are (samples, channels, window) with no future leakage into X."""
    df = pd.DataFrame({"target": list(range(20))})
    frame = _build_feature_frame(df, "target", [], window=4)
    X, y = _window_data(frame, df["target"].to_numpy(dtype=float), window=4, forecast_horizon=3)

    assert X.ndim == 3
    assert X.shape[1] == frame.shape[1]  # channels
    assert X.shape[2] == 4  # window length
    assert y.shape == (X.shape[0], 3)
    # For the first sample, history target channel is rows 0..3 and the
    # forecast targets are the next three rows (4, 5, 6) — strictly future.
    np.testing.assert_array_equal(X[0, 0, :], [0, 1, 2, 3])
    np.testing.assert_array_equal(y[0], [4, 5, 6])


def test_log_target_skipped_for_negative_values():
    """log1p must be skipped when the target can be negative (no NaNs)."""
    df = pd.DataFrame(
        {"target": [0, 0, -2, 0, 3, 0, 0, -1, 0, 4, 0, 0, 5, 0, 0, 2, 0, 0, 3, 0]}
    )
    model = UR2CUTE(n_steps_lag=4, forecast_horizon=3, epochs=2, verbose=False)
    model.fit(df, target_col="target")

    assert model.use_log_target_ is False
    assert not np.isnan(model.predict(df)).any()


def test_large_magnitude_predictions_stay_bounded():
    """Large-magnitude intermittent demand must not explode through expm1."""
    rng = np.random.RandomState(0)
    n = 120
    demand = (rng.rand(n) < 0.3) * rng.randint(100, 5000, size=n)
    df = pd.DataFrame({"target": demand.astype(float)})
    model = UR2CUTE(n_steps_lag=12, forecast_horizon=6, epochs=20, verbose=False)
    model.fit(df, target_col="target")
    preds = model.predict(df)

    assert np.isfinite(preds).all()
    # No forecast should exceed a small margin above the largest observed demand.
    assert preds.max() <= demand.max() * 3


def test_learns_periodic_spike_pattern():
    """On a clean period-4 pattern the model should recover the spike timing."""
    n = 240
    df = pd.DataFrame({"target": [5 if i % 4 == 0 else 0 for i in range(n)]})
    model = UR2CUTE(
        n_steps_lag=12,
        forecast_horizon=8,
        epochs=40,
        batch_size=16,
        patience=8,
        threshold=0.5,
        verbose=False,
    )
    model.fit(df, target_col="target")
    preds = model.predict(df)

    # Some steps fire and some stay zero — it has not collapsed to all-zero or all-spike.
    assert (preds > 0).any()
    assert (preds == 0).any()
