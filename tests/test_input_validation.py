import numpy as np
import pytest

from ekf3d.ekf_predictor import EKFPredictor3D
from ekf3d.ekf_updater import AzimuthElevationMeasurementModel, EKFUpdater3D


def test_predict_rejects_wrong_prior_mean_shape() -> None:
    predictor = EKFPredictor3D()
    with pytest.raises(AssertionError, match="prior_mean"):
        predictor.predict(
            prior_mean=np.zeros(5, dtype=np.float64),
            prior_covariance=np.eye(6, dtype=np.float64),
            dt=0.1,
        )


def test_predict_rejects_nonfinite_dt() -> None:
    predictor = EKFPredictor3D()
    with pytest.raises(AssertionError, match="dt"):
        predictor.predict(
            prior_mean=np.zeros(6, dtype=np.float64),
            prior_covariance=np.eye(6, dtype=np.float64),
            dt=np.nan,
        )


def test_predict_rejects_nonfinite_covariance() -> None:
    predictor = EKFPredictor3D()
    bad_covariance = np.eye(6, dtype=np.float64)
    bad_covariance[0, 0] = np.inf
    with pytest.raises(AssertionError, match="prior_covariance"):
        predictor.predict(
            prior_mean=np.zeros(6, dtype=np.float64),
            prior_covariance=bad_covariance,
            dt=0.1,
        )


def test_measurement_model_rejects_invalid_mapping() -> None:
    with pytest.raises(AssertionError, match="mapping index"):
        AzimuthElevationMeasurementModel(
            noise_covariance=np.eye(2, dtype=np.float64),
            mapping=(0, 2, 6),
            ndim_state=6,
        )


def test_measurement_function_rejects_wrong_state_shape() -> None:
    model = AzimuthElevationMeasurementModel(noise_covariance=np.eye(2, dtype=np.float64))
    with pytest.raises(AssertionError, match="state"):
        model.function(np.zeros(5, dtype=np.float64))


def test_updater_rejects_bad_noise_covariance_shape() -> None:
    with pytest.raises(AssertionError, match="noise_covariance"):
        EKFUpdater3D(noise_covariance=np.array([1.0, 1.0], dtype=np.float64))


def test_update_rejects_sensor_position_wrong_shape() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    predicted_mean = np.array([10.0, 0.1, 2.0, -0.1, 1.0, 0.0], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64)
    measurement = updater.measurement_model.function(predicted_mean)

    with pytest.raises(AssertionError, match="sensor_position"):
        updater.update(
            predicted_mean,
            predicted_covariance,
            measurement,
            sensor_position=np.array([1.0, 2.0], dtype=np.float64),
        )


def test_update_rejects_non_numeric_sensor_rotation() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    predicted_mean = np.array([10.0, 0.1, 2.0, -0.1, 1.0, 0.0], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64)
    measurement = updater.measurement_model.function(predicted_mean)

    with pytest.raises(AssertionError, match="sensor_rotation"):
        updater.update(
            predicted_mean,
            predicted_covariance,
            measurement,
            sensor_rotation=["pitch", "yaw"],
        )


def test_update_rejects_nonfinite_measurement() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    predicted_mean = np.array([10.0, 0.1, 2.0, -0.1, 1.0, 0.0], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64)
    measurement = np.array([0.1, np.nan], dtype=np.float64)

    with pytest.raises(AssertionError, match="measurement"):
        updater.update(predicted_mean, predicted_covariance, measurement)
