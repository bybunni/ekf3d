import numpy as np
import pytest

from ekf3d.ekf_predictor import EKFPredictor3D
from ekf3d.ekf_updater import AzimuthElevationMeasurementModel, EKFUpdater3D


@pytest.mark.perf
def test_benchmark_predict_nominal(benchmark) -> None:
    predictor = EKFPredictor3D(
        noise_diff_coeff_x=0.01,
        noise_diff_coeff_y=0.01,
        noise_diff_coeff_z=0.01,
    )
    prior_mean = np.array([100.0, 1.2, -40.0, 0.3, 20.0, -0.4], dtype=np.float64)
    prior_covariance = np.diag([10.0, 2.0, 8.0, 2.0, 5.0, 1.0]).astype(np.float64)
    dt = 0.1

    def _run() -> tuple[np.ndarray, np.ndarray]:
        return predictor.predict(prior_mean, prior_covariance, dt)

    benchmark(_run)


def _make_update_case(
    scenario: str, with_sensor_pose: bool
) -> tuple[
    EKFUpdater3D,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float, float] | None,
    tuple[float, float] | None,
]:
    sensor_position: tuple[float, float, float] | None = None
    sensor_rotation: tuple[float, float] | None = None

    if scenario == "stress":
        updater = EKFUpdater3D(noise_covariance=np.diag([1e-12, 1e-12]).astype(np.float64))
        predicted_mean = np.array([100.0, 1.0, 20.0, -0.3, 8.0, 0.2], dtype=np.float64)
        predicted_covariance = np.eye(6, dtype=np.float64) * 1e8
    elif scenario == "near_singularity":
        updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
        predicted_mean = np.array([1e-6, 0.0, -1e-6, 0.0, 10.0, 0.0], dtype=np.float64)
        predicted_covariance = np.eye(6, dtype=np.float64)
    else:
        updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
        predicted_mean = np.array([120.0, 0.7, 25.0, -0.2, 6.0, 0.1], dtype=np.float64)
        predicted_covariance = np.diag([12.0, 2.0, 10.0, 2.0, 8.0, 1.0]).astype(np.float64)

    if with_sensor_pose:
        sensor_position = (4.0, -2.0, 1.0)
        sensor_rotation = (0.1, -0.2)
        measurement_model = AzimuthElevationMeasurementModel(
            noise_covariance=updater.noise_covariance,
            mapping=updater.mapping,
            ndim_state=updater.ndim_state,
            translation_offset=sensor_position,
            rotation_offset=sensor_rotation,
        )
    else:
        measurement_model = updater.measurement_model

    measurement = measurement_model.function(predicted_mean)
    return (
        updater,
        predicted_mean,
        predicted_covariance,
        measurement,
        sensor_position,
        sensor_rotation,
    )


@pytest.mark.perf
@pytest.mark.parametrize("scenario", ["nominal", "near_singularity", "stress"])
@pytest.mark.parametrize("kalman_gain_method", ["inv", "solve"])
@pytest.mark.parametrize("with_sensor_pose", [False, True])
def test_benchmark_update(
    benchmark,
    scenario: str,
    kalman_gain_method: str,
    with_sensor_pose: bool,
) -> None:
    (
        updater,
        predicted_mean,
        predicted_covariance,
        measurement,
        sensor_position,
        sensor_rotation,
    ) = _make_update_case(scenario, with_sensor_pose)

    def _run() -> tuple[np.ndarray, np.ndarray]:
        return updater.update(
            predicted_mean,
            predicted_covariance,
            measurement,
            sensor_position=sensor_position,
            sensor_rotation=sensor_rotation,
            kalman_gain_method=kalman_gain_method,
        )

    benchmark(_run)
