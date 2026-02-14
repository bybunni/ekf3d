import numpy as np
from numpy.testing import assert_allclose

from ekf3d.ekf_updater import EKFUpdater3D, normalize_angles


def _residual_norm(
    updater: EKFUpdater3D, state: np.ndarray, measurement: np.ndarray
) -> float:
    pred_meas = updater.measurement_model.function(state)
    residual = normalize_angles(measurement - pred_meas)
    return float(np.linalg.norm(residual))


def test_normalize_angles_range_and_values() -> None:
    angles = np.array(
        [-4.0 * np.pi, np.pi + 0.2, -np.pi - 0.3, 0.0],
        dtype=np.float64,
    )

    normalized = normalize_angles(angles)

    assert np.all(normalized <= np.pi)
    assert np.all(normalized >= -np.pi)
    assert_allclose(normalized[0], 0.0)
    assert_allclose(normalized[1], -np.pi + 0.2)
    assert_allclose(normalized[2], np.pi - 0.3)


def test_update_shapes_and_residual_reduction() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))

    predicted_mean = np.array([100.0, 0.5, 20.0, -0.2, 5.0, 0.1], dtype=np.float64)
    predicted_cov = np.eye(6, dtype=np.float64) * 10.0
    measurement = updater.measurement_model.function(predicted_mean) + np.array(
        [0.05, -0.03],
        dtype=np.float64,
    )

    residual_before = _residual_norm(updater, predicted_mean, measurement)
    posterior_mean, posterior_cov = updater.update(
        predicted_mean,
        predicted_cov,
        measurement,
    )
    residual_after = _residual_norm(updater, posterior_mean, measurement)

    assert posterior_mean.shape == (6,)
    assert posterior_cov.shape == (6, 6)
    assert np.isfinite(posterior_mean).all()
    assert np.isfinite(posterior_cov).all()
    assert residual_after < residual_before
    assert_allclose(posterior_cov, posterior_cov.T, atol=1e-10)


def test_update_wraps_azimuth_residual() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([1e-4, 1e-4]).astype(np.float64))

    predicted_mean = np.array([-1.0, 0.0, -1e-3, 0.0, 0.2, 0.0], dtype=np.float64)
    predicted_cov = np.eye(6, dtype=np.float64)

    predicted_measurement = updater.measurement_model.function(predicted_mean)
    measurement = np.array([np.pi - 0.002, predicted_measurement[1]], dtype=np.float64)

    residual_before = _residual_norm(updater, predicted_mean, measurement)
    posterior_mean, _ = updater.update(predicted_mean, predicted_cov, measurement)
    residual_after = _residual_norm(updater, posterior_mean, measurement)

    assert residual_before < 0.01
    assert residual_after < residual_before
