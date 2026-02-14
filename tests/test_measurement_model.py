import numpy as np
from numpy.testing import assert_allclose

from ekf3d.ekf_updater import AzimuthElevationMeasurementModel


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def test_measurement_function_known_angles() -> None:
    model = AzimuthElevationMeasurementModel(noise_covariance=np.eye(2, dtype=np.float64))
    state = np.array([1.0, 0.0, 1.0, 0.0, np.sqrt(2.0), 0.0], dtype=np.float64)

    measurement = model.function(state)

    expected = np.array([np.pi / 4.0, np.pi / 4.0], dtype=np.float64)
    assert_allclose(measurement, expected)


def test_measurement_jacobian_matches_finite_difference() -> None:
    model = AzimuthElevationMeasurementModel(
        noise_covariance=np.eye(2, dtype=np.float64) * 0.01,
        translation_offset=(2.0, -3.0, 1.0),
        rotation_offset=(0.2, -0.4),
    )
    state = np.array([15.0, 0.2, 4.0, -0.1, 8.0, 0.3], dtype=np.float64)

    H = model.jacobian(state)

    eps = 1e-7
    H_fd = np.zeros_like(H)
    for idx in (0, 2, 4):
        state_plus = state.copy()
        state_minus = state.copy()
        state_plus[idx] += eps
        state_minus[idx] -= eps

        meas_plus = model.function(state_plus)
        meas_minus = model.function(state_minus)
        delta = meas_plus - meas_minus
        delta[0] = _wrap_angle(delta[0])
        H_fd[:, idx] = delta / (2.0 * eps)

    assert_allclose(H, H_fd, rtol=1e-5, atol=1e-6)
    assert_allclose(H[:, [1, 3, 5]], np.zeros((2, 3), dtype=np.float64))
