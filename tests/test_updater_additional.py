import numpy as np
import pytest
from numpy.testing import assert_allclose

from ekf3d.ekf_updater import (
    AzimuthElevationMeasurementModel,
    EKFUpdater3D,
    normalize_angles,
)


def test_normalize_angles_wrap_boundaries() -> None:
    angles = np.array(
        [
            np.pi,
            -np.pi,
            np.pi + 1e-12,
            -np.pi - 1e-12,
            3.0 * np.pi,
            -3.0 * np.pi,
        ],
        dtype=np.float64,
    )

    normalized = normalize_angles(angles)

    assert np.all(normalized <= np.pi)
    assert np.all(normalized >= -np.pi)
    assert_allclose(
        normalized,
        np.array([-np.pi, -np.pi, -np.pi, np.pi, -np.pi, -np.pi], dtype=np.float64),
        atol=1e-10,
    )


def test_measurement_model_custom_mapping_nonzero_columns() -> None:
    model = AzimuthElevationMeasurementModel(
        noise_covariance=np.diag([0.01, 0.01]).astype(np.float64),
        mapping=(1, 4, 7),
        ndim_state=9,
    )
    state = np.zeros(9, dtype=np.float64)
    state[1] = 3.0
    state[4] = -4.0
    state[7] = 2.0

    H = model.jacobian(state)
    nonzero_cols = np.where(np.abs(H).sum(axis=0) > 1e-12)[0]

    assert tuple(nonzero_cols.tolist()) == (1, 4, 7)


def test_update_accepts_array_like_sensor_rotation() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    predicted_mean = np.array([30.0, 0.2, -10.0, 0.1, 8.0, -0.2], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64)
    rotation_tuple = (0.2, -0.3)

    model = AzimuthElevationMeasurementModel(
        noise_covariance=updater.noise_covariance,
        mapping=updater.mapping,
        ndim_state=updater.ndim_state,
        rotation_offset=rotation_tuple,
    )
    measurement = model.function(predicted_mean)

    mean_tuple, cov_tuple = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        sensor_rotation=rotation_tuple,
    )
    mean_list, cov_list = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        sensor_rotation=[0.2, -0.3],
    )
    mean_array, cov_array = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        sensor_rotation=np.array([0.2, -0.3], dtype=np.float64),
    )

    assert_allclose(mean_list, mean_tuple)
    assert_allclose(cov_list, cov_tuple)
    assert_allclose(mean_array, mean_tuple)
    assert_allclose(cov_array, cov_tuple)


def test_update_invalid_kalman_gain_method_raises() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    predicted_mean = np.array([20.0, 0.2, 5.0, -0.1, 2.0, 0.0], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64)
    measurement = updater.measurement_model.function(predicted_mean)

    with pytest.raises(ValueError, match="kalman_gain_method"):
        updater.update(
            predicted_mean,
            predicted_covariance,
            measurement,
            kalman_gain_method="bad-method",
        )


def test_update_inv_and_solve_agree_when_well_conditioned() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.02, 0.02]).astype(np.float64))
    predicted_mean = np.array([50.0, 0.4, 15.0, -0.2, 6.0, 0.1], dtype=np.float64)
    predicted_covariance = np.diag([4.0, 3.0, 5.0, 3.5, 4.5, 2.5]).astype(np.float64)
    measurement = updater.measurement_model.function(predicted_mean) + np.array(
        [0.03, -0.015],
        dtype=np.float64,
    )

    mean_inv, cov_inv = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        kalman_gain_method="inv",
    )
    mean_solve, cov_solve = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        kalman_gain_method="solve",
    )

    assert_allclose(mean_inv, mean_solve, atol=1e-12, rtol=1e-10)
    assert_allclose(cov_inv, cov_solve, atol=1e-12, rtol=1e-10)


def test_update_solve_method_calls_linear_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.02, 0.02]).astype(np.float64))
    predicted_mean = np.array([45.0, 0.3, 12.0, -0.1, 4.0, 0.05], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64) * 3.0
    measurement = updater.measurement_model.function(predicted_mean) + np.array(
        [0.01, -0.005],
        dtype=np.float64,
    )

    original_solve = np.linalg.solve
    call_count = {"n": 0}

    def _counting_solve(*args: object, **kwargs: object) -> np.ndarray:
        call_count["n"] += 1
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", _counting_solve)
    posterior_mean, posterior_covariance = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        kalman_gain_method="solve",
    )

    assert call_count["n"] == 1
    assert np.isfinite(posterior_mean).all()
    assert np.isfinite(posterior_covariance).all()


def test_update_raises_on_incorrect_measurement_shape() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    predicted_mean = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64)

    with pytest.raises(ValueError):
        updater.update(predicted_mean, predicted_covariance, measurement=np.zeros(3))


def test_update_raises_on_incorrect_covariance_shape() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    predicted_mean = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    bad_covariance = np.eye(5, dtype=np.float64)
    measurement = updater.measurement_model.function(predicted_mean)

    with pytest.raises(ValueError):
        updater.update(predicted_mean, bad_covariance, measurement)


def test_near_singularity_sweep_is_finite_for_nonzero_rho() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    covariance = np.eye(6, dtype=np.float64)

    for eps in (1e-1, 1e-3, 1e-6, 1e-9):
        predicted_mean = np.array([eps, 0.0, -eps, 0.0, 10.0, 0.0], dtype=np.float64)
        measurement = updater.measurement_model.function(predicted_mean)
        H = updater.measurement_model.jacobian(predicted_mean)
        posterior_mean, posterior_covariance = updater.update(
            predicted_mean,
            covariance,
            measurement,
        )

        assert np.isfinite(H).all()
        assert np.isfinite(posterior_mean).all()
        assert np.isfinite(posterior_covariance).all()


def test_nominal_repeated_updates_keep_covariance_psd() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))
    rng = np.random.default_rng(7)
    state = np.array([100.0, 1.0, 20.0, -0.3, 8.0, 0.2], dtype=np.float64)
    covariance = np.eye(6, dtype=np.float64) * 10.0

    min_eig = np.inf
    max_asym = 0.0
    for _ in range(100):
        measurement = updater.measurement_model.function(state) + rng.normal(
            scale=[0.02, 0.02],
            size=2,
        )
        state, covariance = updater.update(state, covariance, measurement)
        sym_covariance = 0.5 * (covariance + covariance.T)
        min_eig = min(min_eig, float(np.linalg.eigvalsh(sym_covariance).min()))
        max_asym = max(max_asym, float(np.max(np.abs(covariance - covariance.T))))

    assert min_eig > -1e-10
    assert max_asym < 1e-10
