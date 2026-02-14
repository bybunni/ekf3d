from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from ekf3d.ekf_updater import (
    AzimuthElevationMeasurementModel,
    EKFUpdater3D,
)


def test_overhead_geometry_does_not_produce_non_finite_update() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.01, 0.01]).astype(np.float64))

    predicted_mean = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64)
    measurement = np.array([0.0, np.pi / 2.0], dtype=np.float64)

    H = updater.measurement_model.jacobian(predicted_mean)
    assert np.isfinite(H).all()

    posterior_mean, posterior_covariance = updater.update(
        predicted_mean=predicted_mean,
        predicted_covariance=predicted_covariance,
        measurement=measurement,
    )

    assert np.isfinite(posterior_mean).all()
    assert np.isfinite(posterior_covariance).all()


def test_update_accepts_numpy_sensor_position_input() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.05, 0.05]).astype(np.float64))
    predicted_mean = np.array([30.0, 1.0, -12.0, -0.5, 8.0, 0.2], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64)
    sensor_position = np.array([4.0, -3.0, 1.0], dtype=np.float64)

    model_for_measurement = AzimuthElevationMeasurementModel(
        noise_covariance=updater.noise_covariance,
        mapping=updater.mapping,
        ndim_state=updater.ndim_state,
        translation_offset=(4.0, -3.0, 1.0),
    )
    measurement = model_for_measurement.function(predicted_mean)

    posterior_mean_tuple, posterior_covariance_tuple = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        sensor_position=(4.0, -3.0, 1.0),
    )

    posterior_mean_array, posterior_covariance_array = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        sensor_position=sensor_position,
    )

    assert_allclose(posterior_mean_array, posterior_mean_tuple)
    assert_allclose(posterior_covariance_array, posterior_covariance_tuple)


def test_update_default_uses_matrix_inverse(monkeypatch) -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.02, 0.02]).astype(np.float64))
    predicted_mean = np.array([40.0, 0.3, 25.0, -0.2, 5.0, 0.1], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64) * 5.0
    measurement = updater.measurement_model.function(predicted_mean) + np.array(
        [0.02, -0.01],
        dtype=np.float64,
    )

    orig_inv = np.linalg.inv
    call_count = {"n": 0}

    def _counting_inv(*args: object, **kwargs: object) -> np.ndarray:
        call_count["n"] += 1
        return orig_inv(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "inv", _counting_inv)
    posterior_mean, posterior_covariance = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
    )

    assert call_count["n"] == 1
    assert np.isfinite(posterior_mean).all()
    assert np.isfinite(posterior_covariance).all()


def test_update_solve_method_avoids_matrix_inverse(monkeypatch) -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([0.02, 0.02]).astype(np.float64))
    predicted_mean = np.array([40.0, 0.3, 25.0, -0.2, 5.0, 0.1], dtype=np.float64)
    predicted_covariance = np.eye(6, dtype=np.float64) * 5.0
    measurement = updater.measurement_model.function(predicted_mean) + np.array(
        [0.02, -0.01],
        dtype=np.float64,
    )

    def _forbid_inv(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("np.linalg.inv should not be used when solve mode is selected")

    monkeypatch.setattr(np.linalg, "inv", _forbid_inv)
    posterior_mean, posterior_covariance = updater.update(
        predicted_mean,
        predicted_covariance,
        measurement,
        kalman_gain_method="solve",
    )

    assert np.isfinite(posterior_mean).all()
    assert np.isfinite(posterior_covariance).all()


def test_repeated_updates_keep_covariance_nearly_symmetric_and_psd() -> None:
    updater = EKFUpdater3D(noise_covariance=np.diag([1e-12, 1e-12]).astype(np.float64))
    rng = np.random.default_rng(0)

    state = np.array([100.0, 1.0, 20.0, -0.3, 8.0, 0.2], dtype=np.float64)
    covariance = np.eye(6, dtype=np.float64) * 1e8

    max_asymmetry = 0.0
    min_eigenvalue = np.inf
    for _ in range(500):
        measurement = updater.measurement_model.function(state) + rng.normal(
            scale=[1e-3, 1e-3],
            size=2,
        )
        state, covariance = updater.update(state, covariance, measurement)

        asymmetry = float(np.max(np.abs(covariance - covariance.T)))
        max_asymmetry = max(max_asymmetry, asymmetry)
        sym_covariance = 0.5 * (covariance + covariance.T)
        min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(sym_covariance).min()))

    assert max_asymmetry < 1e-4
    assert min_eigenvalue > -1e-4


def test_readme_uses_ekf3d_package_paths() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "from ekf3d.ekf_predictor import EKFPredictor3D" in readme
    assert "from ekf3d.ekf_updater import EKFUpdater3D" in readme
    assert "src/ekf3d/ekf_predictor.py" in readme
    assert "src/ekf3d/ekf_updater.py" in readme
    assert 'kalman_gain_method="solve"' in readme
    assert "bearing_only" not in readme
