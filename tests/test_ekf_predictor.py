import numpy as np
from numpy.testing import assert_allclose

from ekf3d.ekf_predictor import CombinedCVModel3D, EKFPredictor3D


def test_transition_matrix_known_dt() -> None:
    model = CombinedCVModel3D()
    dt = 0.5

    F = model.transition_matrix(dt)

    expected = np.array(
        [
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert_allclose(F, expected)


def test_process_noise_known_dt_and_axis_coefficients() -> None:
    model = CombinedCVModel3D(
        noise_diff_coeff_x=1.0,
        noise_diff_coeff_y=2.0,
        noise_diff_coeff_z=3.0,
    )
    dt = 2.0

    Q = model.process_noise(dt)

    base = np.array([[8.0 / 3.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    expected = np.zeros((6, 6), dtype=np.float64)
    expected[0:2, 0:2] = 1.0 * base
    expected[2:4, 2:4] = 2.0 * base
    expected[4:6, 4:6] = 3.0 * base
    assert_allclose(Q, expected)


def test_predict_matches_closed_form_equations() -> None:
    predictor = EKFPredictor3D(
        noise_diff_coeff_x=0.2,
        noise_diff_coeff_y=0.3,
        noise_diff_coeff_z=0.4,
    )
    prior_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    prior_cov = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float64)
    dt = 0.25

    predicted_mean, predicted_cov = predictor.predict(prior_mean, prior_cov, dt)

    F = predictor.motion_model.transition_matrix(dt)
    Q = predictor.motion_model.process_noise(dt)
    assert_allclose(predicted_mean, F @ prior_mean)
    assert_allclose(predicted_cov, F @ prior_cov @ F.T + Q)
