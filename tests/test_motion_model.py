import numpy as np
from numpy.testing import assert_allclose

from ekf3d.ekf_predictor import CombinedCVModel3D, ConstantVelocityModel, EKFPredictor3D


def test_constant_velocity_transition_matrix_known_dt() -> None:
    model = ConstantVelocityModel(noise_diff_coeff=0.7)
    dt = 1.25

    F = model.transition_matrix(dt)

    expected = np.array([[1.0, 1.25], [0.0, 1.0]], dtype=np.float64)
    assert_allclose(F, expected)


def test_constant_velocity_process_noise_known_dt() -> None:
    noise_diff_coeff = 2.5
    model = ConstantVelocityModel(noise_diff_coeff=noise_diff_coeff)
    dt = 0.4

    Q = model.process_noise(dt)

    base = np.array(
        [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]],
        dtype=np.float64,
    )
    expected = noise_diff_coeff * base
    assert_allclose(Q, expected)
    assert_allclose(Q, Q.T)


def test_combined_model_is_block_diagonal() -> None:
    model = CombinedCVModel3D(
        noise_diff_coeff_x=1.0,
        noise_diff_coeff_y=2.0,
        noise_diff_coeff_z=3.0,
    )
    dt = 0.3

    F = model.transition_matrix(dt)
    Q = model.process_noise(dt)

    assert_allclose(F[0:2, 0:2], model.model_x.transition_matrix(dt))
    assert_allclose(F[2:4, 2:4], model.model_y.transition_matrix(dt))
    assert_allclose(F[4:6, 4:6], model.model_z.transition_matrix(dt))
    assert_allclose(Q[0:2, 0:2], model.model_x.process_noise(dt))
    assert_allclose(Q[2:4, 2:4], model.model_y.process_noise(dt))
    assert_allclose(Q[4:6, 4:6], model.model_z.process_noise(dt))

    assert_allclose(F[0:2, 2:6], np.zeros((2, 4), dtype=np.float64))
    assert_allclose(F[2:4, 0:2], np.zeros((2, 2), dtype=np.float64))
    assert_allclose(F[2:4, 4:6], np.zeros((2, 2), dtype=np.float64))
    assert_allclose(F[4:6, 0:4], np.zeros((2, 4), dtype=np.float64))

    assert_allclose(Q[0:2, 2:6], np.zeros((2, 4), dtype=np.float64))
    assert_allclose(Q[2:4, 0:2], np.zeros((2, 2), dtype=np.float64))
    assert_allclose(Q[2:4, 4:6], np.zeros((2, 2), dtype=np.float64))
    assert_allclose(Q[4:6, 0:4], np.zeros((2, 4), dtype=np.float64))


def test_predict_dt_zero_is_identity_propagation() -> None:
    predictor = EKFPredictor3D(
        noise_diff_coeff_x=0.3,
        noise_diff_coeff_y=0.4,
        noise_diff_coeff_z=0.5,
    )
    prior_mean = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    prior_cov = np.array(
        [
            [2.0, 0.1, 0.2, 0.0, 0.0, 0.1],
            [0.1, 1.5, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 2.5, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.3, 1.2, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1, 1.8, 0.2],
            [0.1, 0.0, 0.0, 0.0, 0.2, 1.1],
        ],
        dtype=np.float64,
    )

    predicted_mean, predicted_cov = predictor.predict(prior_mean, prior_cov, dt=0.0)

    assert_allclose(predicted_mean, prior_mean)
    assert_allclose(predicted_cov, prior_cov)
