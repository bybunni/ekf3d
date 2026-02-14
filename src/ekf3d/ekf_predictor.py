"""Standalone Extended Kalman Filter Predictor.

This module provides a self-contained EKF predictor implementation
that can be easily ported to other languages. All code is in a single
file with no external dependencies beyond NumPy.

Mathematical Reference:
    x_pred = F @ x_prior
    P_pred = F @ P_prior @ F.T + Q

Where:
    F = state transition matrix (4x4 block diagonal)
    Q = process noise covariance (4x4 block diagonal)
    x = state vector [x, vx, y, vy]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _assert_real_scalar(name: str, value: object, *, min_value: float | None = None) -> float:
    assert np.isscalar(value), f"{name} must be a real scalar; got type {type(value).__name__}"
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        assert False, f"{name} must be convertible to float; got {value!r}"
    assert np.isfinite(scalar), f"{name} must be finite; got {scalar!r}"
    if min_value is not None:
        assert scalar >= min_value, f"{name} must be >= {min_value}; got {scalar}"
    return scalar


def _coerce_vector(
    name: str, value: NDArray[np.float64] | list[float] | tuple[float, ...], length: int
) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        assert False, f"{name} must be array-like with numeric values; got {value!r}"
    assert array.shape == (length,), f"{name} must have shape ({length},); got {array.shape}"
    assert np.isfinite(array).all(), f"{name} must contain only finite values"
    return array


def _coerce_matrix(
    name: str,
    value: NDArray[np.float64] | list[list[float]] | tuple[tuple[float, ...], ...],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        assert False, f"{name} must be array-like with numeric values; got {value!r}"
    assert array.shape == shape, f"{name} must have shape {shape}; got {array.shape}"
    assert np.isfinite(array).all(), f"{name} must contain only finite values"
    return array


class ConstantVelocityModel:
    """1D constant velocity transition model.

    Models motion where velocity is approximately constant and
    acceleration is modeled as white noise.

    State vector: [position, velocity]

    Transition matrix (2x2):
        F = [[1, dt],
             [0, 1 ]]

    Process noise covariance (2x2):
        Q = noise_diff_coeff * [[dt³/3, dt²/2],
                                [dt²/2, dt    ]]
    """

    def __init__(self, noise_diff_coeff: float = 1.0) -> None:
        """Initialize the constant velocity model.

        Args:
            noise_diff_coeff: Process noise diffusion coefficient (variance).
                Controls how much the velocity can change between timesteps.
        """
        self.noise_diff_coeff = _assert_real_scalar(
            "noise_diff_coeff", noise_diff_coeff, min_value=0.0
        )

    def transition_matrix(self, dt: float) -> NDArray[np.float64]:
        """Compute the 2x2 state transition matrix.

        Args:
            dt: Time step in seconds.

        Returns:
            2x2 transition matrix F = [[1, dt], [0, 1]].
        """
        dt = _assert_real_scalar("dt", dt, min_value=0.0)
        return np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)

    def process_noise(self, dt: float) -> NDArray[np.float64]:
        """Compute the 2x2 process noise covariance matrix.

        The process noise models random acceleration as white noise.

        Args:
            dt: Time step in seconds.

        Returns:
            2x2 process noise covariance matrix Q.
        """
        dt = _assert_real_scalar("dt", dt, min_value=0.0)
        dt2 = dt * dt
        dt3 = dt2 * dt

        Q = np.array([[dt3 / 3.0, dt2 / 2.0], [dt2 / 2.0, dt]], dtype=np.float64)

        return Q * self.noise_diff_coeff


class CombinedCVModel3D:
    """Combined 3D constant velocity model.

    Combines three 1D constant velocity models (for x, y, and z dimensions)
    into a single 6D model. The resulting matrices are block diagonal.

    State vector: [x, vx, y, vy, z, vz]

    The models are assumed to be independent, so the combined
    transition matrix and process noise are block diagonal:

        F_combined = [[F_x,  0,   0 ],
                      [ 0,  F_y,  0 ],
                      [ 0,   0,  F_z]]

        Q_combined = [[Q_x,  0,   0 ],
                      [ 0,  Q_y,  0 ],
                      [ 0,   0,  Q_z]]
    """

    def __init__(
        self,
        noise_diff_coeff_x: float = 1.0,
        noise_diff_coeff_y: float = 1.0,
        noise_diff_coeff_z: float = 1.0,
    ) -> None:
        """Initialize the combined 3D model.

        Args:
            noise_diff_coeff_x: Process noise coefficient for x dimension.
            noise_diff_coeff_y: Process noise coefficient for y dimension.
            noise_diff_coeff_z: Process noise coefficient for z dimension.
        """
        self.model_x = ConstantVelocityModel(
            _assert_real_scalar("noise_diff_coeff_x", noise_diff_coeff_x, min_value=0.0)
        )
        self.model_y = ConstantVelocityModel(
            _assert_real_scalar("noise_diff_coeff_y", noise_diff_coeff_y, min_value=0.0)
        )
        self.model_z = ConstantVelocityModel(
            _assert_real_scalar("noise_diff_coeff_z", noise_diff_coeff_z, min_value=0.0)
        )

    def transition_matrix(self, dt: float) -> NDArray[np.float64]:
        """Compute the 6x6 block-diagonal state transition matrix.

        Args:
            dt: Time step in seconds.

        Returns:
            6x6 block-diagonal transition matrix for [x, vx, y, vy, z, vz] state.
            Structure: [[F_x,  0,   0 ],
                        [ 0,  F_y,  0 ],
                        [ 0,   0,  F_z]]
        """
        F_x = self.model_x.transition_matrix(dt)
        F_y = self.model_y.transition_matrix(dt)
        F_z = self.model_z.transition_matrix(dt)

        # Build 6x6 block diagonal matrix
        F = np.zeros((6, 6), dtype=np.float64)
        F[0:2, 0:2] = F_x
        F[2:4, 2:4] = F_y
        F[4:6, 4:6] = F_z

        return F

    def process_noise(self, dt: float) -> NDArray[np.float64]:
        """Compute the 6x6 block-diagonal process noise covariance.

        Args:
            dt: Time step in seconds.

        Returns:
            6x6 block-diagonal process noise covariance for [x, vx, y, vy, z, vz] state.
            Structure: [[Q_x,  0,   0 ],
                        [ 0,  Q_y,  0 ],
                        [ 0,   0,  Q_z]]
        """
        Q_x = self.model_x.process_noise(dt)
        Q_y = self.model_y.process_noise(dt)
        Q_z = self.model_z.process_noise(dt)

        # Build 6x6 block diagonal matrix
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:2, 0:2] = Q_x
        Q[2:4, 2:4] = Q_y
        Q[4:6, 4:6] = Q_z

        return Q


class EKFPredictor3D:
    """Extended Kalman Filter predictor for 3D state.

    Performs the prediction step of the EKF using a combined
    3D constant velocity motion model.

    State vector: [x, vx, y, vy, z, vz]

    Prediction equations:
        x_pred = F @ x_prior
        P_pred = F @ P_prior @ F.T + Q
    """

    def __init__(
        self,
        noise_diff_coeff_x: float = 1.0,
        noise_diff_coeff_y: float = 1.0,
        noise_diff_coeff_z: float = 1.0,
    ) -> None:
        """Initialize the 3D EKF predictor.

        Args:
            noise_diff_coeff_x: Process noise coefficient for x dimension.
            noise_diff_coeff_y: Process noise coefficient for y dimension.
            noise_diff_coeff_z: Process noise coefficient for z dimension.
        """
        self.motion_model = CombinedCVModel3D(
            noise_diff_coeff_x, noise_diff_coeff_y, noise_diff_coeff_z
        )

    def predict(
        self,
        prior_mean: NDArray[np.float64],
        prior_covariance: NDArray[np.float64],
        dt: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Perform the EKF prediction step.

        Args:
            prior_mean: Prior state mean vector (6,) or (6, 1).
            prior_covariance: Prior state covariance matrix (6, 6).
            dt: Time step in seconds.

        Returns:
            Tuple of (predicted_mean, predicted_covariance):
            - predicted_mean: Predicted state mean (6,) array.
            - predicted_covariance: Predicted covariance (6, 6) matrix.
        """
        # Ensure prior_mean is a column vector for matrix operations
        x = _coerce_vector("prior_mean", prior_mean, length=6)
        P = _coerce_matrix("prior_covariance", prior_covariance, shape=(6, 6))
        dt = _assert_real_scalar("dt", dt, min_value=0.0)

        # Get transition matrix and process noise
        F = self.motion_model.transition_matrix(dt)
        Q = self.motion_model.process_noise(dt)

        # Prediction equations
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        return x_pred, P_pred
