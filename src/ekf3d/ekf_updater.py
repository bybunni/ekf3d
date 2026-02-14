"""Standalone Extended Kalman Filter Updater.

This module provides a self-contained EKF updater implementation
that can be easily ported to other languages. All code is in a single
file with no external dependencies beyond NumPy.

Mathematical Reference:
    h(x) = atan2(y, x)                    # Measurement function
    H = [-y/r², 0, x/r², 0]               # Jacobian where r² = x² + y²
    R = [[noise_variance]]                # Measurement noise (1x1)

    S = H @ P_pred @ H.T + R              # Innovation covariance
    K = P_pred @ H.T @ inv(S)             # Kalman gain
    innovation = z - h(x_pred)            # Measurement residual

    x_post = x_pred + K @ innovation      # Posterior mean
    P_post = (I - K @ H) @ P_pred         # Posterior covariance
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_angles(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a vector of angles to the range [-pi, pi].

    Args:
        angles: Array of angles in radians.

    Returns:
        Array of normalized angles in radians.
    """
    result = np.asarray(angles, dtype=np.float64).copy()
    result = np.mod(result + np.pi, 2.0 * np.pi) - np.pi
    return result


class AzimuthElevationMeasurementModel:
    """3D measurement model for azimuth and elevation angles.

    Converts 3D Cartesian position to azimuth and elevation angles,
    accounting for sensor position and orientation.

    State vector: [x, vx, y, vy, z, vz] (6D)

    Measurement function (after applying translation and rotation):
        h(x) = [φ]  = [atan2(y', x')        ]   # Azimuth
               [θ]    [arcsin(z'/r)         ]   # Elevation

        where r = √(x'² + y'² + z'²)
        and (x', y', z') is the position after translation and rotation.

    Jacobian (2x6):
        H = [[∂φ/∂x, 0, ∂φ/∂y, 0, ∂φ/∂z, 0],
             [∂θ/∂x, 0, ∂θ/∂y, 0, ∂θ/∂z, 0]]

        Where (before rotation):
          ∂φ/∂x = -y/(x² + y²)
          ∂φ/∂y = x/(x² + y²)
          ∂φ/∂z = 0

          ∂θ/∂x = -xz / (r² · √(x² + y²))
          ∂θ/∂y = -yz / (r² · √(x² + y²))
          ∂θ/∂z = √(x² + y²) / r²

    Note: Stone Soup returns [elevation, bearing] but we return [azimuth, elevation]
    for consistency with our 2D convention where bearing is the primary measurement.
    """

    def __init__(
        self,
        noise_covariance: NDArray[np.float64],
        mapping: tuple[int, int, int] = (0, 2, 4),
        ndim_state: int = 6,
        translation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_offset: tuple[float, float] | None = None,
    ) -> None:
        """Initialize the azimuth-elevation measurement model.

        Args:
            noise_covariance: 2x2 measurement noise covariance matrix.
                [[sigma_azimuth^2, 0], [0, sigma_elevation^2]]
            mapping: Tuple of (x_index, y_index, z_index) into the state vector.
                Default (0, 2, 4) for state [x, vx, y, vy, z, vz].
            ndim_state: Dimension of the state vector. Default 6.
            translation_offset: Sensor position (x, y, z). Default (0, 0, 0).
            rotation_offset: Sensor orientation as (pitch, yaw) in radians.
                For a roll-stabilized sensor (roll=0).
                If None, no rotation is applied.
        """
        self.noise_covariance = np.asarray(noise_covariance, dtype=np.float64)
        self.mapping = mapping
        self.ndim_state = ndim_state
        self.translation_offset = translation_offset
        self.rotation_offset = rotation_offset

        # Precompute rotation matrix if rotation is specified
        if rotation_offset is not None:
            self._rotation_matrix = self._build_rotation_matrix(rotation_offset)
        else:
            self._rotation_matrix = np.eye(3, dtype=np.float64)

    @staticmethod
    def _build_rotation_matrix(
        rotation_offset: tuple[float, float],
    ) -> NDArray[np.float64]:
        """Build 3x3 rotation matrix for roll-stabilized sensor.

        Uses convention matching Stone Soup: R = rotx(-roll) @ roty(pitch) @ rotz(-yaw)
        For roll-stabilized sensor (roll=0): R = roty(pitch) @ rotz(-yaw)

        Args:
            rotation_offset: (pitch, yaw) angles in radians.

        Returns:
            3x3 rotation matrix.
        """
        pitch, yaw = rotation_offset

        # roty(pitch)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Ry = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64
        )

        # rotz(-yaw)
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        Rz = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        return Ry @ Rz

    @property
    def noise_covar(self) -> NDArray[np.float64]:
        """Return the (2, 2) measurement noise covariance matrix R."""
        return self.noise_covariance

    def function(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the measurement function.

        Args:
            state: State vector of shape (ndim_state,) or (ndim_state, 1).

        Returns:
            Measurement vector [azimuth, elevation] as (2,) array.
        """
        state = np.asarray(state, dtype=np.float64).flatten()

        # Extract position and apply translation
        dx = state[self.mapping[0]] - self.translation_offset[0]
        dy = state[self.mapping[1]] - self.translation_offset[1]
        dz = state[self.mapping[2]] - self.translation_offset[2]

        # Apply rotation
        xyz = np.array([dx, dy, dz])
        xyz_rot = self._rotation_matrix @ xyz
        x_rot, y_rot, z_rot = xyz_rot

        # Compute range
        r = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)

        # Compute azimuth and elevation
        azimuth = np.arctan2(y_rot, x_rot)
        elevation = np.arcsin(np.clip(z_rot / r, -1.0, 1.0))

        return np.array([azimuth, elevation], dtype=np.float64)

    def jacobian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the measurement Jacobian matrix.

        Args:
            state: State vector of shape (ndim_state,) or (ndim_state, 1).

        Returns:
            Jacobian matrix H of shape (2, ndim_state).
        """
        state = np.asarray(state, dtype=np.float64).flatten()

        # Extract position and apply translation
        dx = state[self.mapping[0]] - self.translation_offset[0]
        dy = state[self.mapping[1]] - self.translation_offset[1]
        dz = state[self.mapping[2]] - self.translation_offset[2]

        # Apply rotation
        R = self._rotation_matrix
        xyz = np.array([dx, dy, dz])
        xyz_rot = R @ xyz
        x_rot, y_rot, z_rot = xyz_rot

        # Compute intermediate values
        rho_sq = x_rot**2 + y_rot**2  # xy-plane distance squared
        r_sq = rho_sq + z_rot**2  # 3D range squared
        rho = np.sqrt(rho_sq)  # xy-plane distance

        # Jacobian of [azimuth, elevation] w.r.t. rotated coordinates [x', y', z']
        # dφ/dx' = -y'/(x'² + y'²)
        # dφ/dy' = x'/(x'² + y'²)
        # dφ/dz' = 0
        dphi_dxrot = -y_rot / rho_sq
        dphi_dyrot = x_rot / rho_sq
        dphi_dzrot = 0.0

        # dθ/dx' = -x'z' / (r² · √(x'² + y'²))
        # dθ/dy' = -y'z' / (r² · √(x'² + y'²))
        # dθ/dz' = √(x'² + y'²) / r²
        dtheta_dxrot = -x_rot * z_rot / (r_sq * rho)
        dtheta_dyrot = -y_rot * z_rot / (r_sq * rho)
        dtheta_dzrot = rho / r_sq

        # Jacobian w.r.t. rotated coordinates (2x3)
        J_rot = np.array(
            [
                [dphi_dxrot, dphi_dyrot, dphi_dzrot],
                [dtheta_dxrot, dtheta_dyrot, dtheta_dzrot],
            ],
            dtype=np.float64,
        )

        # Chain rule: J_state = J_rot @ R
        # Because xyz_rot = R @ xyz, d(xyz_rot)/d(xyz) = R
        J_xyz = J_rot @ R

        # Build full Jacobian (2 x ndim_state)
        H = np.zeros((2, self.ndim_state), dtype=np.float64)
        H[:, self.mapping[0]] = J_xyz[:, 0]  # d/dx
        H[:, self.mapping[1]] = J_xyz[:, 1]  # d/dy
        H[:, self.mapping[2]] = J_xyz[:, 2]  # d/dz

        return H


class EKFUpdater3D:
    """Extended Kalman Filter updater for 3D azimuth-elevation measurements.

    Performs the update step of the EKF using azimuth and elevation measurements.

    State vector: [x, vx, y, vy, z, vz] (6D)
    Measurement vector: [azimuth, elevation] (2D)

    Update equations:
        H = jacobian(x_pred)
        S = H @ P_pred @ H.T + R          # Innovation covariance
        K = P_pred @ H.T @ inv(S)         # Kalman gain
        innovation = z - h(x_pred)        # Measurement residual
        x_post = x_pred + K @ innovation  # Posterior mean
        P_post = (I - K @ H) @ P_pred     # Posterior covariance
    """

    def __init__(
        self,
        noise_covariance: NDArray[np.float64],
        mapping: tuple[int, int, int] = (0, 2, 4),
        ndim_state: int = 6,
    ) -> None:
        """Initialize the 3D EKF updater.

        Args:
            noise_covariance: 2x2 measurement noise covariance matrix.
            mapping: Tuple of (x_index, y_index, z_index) into the state vector.
            ndim_state: Dimension of the state vector.
        """
        self.noise_covariance = np.asarray(noise_covariance, dtype=np.float64)
        self.mapping = mapping
        self.ndim_state = ndim_state
        self.measurement_model = AzimuthElevationMeasurementModel(
            noise_covariance=noise_covariance,
            mapping=mapping,
            ndim_state=ndim_state,
        )

    def update(
        self,
        predicted_mean: NDArray[np.float64],
        predicted_covariance: NDArray[np.float64],
        measurement: NDArray[np.float64],
        sensor_position: tuple[float, float, float] | None = None,
        sensor_rotation: tuple[float, float] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Perform the EKF update step.

        Args:
            predicted_mean: Predicted state mean vector (6,) or (6, 1).
            predicted_covariance: Predicted covariance matrix (6, 6).
            measurement: Measurement vector [azimuth, elevation] in radians.
            sensor_position: Sensor (x, y, z) position for measurement model.
                If None, uses (0, 0, 0).
            sensor_rotation: Sensor orientation as (pitch, yaw) in radians.
                If None, no rotation is applied.

        Returns:
            Tuple of (posterior_mean, posterior_covariance):
            - posterior_mean: Updated state mean (6,) array.
            - posterior_covariance: Updated covariance (6, 6) matrix.
        """
        # Ensure inputs are proper numpy arrays
        x_pred = np.asarray(predicted_mean, dtype=np.float64).flatten()
        P_pred = np.asarray(predicted_covariance, dtype=np.float64)
        z = np.asarray(measurement, dtype=np.float64).flatten()

        ndim = len(x_pred)

        # Create measurement model with appropriate sensor position and rotation
        if sensor_position is not None or sensor_rotation is not None:
            meas_model = AzimuthElevationMeasurementModel(
                noise_covariance=self.noise_covariance,
                mapping=self.mapping,
                ndim_state=self.ndim_state,
                translation_offset=(
                    sensor_position if sensor_position else (0.0, 0.0, 0.0)
                ),
                rotation_offset=sensor_rotation,
            )
        else:
            meas_model = self.measurement_model

        # Compute measurement Jacobian and predicted measurement
        H = meas_model.jacobian(x_pred)
        h_pred = meas_model.function(x_pred)
        R = meas_model.noise_covar

        # Innovation covariance: S = H @ P_pred @ H.T + R
        S = H @ P_pred @ H.T + R

        # Kalman gain: K = P_pred @ H.T @ inv(S)
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Innovation (measurement residual) with angle normalization
        innovation = normalize_angles(z - h_pred)

        # Posterior mean: x_post = x_pred + K @ innovation
        x_post = x_pred + K @ innovation

        # Posterior covariance: P_post = (I - K @ H) @ P_pred
        I = np.eye(ndim, dtype=np.float64)
        P_post = (I - K @ H) @ P_pred

        return x_post, P_post
