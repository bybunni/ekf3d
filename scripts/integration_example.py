#!/usr/bin/env python3
"""Simple end-user integration example for ekf3d.

Flow:
1. Initialize predictor/updater and filter state.
2. Update with each sensor measurement and sensor pose (position + orientation).
3. Read estimated target position and velocity.
4. Repeat for the next measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ekf3d.ekf_predictor import EKFPredictor3D
from ekf3d.ekf_updater import AzimuthElevationMeasurementModel, EKFUpdater3D


@dataclass(frozen=True)
class SensorMeasurement:
    timestamp_s: float
    azimuth_rad: float
    elevation_rad: float
    sensor_position_m: tuple[float, float, float]
    sensor_rotation_rad: tuple[float, float]


@dataclass(frozen=True)
class Estimate:
    timestamp_s: float
    position_m: np.ndarray
    velocity_mps: np.ndarray


def _propagate_target_constant_velocity(state: np.ndarray, dt_s: float) -> np.ndarray:
    next_state = state.copy()
    next_state[0] += next_state[1] * dt_s
    next_state[2] += next_state[3] * dt_s
    next_state[4] += next_state[5] * dt_s
    return next_state


def _sensor_pose_for_demo(t_s: float, target_state: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float]]:
    ownship_pos = np.array(
        [
            80.0 * np.cos(0.08 * t_s),
            80.0 * np.sin(0.08 * t_s),
            10.0 + 2.0 * np.sin(0.16 * t_s),
        ],
        dtype=np.float64,
    )

    target_pos = target_state[[0, 2, 4]]
    line_of_sight = target_pos - ownship_pos
    yaw = float(np.arctan2(line_of_sight[1], line_of_sight[0]))
    horizontal_range = float(np.hypot(line_of_sight[0], line_of_sight[1]))
    pitch = float(np.arctan2(line_of_sight[2], horizontal_range))

    return (
        (float(ownship_pos[0]), float(ownship_pos[1]), float(ownship_pos[2])),
        (pitch, yaw),
    )


def generate_demo_measurements(
    *,
    steps: int,
    dt_s: float,
    azimuth_noise_std_rad: float,
    elevation_noise_std_rad: float,
    seed: int,
) -> list[SensorMeasurement]:
    rng = np.random.default_rng(seed)
    target_state = np.array([150.0, 1.2, -40.0, 0.8, 25.0, -0.15], dtype=np.float64)

    measurements: list[SensorMeasurement] = []
    for step in range(steps):
        t_s = step * dt_s
        target_state = _propagate_target_constant_velocity(target_state, dt_s)
        sensor_position, sensor_rotation = _sensor_pose_for_demo(t_s, target_state)

        model = AzimuthElevationMeasurementModel(
            noise_covariance=np.diag(
                [azimuth_noise_std_rad**2, elevation_noise_std_rad**2]
            ).astype(np.float64),
            translation_offset=sensor_position,
            rotation_offset=sensor_rotation,
        )
        measurement = model.function(target_state)
        measurement += rng.normal(
            loc=0.0,
            scale=np.array([azimuth_noise_std_rad, elevation_noise_std_rad], dtype=np.float64),
            size=2,
        )

        measurements.append(
            SensorMeasurement(
                timestamp_s=t_s,
                azimuth_rad=float(measurement[0]),
                elevation_rad=float(measurement[1]),
                sensor_position_m=sensor_position,
                sensor_rotation_rad=sensor_rotation,
            )
        )

    return measurements


def run_filter(
    measurements: Iterable[SensorMeasurement],
    *,
    dt_s: float,
    process_noise_q: float,
    measurement_noise_std_rad: float,
) -> list[Estimate]:
    # 1) Initialization
    predictor = EKFPredictor3D(
        noise_diff_coeff_x=process_noise_q,
        noise_diff_coeff_y=process_noise_q,
        noise_diff_coeff_z=process_noise_q,
    )
    noise_covariance = np.diag(
        [measurement_noise_std_rad**2, measurement_noise_std_rad**2]
    ).astype(np.float64)
    updater = EKFUpdater3D(noise_covariance=noise_covariance)

    x_est = np.array([120.0, 0.0, -20.0, 0.0, 20.0, 0.0], dtype=np.float64)
    p_est = np.diag([400.0, 25.0, 400.0, 25.0, 200.0, 16.0]).astype(np.float64)

    estimates: list[Estimate] = []
    for measurement in measurements:
        # 2) Predict + update using measurement and current sensor pose
        x_pred, p_pred = predictor.predict(x_est, p_est, dt_s)
        z = np.array([measurement.azimuth_rad, measurement.elevation_rad], dtype=np.float64)
        x_est, p_est = updater.update(
            x_pred,
            p_pred,
            z,
            sensor_position=measurement.sensor_position_m,
            sensor_rotation=measurement.sensor_rotation_rad,
        )

        # 3) Consume estimated target position and velocity
        estimates.append(
            Estimate(
                timestamp_s=measurement.timestamp_s,
                position_m=x_est[[0, 2, 4]].copy(),
                velocity_mps=x_est[[1, 3, 5]].copy(),
            )
        )
        # 4) Loop to next measurement

    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple end-user EKF integration example.")
    parser.add_argument("--steps", type=int, default=15, help="number of measurements")
    parser.add_argument("--dt-s", type=float, default=1.0, help="measurement interval seconds")
    parser.add_argument("--seed", type=int, default=7, help="random seed for synthetic demo data")
    parser.add_argument(
        "--process-noise-q",
        type=float,
        default=0.01,
        help="process noise diffusion coefficient for x/y/z CV models",
    )
    parser.add_argument(
        "--measurement-noise-std-rad",
        type=float,
        default=0.001,
        help="azimuth/elevation measurement noise standard deviation in radians",
    )
    args = parser.parse_args()

    measurements = generate_demo_measurements(
        steps=args.steps,
        dt_s=args.dt_s,
        azimuth_noise_std_rad=args.measurement_noise_std_rad,
        elevation_noise_std_rad=args.measurement_noise_std_rad,
        seed=args.seed,
    )
    estimates = run_filter(
        measurements,
        dt_s=args.dt_s,
        process_noise_q=args.process_noise_q,
        measurement_noise_std_rad=args.measurement_noise_std_rad,
    )

    print("timestamp_s, est_x_m, est_y_m, est_z_m, est_vx_mps, est_vy_mps, est_vz_mps")
    for est in estimates:
        print(
            f"{est.timestamp_s:9.3f}, "
            f"{est.position_m[0]:8.3f}, {est.position_m[1]:8.3f}, {est.position_m[2]:8.3f}, "
            f"{est.velocity_mps[0]:7.3f}, {est.velocity_mps[1]:7.3f}, {est.velocity_mps[2]:7.3f}"
        )


if __name__ == "__main__":
    main()
