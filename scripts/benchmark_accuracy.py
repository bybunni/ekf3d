#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ekf3d.ekf_predictor import EKFPredictor3D
from ekf3d.ekf_updater import AzimuthElevationMeasurementModel, EKFUpdater3D


@dataclass
class AccuracyStats:
    interval_s: float
    interval_ms: float
    frames: int
    used_frames: int
    pos_rmse: float
    pos_mae: float
    pos_p50: float
    pos_p95: float
    pos_p99: float
    vel_rmse: float
    vel_mae: float
    vel_p50: float
    vel_p95: float
    vel_p99: float
    pos_x_rmse: float
    pos_y_rmse: float
    pos_z_rmse: float
    vel_x_rmse: float
    vel_y_rmse: float
    vel_z_rmse: float


@dataclass
class AccuracyTrace:
    interval_s: float
    interval_ms: float
    time_s: np.ndarray
    warmup_index: int
    true_position: np.ndarray
    estimated_position: np.ndarray
    ownship_position: np.ndarray
    pos_error_norm: np.ndarray
    vel_error_norm: np.ndarray


def _parse_interval_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one measurement interval is required")
    for value in values:
        if value <= 0.0:
            raise ValueError(f"measurement intervals must be > 0.0, got {value}")
    return values


def _resolve_steps(interval_s: float, steps: int, duration_s: float | None) -> int:
    if duration_s is None:
        return steps
    if duration_s <= 0.0:
        raise ValueError(f"duration must be > 0.0, got {duration_s}")
    return max(1, int(np.ceil(duration_s / interval_s)))


def _build_noise_covariance(scenario: str) -> np.ndarray:
    if scenario == "stress":
        return np.diag([1e-12, 1e-12]).astype(np.float64)
    if scenario == "near_singularity":
        return np.diag([1e-3, 1e-3]).astype(np.float64)
    return np.diag([1e-2, 1e-2]).astype(np.float64)


def _initial_state_and_covariance(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    if scenario == "stress":
        state = np.array([120.0, 1.0, 15.0, -0.4, 8.0, 0.2], dtype=np.float64)
        covariance = np.eye(6, dtype=np.float64) * 1e8
    elif scenario == "near_singularity":
        state = np.array([1e-6, 0.1, -1e-6, -0.1, 20.0, 0.1], dtype=np.float64)
        covariance = np.diag([1.0, 0.2, 1.0, 0.2, 2.0, 0.2]).astype(np.float64)
    else:
        state = np.array([150.0, 1.5, -25.0, 0.6, 12.0, -0.3], dtype=np.float64)
        covariance = np.diag([10.0, 2.0, 10.0, 2.0, 8.0, 1.5]).astype(np.float64)
    return state, covariance


def _sensor_pose_for_step(
    step: int, dt: float, target_state: np.ndarray
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Generate a sinusoidal pursuit trajectory around the moving target."""
    t = step * dt
    target_position = target_state[[0, 2, 4]]
    target_velocity = target_state[[1, 3, 5]]

    speed = float(np.linalg.norm(target_velocity))
    if speed > 1e-9:
        forward = target_velocity / speed
    else:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    lateral = np.array([-forward[1], forward[0], 0.0], dtype=np.float64)
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm > 1e-9:
        lateral /= lateral_norm
    else:
        lateral = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    standoff_distance = 120.0
    lateral_amplitude = 30.0
    vertical_base = 8.0
    vertical_amplitude = 6.0
    weave_frequency = 0.15

    offset = (
        -standoff_distance * forward
        + lateral_amplitude * np.sin(weave_frequency * t) * lateral
        + (vertical_base + vertical_amplitude * np.sin(0.5 * weave_frequency * t)) * up
    )
    sensor_position_vec = target_position + offset

    line_of_sight = target_position - sensor_position_vec
    yaw = float(np.arctan2(line_of_sight[1], line_of_sight[0])) + 0.06 * np.sin(0.27 * t)
    horizontal_range = float(np.hypot(line_of_sight[0], line_of_sight[1]))
    pitch = float(np.arctan2(line_of_sight[2], horizontal_range)) + 0.03 * np.sin(0.35 * t)

    sensor_position = (
        float(sensor_position_vec[0]),
        float(sensor_position_vec[1]),
        float(sensor_position_vec[2]),
    )
    sensor_rotation = (pitch, yaw)
    return sensor_position, sensor_rotation


def _random_vector_with_norm(rng: np.random.Generator, norm: float, size: int) -> np.ndarray:
    if norm <= 0.0:
        return np.zeros(size, dtype=np.float64)
    vec = rng.normal(size=size)
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0.0:
        vec = np.ones(size, dtype=np.float64)
        vec_norm = np.linalg.norm(vec)
    return (norm / vec_norm) * vec


def _run_interval_accuracy(
    interval_s: float,
    steps: int,
    warmup_steps: int,
    scenario: str,
    kalman_gain_method: str,
    with_sensor_pose: bool,
    seed: int,
    initial_position_error_norm: float,
    initial_velocity_error_norm: float,
    collect_trace: bool = False,
) -> tuple[AccuracyStats, AccuracyTrace | None]:
    rng = np.random.default_rng(seed)
    predictor = EKFPredictor3D(
        noise_diff_coeff_x=0.01,
        noise_diff_coeff_y=0.01,
        noise_diff_coeff_z=0.01,
    )
    noise_covariance = _build_noise_covariance(scenario)
    updater = EKFUpdater3D(noise_covariance=noise_covariance)

    true_state, estimate_covariance = _initial_state_and_covariance(scenario)
    position_bias = _random_vector_with_norm(rng, initial_position_error_norm, 3)
    velocity_bias = _random_vector_with_norm(rng, initial_velocity_error_norm, 3)
    estimate_state = true_state.copy()
    estimate_state[[0, 2, 4]] += position_bias
    estimate_state[[1, 3, 5]] += velocity_bias

    measurement_std = np.sqrt(np.diag(noise_covariance))

    pos_error_components = np.zeros((steps, 3), dtype=np.float64)
    vel_error_components = np.zeros((steps, 3), dtype=np.float64)
    pos_error_norm_all = np.zeros(steps, dtype=np.float64)
    vel_error_norm_all = np.zeros(steps, dtype=np.float64)

    if collect_trace:
        time_s = np.arange(steps, dtype=np.float64) * interval_s
        true_position = np.zeros((steps, 3), dtype=np.float64)
        estimated_position = np.zeros((steps, 3), dtype=np.float64)
        ownship_position = np.full((steps, 3), np.nan, dtype=np.float64)
    else:
        time_s = None
        true_position = None
        estimated_position = None
        ownship_position = None

    for step in range(steps):
        true_state, _ = predictor.predict(true_state, np.eye(6, dtype=np.float64), interval_s)
        estimate_state_pred, estimate_cov_pred = predictor.predict(
            estimate_state, estimate_covariance, interval_s
        )

        sensor_position: tuple[float, float, float] | None = None
        sensor_rotation: tuple[float, float] | None = None
        if with_sensor_pose:
            sensor_position, sensor_rotation = _sensor_pose_for_step(
                step, interval_s, true_state
            )

        measurement_model = AzimuthElevationMeasurementModel(
            noise_covariance=noise_covariance,
            mapping=updater.mapping,
            ndim_state=updater.ndim_state,
            translation_offset=sensor_position or (0.0, 0.0, 0.0),
            rotation_offset=sensor_rotation,
        )
        measurement = measurement_model.function(true_state)
        measurement = measurement + rng.normal(loc=0.0, scale=measurement_std, size=2)

        estimate_state, estimate_covariance = updater.update(
            estimate_state_pred,
            estimate_cov_pred,
            measurement,
            sensor_position=sensor_position,
            sensor_rotation=sensor_rotation,
            kalman_gain_method=kalman_gain_method,
        )

        pos_error_components[step] = estimate_state[[0, 2, 4]] - true_state[[0, 2, 4]]
        vel_error_components[step] = estimate_state[[1, 3, 5]] - true_state[[1, 3, 5]]
        pos_error_norm_all[step] = float(np.linalg.norm(pos_error_components[step]))
        vel_error_norm_all[step] = float(np.linalg.norm(vel_error_components[step]))

        if collect_trace:
            true_position[step] = true_state[[0, 2, 4]]
            estimated_position[step] = estimate_state[[0, 2, 4]]
            if sensor_position is not None:
                ownship_position[step] = np.asarray(sensor_position, dtype=np.float64)

    start_index = min(max(warmup_steps, 0), steps - 1)
    pos_error_components = pos_error_components[start_index:]
    vel_error_components = vel_error_components[start_index:]
    used_frames = len(pos_error_components)

    pos_error_norm = np.linalg.norm(pos_error_components, axis=1)
    vel_error_norm = np.linalg.norm(vel_error_components, axis=1)

    stats = AccuracyStats(
        interval_s=interval_s,
        interval_ms=interval_s * 1e3,
        frames=steps,
        used_frames=used_frames,
        pos_rmse=float(np.sqrt(np.mean(pos_error_norm**2))),
        pos_mae=float(np.mean(np.abs(pos_error_norm))),
        pos_p50=float(np.percentile(pos_error_norm, 50)),
        pos_p95=float(np.percentile(pos_error_norm, 95)),
        pos_p99=float(np.percentile(pos_error_norm, 99)),
        vel_rmse=float(np.sqrt(np.mean(vel_error_norm**2))),
        vel_mae=float(np.mean(np.abs(vel_error_norm))),
        vel_p50=float(np.percentile(vel_error_norm, 50)),
        vel_p95=float(np.percentile(vel_error_norm, 95)),
        vel_p99=float(np.percentile(vel_error_norm, 99)),
        pos_x_rmse=float(np.sqrt(np.mean(pos_error_components[:, 0] ** 2))),
        pos_y_rmse=float(np.sqrt(np.mean(pos_error_components[:, 1] ** 2))),
        pos_z_rmse=float(np.sqrt(np.mean(pos_error_components[:, 2] ** 2))),
        vel_x_rmse=float(np.sqrt(np.mean(vel_error_components[:, 0] ** 2))),
        vel_y_rmse=float(np.sqrt(np.mean(vel_error_components[:, 1] ** 2))),
        vel_z_rmse=float(np.sqrt(np.mean(vel_error_components[:, 2] ** 2))),
    )

    trace: AccuracyTrace | None = None
    if collect_trace:
        trace = AccuracyTrace(
            interval_s=interval_s,
            interval_ms=interval_s * 1e3,
            time_s=time_s,
            warmup_index=start_index,
            true_position=true_position,
            estimated_position=estimated_position,
            ownship_position=ownship_position,
            pos_error_norm=pos_error_norm_all,
            vel_error_norm=vel_error_norm_all,
        )

    return stats, trace


def _print_accuracy_table(stats_by_interval: list[AccuracyStats]) -> None:
    print(
        " interval_ms | pos_rmse | pos_p95 | vel_rmse | vel_p95 | "
        "pos_axis_rmse(x,y,z) | vel_axis_rmse(x,y,z)"
    )
    print("-" * 124)
    for stats in stats_by_interval:
        print(
            f"{stats.interval_ms:11.3f} | "
            f"{stats.pos_rmse:8.4f} | {stats.pos_p95:7.4f} | "
            f"{stats.vel_rmse:8.4f} | {stats.vel_p95:7.4f} | "
            f"({stats.pos_x_rmse:6.3f},{stats.pos_y_rmse:6.3f},{stats.pos_z_rmse:6.3f}) | "
            f"({stats.vel_x_rmse:6.3f},{stats.vel_y_rmse:6.3f},{stats.vel_z_rmse:6.3f})"
        )


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for --plot-dir. "
            "Run with: uv run --with matplotlib python scripts/benchmark_accuracy.py ..."
        ) from exc
    return plt


def _interval_tag(interval_ms: float) -> str:
    return f"{interval_ms:.3f}ms".replace(".", "p")


def _generate_summary_plot(stats_by_interval: list[AccuracyStats], plot_dir: Path) -> Path:
    plt = _import_pyplot()
    plot_dir.mkdir(parents=True, exist_ok=True)

    intervals_ms = np.array([stats.interval_ms for stats in stats_by_interval], dtype=np.float64)
    pos_rmse = np.array([stats.pos_rmse for stats in stats_by_interval], dtype=np.float64)
    pos_p95 = np.array([stats.pos_p95 for stats in stats_by_interval], dtype=np.float64)
    vel_rmse = np.array([stats.vel_rmse for stats in stats_by_interval], dtype=np.float64)
    vel_p95 = np.array([stats.vel_p95 for stats in stats_by_interval], dtype=np.float64)

    fig, (ax_pos, ax_vel) = plt.subplots(1, 2, figsize=(13, 5))

    ax_pos.plot(intervals_ms, pos_rmse, marker="o", label="Position RMSE")
    ax_pos.plot(intervals_ms, pos_p95, marker="o", label="Position P95")
    ax_pos.set_xscale("log")
    ax_pos.set_yscale("log")
    ax_pos.set_xlabel("Measurement interval (ms)")
    ax_pos.set_ylabel("Position error")
    ax_pos.set_title("Position Error vs Interval")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend()

    ax_vel.plot(intervals_ms, vel_rmse, marker="o", label="Velocity RMSE")
    ax_vel.plot(intervals_ms, vel_p95, marker="o", label="Velocity P95")
    ax_vel.set_xscale("log")
    ax_vel.set_yscale("log")
    ax_vel.set_xlabel("Measurement interval (ms)")
    ax_vel.set_ylabel("Velocity error")
    ax_vel.set_title("Velocity Error vs Interval")
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend()

    fig.tight_layout()
    out_path = plot_dir / "accuracy_summary.png"
    fig.savefig(out_path, dpi=180)
    return out_path


def _generate_interval_plot(stats: AccuracyStats, trace: AccuracyTrace, plot_dir: Path) -> Path:
    plt = _import_pyplot()
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9))
    ax_xy = fig.add_subplot(2, 2, 1)
    ax_z = fig.add_subplot(2, 2, 2)
    ax_pos_err = fig.add_subplot(2, 2, 3)
    ax_vel_err = fig.add_subplot(2, 2, 4)

    # XY trajectories
    ax_xy.plot(
        trace.true_position[:, 0],
        trace.true_position[:, 1],
        label="Target (true)",
        linewidth=2,
    )
    ax_xy.plot(
        trace.estimated_position[:, 0],
        trace.estimated_position[:, 1],
        label="Target (estimate)",
        linewidth=1.8,
    )
    ownship_valid = np.isfinite(trace.ownship_position[:, 0])
    if np.any(ownship_valid):
        ax_xy.plot(
            trace.ownship_position[ownship_valid, 0],
            trace.ownship_position[ownship_valid, 1],
            label="Ownship",
            linewidth=1.6,
            alpha=0.85,
        )
    ax_xy.set_title("XY Trajectories")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend()

    # Z over time
    ax_z.plot(trace.time_s, trace.true_position[:, 2], label="Target Z (true)", linewidth=2)
    ax_z.plot(
        trace.time_s,
        trace.estimated_position[:, 2],
        label="Target Z (estimate)",
        linewidth=1.8,
    )
    if np.any(ownship_valid):
        ax_z.plot(
            trace.time_s[ownship_valid],
            trace.ownship_position[ownship_valid, 2],
            label="Ownship Z",
            linewidth=1.6,
            alpha=0.85,
        )
    ax_z.set_title("Z vs Time")
    ax_z.set_xlabel("Time (s)")
    ax_z.set_ylabel("Z")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend()

    # Error curves
    ax_pos_err.plot(trace.time_s, trace.pos_error_norm, color="tab:red", linewidth=1.2)
    ax_pos_err.axvline(
        trace.time_s[trace.warmup_index], color="k", linestyle="--", linewidth=1, label="Warmup end"
    )
    ax_pos_err.axhline(
        stats.pos_rmse,
        color="tab:red",
        linestyle=":",
        linewidth=1.1,
        label=f"RMSE={stats.pos_rmse:.3f}",
    )
    ax_pos_err.set_title("Position Error Norm")
    ax_pos_err.set_xlabel("Time (s)")
    ax_pos_err.set_ylabel("||position error||")
    ax_pos_err.grid(True, alpha=0.3)
    ax_pos_err.legend()

    ax_vel_err.plot(trace.time_s, trace.vel_error_norm, color="tab:blue", linewidth=1.2)
    ax_vel_err.axvline(
        trace.time_s[trace.warmup_index], color="k", linestyle="--", linewidth=1, label="Warmup end"
    )
    ax_vel_err.axhline(
        stats.vel_rmse,
        color="tab:blue",
        linestyle=":",
        linewidth=1.1,
        label=f"RMSE={stats.vel_rmse:.3f}",
    )
    ax_vel_err.set_title("Velocity Error Norm")
    ax_vel_err.set_xlabel("Time (s)")
    ax_vel_err.set_ylabel("||velocity error||")
    ax_vel_err.grid(True, alpha=0.3)
    ax_vel_err.legend()

    fig.suptitle(
        f"Accuracy Diagnostics ({stats.interval_ms:.3f} ms interval, frames={stats.frames})",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = plot_dir / f"accuracy_interval_{_interval_tag(stats.interval_ms)}.png"
    fig.savefig(out_path, dpi=180)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EKF accuracy benchmark. Reports state-estimation error versus "
            "measurement interval."
        )
    )
    parser.add_argument(
        "--measurement-intervals",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0",
        help="comma-separated measurement intervals in seconds",
    )
    parser.add_argument("--steps", type=int, default=20000, help="frames to simulate")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help=(
            "fixed simulated duration in seconds. If set, steps are computed per interval "
            "as ceil(duration / interval)."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="initial frames to ignore in metric aggregation",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--scenario",
        choices=["nominal", "near_singularity", "stress"],
        default="nominal",
        help="state/covariance operating regime",
    )
    parser.add_argument(
        "--kalman-gain-method",
        choices=["inv", "solve"],
        default="inv",
        help="Kalman gain implementation path",
    )
    parser.add_argument(
        "--with-sensor-pose",
        action="store_true",
        help="include changing sensor position/orientation each frame",
    )
    parser.add_argument(
        "--initial-position-error-norm",
        type=float,
        default=50.0,
        help="norm of initial position error injected into the estimator",
    )
    parser.add_argument(
        "--initial-velocity-error-norm",
        type=float,
        default=1.0,
        help="norm of initial velocity error injected into the estimator",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional JSON output path for automated comparisons",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help=(
            "optional directory for generated plots (summary + per-interval trajectory/error). "
            "Requires matplotlib."
        ),
    )
    args = parser.parse_args()

    intervals = _parse_interval_list(args.measurement_intervals)
    stats_by_interval: list[AccuracyStats] = []
    traces: list[tuple[AccuracyStats, AccuracyTrace]] = []
    for interval_s in intervals:
        interval_steps = _resolve_steps(interval_s, args.steps, args.duration_s)
        stats, trace = _run_interval_accuracy(
            interval_s=interval_s,
            steps=interval_steps,
            warmup_steps=args.warmup_steps,
            scenario=args.scenario,
            kalman_gain_method=args.kalman_gain_method,
            with_sensor_pose=args.with_sensor_pose,
            seed=args.seed,
            initial_position_error_norm=args.initial_position_error_norm,
            initial_velocity_error_norm=args.initial_velocity_error_norm,
            collect_trace=args.plot_dir is not None,
        )
        stats_by_interval.append(stats)
        if trace is not None:
            traces.append((stats, trace))

    print("EKF Accuracy Benchmark")
    print(
        f"scenario={args.scenario} gain={args.kalman_gain_method} "
        f"sensor_pose={'on' if args.with_sensor_pose else 'off'} "
        f"steps={'duration-based' if args.duration_s is not None else args.steps} "
        f"warmup={args.warmup_steps}"
    )
    if args.duration_s is not None:
        print(f"requested_duration_s={args.duration_s}")
    print(
        f"initial_position_error_norm={args.initial_position_error_norm} "
        f"initial_velocity_error_norm={args.initial_velocity_error_norm}"
    )
    _print_accuracy_table(stats_by_interval)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scenario": args.scenario,
            "kalman_gain_method": args.kalman_gain_method,
            "with_sensor_pose": args.with_sensor_pose,
            "steps": args.steps,
            "duration_s": args.duration_s,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
            "initial_position_error_norm": args.initial_position_error_norm,
            "initial_velocity_error_norm": args.initial_velocity_error_norm,
            "results": [asdict(stats) for stats in stats_by_interval],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON results to {args.json_out}")

    if args.plot_dir is not None:
        summary_plot = _generate_summary_plot(stats_by_interval, args.plot_dir)
        print(f"Wrote plot to {summary_plot}")
        for stats, trace in traces:
            out_path = _generate_interval_plot(stats, trace, args.plot_dir)
            print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
