#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ekf3d.ekf_predictor import EKFPredictor3D
from ekf3d.ekf_updater import AzimuthElevationMeasurementModel, EKFUpdater3D


@dataclass
class FrameTimingStats:
    interval_s: float
    interval_ms: float
    frames: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    miss_count: int
    miss_rate_percent: float
    worst_overrun_ms: float
    mean_slack_ms: float
    p05_slack_ms: float
    p50_slack_ms: float
    utilization_p95_percent: float
    estimated_p95_headroom_hz: float


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


def _run_interval_benchmark(
    interval_s: float,
    steps: int,
    scenario: str,
    kalman_gain_method: str,
    with_sensor_pose: bool,
    seed: int,
) -> FrameTimingStats:
    rng = np.random.default_rng(seed)
    predictor = EKFPredictor3D(
        noise_diff_coeff_x=0.01,
        noise_diff_coeff_y=0.01,
        noise_diff_coeff_z=0.01,
    )
    noise_covariance = _build_noise_covariance(scenario)
    updater = EKFUpdater3D(noise_covariance=noise_covariance)

    estimate_state, estimate_covariance = _initial_state_and_covariance(scenario)
    true_state = estimate_state.copy()
    measurement_std = np.sqrt(np.diag(noise_covariance))

    frame_durations_ns = np.empty(steps, dtype=np.int64)
    budget_ns = interval_s * 1e9

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

        frame_start_ns = time.perf_counter_ns()
        estimate_state, estimate_covariance = updater.update(
            estimate_state_pred,
            estimate_cov_pred,
            measurement,
            sensor_position=sensor_position,
            sensor_rotation=sensor_rotation,
            kalman_gain_method=kalman_gain_method,
        )
        frame_end_ns = time.perf_counter_ns()
        frame_durations_ns[step] = frame_end_ns - frame_start_ns

    durations_ms = frame_durations_ns.astype(np.float64) / 1e6
    slack_ns = budget_ns - frame_durations_ns.astype(np.float64)
    miss_mask = slack_ns < 0.0
    miss_count = int(np.count_nonzero(miss_mask))
    worst_overrun_ms = float(max(0.0, -slack_ns.min() / 1e6))

    p95_ms = float(np.percentile(durations_ms, 95))
    return FrameTimingStats(
        interval_s=interval_s,
        interval_ms=interval_s * 1e3,
        frames=steps,
        mean_ms=float(np.mean(durations_ms)),
        p50_ms=float(np.percentile(durations_ms, 50)),
        p95_ms=p95_ms,
        p99_ms=float(np.percentile(durations_ms, 99)),
        max_ms=float(np.max(durations_ms)),
        miss_count=miss_count,
        miss_rate_percent=100.0 * miss_count / steps,
        worst_overrun_ms=worst_overrun_ms,
        mean_slack_ms=float(np.mean(slack_ns) / 1e6),
        p05_slack_ms=float(np.percentile(slack_ns, 5) / 1e6),
        p50_slack_ms=float(np.percentile(slack_ns, 50) / 1e6),
        utilization_p95_percent=100.0 * p95_ms / (interval_s * 1e3),
        estimated_p95_headroom_hz=(1000.0 / p95_ms) if p95_ms > 0.0 else float("inf"),
    )


def _print_table(stats_by_interval: list[FrameTimingStats]) -> None:
    print(
        " interval_ms | mean_ms | p95_ms | p99_ms | miss_% | worst_overrun_ms | "
        "mean_slack_ms | p05_slack_ms | util_p95_% | est_headroom_hz"
    )
    print("-" * 128)
    for stats in stats_by_interval:
        print(
            f"{stats.interval_ms:11.3f} | {stats.mean_ms:7.4f} | {stats.p95_ms:6.4f} | "
            f"{stats.p99_ms:6.4f} | {stats.miss_rate_percent:6.3f} | "
            f"{stats.worst_overrun_ms:16.6f} | {stats.mean_slack_ms:13.6f} | "
            f"{stats.p05_slack_ms:12.6f} | {stats.utilization_p95_percent:10.3f} | "
            f"{stats.estimated_p95_headroom_hz:15.2f}"
        )


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for --plot-out. "
            "Run with: uv run --with matplotlib python scripts/benchmark_realtime.py ..."
        ) from exc
    return plt


def _generate_plot(stats_by_interval: list[FrameTimingStats], plot_out: Path) -> Path:
    plt = _import_pyplot()
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    intervals_ms = np.array([stats.interval_ms for stats in stats_by_interval], dtype=np.float64)
    mean_ms = np.array([stats.mean_ms for stats in stats_by_interval], dtype=np.float64)
    p95_ms = np.array([stats.p95_ms for stats in stats_by_interval], dtype=np.float64)
    p99_ms = np.array([stats.p99_ms for stats in stats_by_interval], dtype=np.float64)
    miss_rate = np.array(
        [stats.miss_rate_percent for stats in stats_by_interval], dtype=np.float64
    )
    utilization = np.array(
        [stats.utilization_p95_percent for stats in stats_by_interval], dtype=np.float64
    )

    fig, (ax_latency, ax_deadline) = plt.subplots(1, 2, figsize=(13, 5))

    ax_latency.plot(intervals_ms, mean_ms, marker="o", label="Mean frame time")
    ax_latency.plot(intervals_ms, p95_ms, marker="o", label="P95 frame time")
    ax_latency.plot(intervals_ms, p99_ms, marker="o", label="P99 frame time")
    ax_latency.plot(intervals_ms, intervals_ms, "--", label="Deadline budget")
    ax_latency.set_xscale("log")
    ax_latency.set_xlabel("Measurement interval (ms)")
    ax_latency.set_ylabel("Frame time (ms)")
    ax_latency.set_title("Realtime Latency vs Interval")
    ax_latency.grid(True, alpha=0.3)
    ax_latency.legend()

    ax_deadline.plot(intervals_ms, miss_rate, marker="o", color="tab:red", label="Miss rate (%)")
    ax_deadline.set_xscale("log")
    ax_deadline.set_xlabel("Measurement interval (ms)")
    ax_deadline.set_ylabel("Miss rate (%)", color="tab:red")
    ax_deadline.tick_params(axis="y", labelcolor="tab:red")
    ax_deadline.grid(True, alpha=0.3)

    ax_util = ax_deadline.twinx()
    ax_util.plot(
        intervals_ms,
        utilization,
        marker="s",
        color="tab:blue",
        label="P95 utilization (%)",
    )
    ax_util.set_ylabel("P95 utilization (%)", color="tab:blue")
    ax_util.tick_params(axis="y", labelcolor="tab:blue")

    ax_deadline.set_title("Deadline Health")
    lines, labels = ax_deadline.get_legend_handles_labels()
    lines2, labels2 = ax_util.get_legend_handles_labels()
    ax_deadline.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(plot_out, dpi=180)
    return plot_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime EKF frame-time benchmark. Reports deadline miss/slack metrics "
            "for each measurement interval."
        )
    )
    parser.add_argument(
        "--measurement-intervals",
        type=str,
        default="0.033333,0.02,0.01,0.005",
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
        "--json-out",
        type=Path,
        default=None,
        help="optional JSON output path for automated comparisons",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="optional plot output path (PNG). Requires matplotlib.",
    )
    args = parser.parse_args()

    intervals = _parse_interval_list(args.measurement_intervals)
    stats_by_interval: list[FrameTimingStats] = []
    for interval_s in intervals:
        interval_steps = _resolve_steps(interval_s, args.steps, args.duration_s)
        stats = _run_interval_benchmark(
            interval_s=interval_s,
            steps=interval_steps,
            scenario=args.scenario,
            kalman_gain_method=args.kalman_gain_method,
            with_sensor_pose=args.with_sensor_pose,
            seed=args.seed,
        )
        stats_by_interval.append(stats)

    print("EKF Realtime Frame-Time Benchmark")
    print(
        f"scenario={args.scenario} gain={args.kalman_gain_method} "
        f"sensor_pose={'on' if args.with_sensor_pose else 'off'} "
        f"steps={'duration-based' if args.duration_s is not None else args.steps}"
    )
    if args.duration_s is not None:
        print(f"requested_duration_s={args.duration_s}")
    _print_table(stats_by_interval)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scenario": args.scenario,
            "kalman_gain_method": args.kalman_gain_method,
            "with_sensor_pose": args.with_sensor_pose,
            "steps": args.steps,
            "duration_s": args.duration_s,
            "seed": args.seed,
            "results": [asdict(stats) for stats in stats_by_interval],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON results to {args.json_out}")

    if args.plot_out is not None:
        out_path = _generate_plot(stats_by_interval, args.plot_out)
        print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
