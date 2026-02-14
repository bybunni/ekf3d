# 3D EKF Bearing-Only Tracker

This document describes the minimum set of components needed to run the 3D Extended Kalman Filter and the data flow from sensor measurements through each stage.

## Assumptions

- **Target**: Constant velocity. The EKF motion model assumes the target travels in a straight line at constant speed, with process noise allowing for small deviations.
- **Sensor (ownship)**: Maneuvering. The sensor platform changes position and orientation over time. Its pose is provided to the filter each timestep as known inputs (`sensor_position`, `sensor_rotation`) -- the filter does not estimate the ownship state, it only estimates the target. Ownship maneuver is what provides the changing geometry that makes range observable from angle-only measurements.

## Coordinate Frames

### World frame

The EKF state vector `[x, vx, y, vy, z, vz]` and the `sensor_position` tuple are both expressed in a common world frame (e.g. ENU, NED, or ECEF -- whichever your system uses). The filter is frame-agnostic as long as both are in the same one.

### Sensor body frame (measurements)

Measurements `[azimuth, elevation]` are in the **sensor body frame** -- angles are relative to the sensor's boresight, not world axes.

- **Azimuth**: `atan2(y_body, x_body)` -- output range is `(-pi, +pi]`. 0 is dead ahead (boresight / nose), positive values are counter-clockwise (left) when viewed from above, negative values are clockwise (right). +pi/-pi is directly behind the sensor.
- **Elevation**: `arcsin(z_body / range)` -- 0 is the horizontal plane of the sensor, positive up.

This matches typical sensor output: the sensor reports where the target is relative to itself.

### Ownship orientation (`sensor_rotation`)

The `sensor_rotation=(pitch, yaw)` parameter tells the measurement model how the sensor body frame is oriented relative to the world frame. These are **Euler angles in radians** for a roll-stabilized sensor (roll = 0):

- **`pitch`**: nose-up rotation about the body y-axis (positive = nose up)
- **`yaw`**: heading rotation about the body z-axis (positive = counter-clockwise from world +x when viewed from above)

The rotation matrix is constructed as `R = Ry(pitch) @ Rz(-yaw)`, following the Stone Soup convention. This `R` transforms a vector from world frame to sensor body frame.

**What the measurement model does with these:**

1. Computes the target-to-sensor relative position in world frame: `d = x_target - x_sensor`
2. Rotates into sensor body frame: `d_body = R @ d`
3. Computes azimuth and elevation from `d_body`
4. Compares against the actual sensor measurement to produce the innovation

This means the Jacobian `H` also accounts for the rotation, so the Kalman gain correctly maps angle-space innovations back into world-frame state corrections.

**If your system provides the full rotation matrix** instead of Euler angles, you would replace the `_build_rotation_matrix` method or convert to `(pitch, yaw)` before calling update. For a roll-stabilized sensor this is:

```python
# Extract pitch and yaw from a 3x3 rotation matrix R_world_to_body
pitch = np.arcsin(R[0, 2])
yaw = -np.arctan2(R[1, 0], R[0, 0])   # note negation (Stone Soup convention)
```

## Components

There are three classes, all in two files with no dependencies beyond NumPy.

| Class | File | Role |
|---|---|---|
| `EKFPredictor3D` | `ekf_predictor.py` | Time-propagate state and covariance forward by `dt` |
| `AzimuthElevationMeasurementModel` | `ekf_updater.py` | Convert Cartesian state to `[azimuth, elevation]` and compute the Jacobian |
| `EKFUpdater3D` | `ekf_updater.py` | Fuse a measurement into the predicted state (owns a measurement model internally) |

Internal pieces (you never instantiate these directly):

- `ConstantVelocityModel` -- a single-axis 2x2 constant-velocity motion model
- `CombinedCVModel3D` -- assembles three `ConstantVelocityModel` instances into one 6x6 model
- `normalize_angles()` -- wraps angle residuals to `[-pi, pi]`

### How ConstantVelocityModel and CombinedCVModel3D interact

Both are required, but both are internal to `EKFPredictor3D` -- you never touch them.

`ConstantVelocityModel` is the atomic building block. It models one axis of motion (e.g. x-axis) with a 2-element state `[position, velocity]` and produces the 2x2 transition matrix and 2x2 process noise for that axis:

```
F_axis = [[1, dt],    Q_axis = noise_diff_coeff * [[dt^3/3, dt^2/2],
          [0,  1]]                                  [dt^2/2, dt    ]]
```

`CombinedCVModel3D` creates three of these (one per axis, each with its own `noise_diff_coeff`) and stacks them into 6x6 block-diagonal matrices:

```
CombinedCVModel3D
  ├── ConstantVelocityModel(noise_diff_coeff_x)  →  F_x, Q_x  (indices 0-1)
  ├── ConstantVelocityModel(noise_diff_coeff_y)  →  F_y, Q_y  (indices 2-3)
  └── ConstantVelocityModel(noise_diff_coeff_z)  →  F_z, Q_z  (indices 4-5)

F = [[F_x,  0,   0 ],     Q = [[Q_x,  0,   0 ],
     [ 0,  F_y,  0 ],          [ 0,  Q_y,  0 ],
     [ 0,   0,  F_z]]          [ 0,   0,  Q_z]]
```

`EKFPredictor3D` owns a `CombinedCVModel3D` and calls its `.transition_matrix(dt)` and `.process_noise(dt)` methods during predict. The full ownership chain:

```
EKFPredictor3D                          ← you instantiate this
  └── CombinedCVModel3D                 ← created automatically in __init__
        ├── ConstantVelocityModel (x)   ← created automatically
        ├── ConstantVelocityModel (y)   ← created automatically
        └── ConstantVelocityModel (z)   ← created automatically
```

For a constant-velocity target this is the correct motion model. The `noise_diff_coeff` parameters don't mean the target accelerates -- they represent the filter's allowance for small deviations from perfect constant velocity (sensor noise, atmospheric effects, minor course corrections). For a truly constant-velocity target, keep these values small (e.g. 0.01).

## State Vector

All components operate on a 6D state vector with interleaved position/velocity:

```
x = [x, vx, y, vy, z, vz]
     0   1   2   3   4   5
```

The `mapping` parameter `(0, 2, 4)` tells the measurement model which indices are the position components. If you embed this state inside a larger vector, change `mapping` and `ndim_state` accordingly.

## Data Flow

```
                 sensor_position (x, y, z)
                 sensor_rotation (pitch, yaw)  [optional]
                            |
                            v
  ┌──────────────┐    ┌───────────────┐    measurement
  │              │    │  Azimuth      │    [azimuth, elevation]
  │ EKFPredictor │    │  Elevation    │         |
  │     3D       │    │  Measurement  │         |
  │              │    │  Model        │         |
  └──────┬───────┘    └───────┬───────┘         |
         │                    │                 │
         │  x_pred, P_pred    │  h(x), H(x)    │  z
         │                    │                 │
         └────────────┬───────┘                 │
                      │                         │
                      v                         │
               ┌──────────────┐                 │
               │              │<────────────────┘
               │ EKFUpdater3D │
               │              │
               └──────┬───────┘
                      │
                      v
                x_post, P_post
```

### Per-Timestep Loop

```python
for each timestep:
    1. PREDICT  -->  x_pred, P_pred = predictor.predict(x_post, P_post, dt)
    2. MEASURE  -->  receive [azimuth, elevation] from sensor
    3. UPDATE   -->  x_post, P_post = updater.update(
                         x_pred, P_pred, measurement,
                         sensor_position=...,
                         sensor_rotation=...   # omit if sensor is at origin
                     )
```

On the first timestep, skip the predict and pass your initial state estimate directly to the updater.

## What Each Component Does

### EKFPredictor3D

Propagates the state forward in time assuming constant velocity per axis. Internally delegates to `CombinedCVModel3D` to build `F` and `Q` (see [class hierarchy above](#how-constantvelocitymodel-and-combinedcvmodel3d-interact)).

```
x_pred = F @ x_prior
P_pred = F @ P_prior @ F.T + Q
```

The resulting `F` is 6x6 block-diagonal -- each axis predicts `position += velocity * dt` independently:

```
F = [[1, dt, 0,  0, 0,  0],       x_new  = x  + vx * dt
     [0,  1, 0,  0, 0,  0],       vx_new = vx
     [0,  0, 1, dt, 0,  0],       y_new  = y  + vy * dt
     [0,  0, 0,  1, 0,  0],       vy_new = vy
     [0,  0, 0,  0, 1, dt],       z_new  = z  + vz * dt
     [0,  0, 0,  0, 0,  1]]       vz_new = vz
```

`Q` is also 6x6 block-diagonal. Each 2x2 block is the process noise for one axis, modeling small random accelerations:

```
Q_axis = noise_diff_coeff * [[dt^3/3, dt^2/2],
                              [dt^2/2, dt    ]]
```

**Constructor:**
```python
predictor = EKFPredictor3D(
    noise_diff_coeff_x=0.01,  # process noise per axis
    noise_diff_coeff_y=0.01,  # for a constant-velocity target, keep small
    noise_diff_coeff_z=0.01,
)
```

**Call:**
```python
x_pred, P_pred = predictor.predict(x_prior, P_prior, dt)
```

### AzimuthElevationMeasurementModel

Converts a 6D Cartesian state to the `[azimuth, elevation]` angles that the sensor would observe, and computes the Jacobian `H` needed by the EKF update.

You do not call this directly -- `EKFUpdater3D` creates and uses it internally. It is documented here because it contains the core math you would need to reimplement.

**Measurement function** `h(x)`:

```
dx = x_target - x_sensor       # relative position
dy = y_target - y_sensor
dz = z_target - z_sensor

[dx', dy', dz'] = R @ [dx, dy, dz]   # rotate to sensor frame (identity if no rotation)

azimuth   = atan2(dy', dx')
elevation = arcsin(dz' / sqrt(dx'^2 + dy'^2 + dz'^2))
```

**Jacobian** `H(x)` (2x6, non-zero only at position indices):

```
rho^2 = dx'^2 + dy'^2          # horizontal range squared
r^2   = rho^2 + dz'^2          # 3D range squared
rho   = sqrt(rho^2)

dφ/d[dx',dy',dz'] = [-dy'/rho^2,  dx'/rho^2,  0        ]
dθ/d[dx',dy',dz'] = [-dx'*dz'/(r^2*rho), -dy'*dz'/(r^2*rho), rho/r^2]

H_xyz = J_angles @ R           # chain rule through the rotation
H[:, mapping] = H_xyz          # place into full 6-wide Jacobian
```

**Parameters:**

| Parameter | Type | Purpose |
|---|---|---|
| `noise_covariance` | 2x2 array | `R` matrix: `[[sigma_az^2, 0], [0, sigma_el^2]]` |
| `mapping` | `(int, int, int)` | Position indices in the state vector. Default `(0, 2, 4)` |
| `ndim_state` | `int` | Length of the full state vector. Default `6` |
| `translation_offset` | `(x, y, z)` | Sensor position in world frame |
| `rotation_offset` | `(pitch, yaw)` | Sensor orientation (roll-stabilized). `None` = no rotation |

### EKFUpdater3D

Fuses one `[azimuth, elevation]` measurement into the predicted state.

```
H       = jacobian(x_pred)              # from measurement model
h_pred  = h(x_pred)                     # predicted measurement
S       = H @ P_pred @ H.T + R          # innovation covariance
K       = P_pred @ H.T @ inv(S)         # Kalman gain (6x2)
innov   = normalize([z - h_pred])        # angle-wrapped residual
x_post  = x_pred + K @ innov
P_post  = (I - K @ H) @ P_pred
```

**Constructor:**
```python
R = np.array([[azimuth_noise_std**2, 0],
              [0, elevation_noise_std**2]])

updater = EKFUpdater3D(
    noise_covariance=R,
    mapping=(0, 2, 4),
    ndim_state=6,
)
```

**Call:**
```python
x_post, P_post = updater.update(
    x_pred, P_pred,
    measurement=np.array([azimuth, elevation]),  # radians
    sensor_position=(sx, sy, sz),                # where the sensor is right now
    sensor_rotation=(pitch, yaw),                # sensor orientation, or omit
    kalman_gain_method="solve",                  # optional; default is "inv"
)
```

`sensor_position` and `sensor_rotation` are passed per-call because the ownship is maneuvering -- its pose changes every timestep. The caller is responsible for providing the current ownship position and orientation from its own navigation solution (INS/GPS). The filter treats these as known inputs, not estimated quantities.

## Minimal Integration Example

```python
import numpy as np
from ekf3d.ekf_predictor import EKFPredictor3D
from ekf3d.ekf_updater import EKFUpdater3D

# --- Configuration ---
dt = 1.0  # seconds between measurements
noise_diff_coeff = 0.01  # process noise (match to target dynamics)
azimuth_noise_std = np.deg2rad(0.5)    # sensor spec, radians
elevation_noise_std = np.deg2rad(0.5)  # sensor spec, radians

predictor = EKFPredictor3D(
    noise_diff_coeff_x=noise_diff_coeff,
    noise_diff_coeff_y=noise_diff_coeff,
    noise_diff_coeff_z=noise_diff_coeff,
)

R = np.array([[azimuth_noise_std**2, 0],
              [0, elevation_noise_std**2]])

updater = EKFUpdater3D(noise_covariance=R, mapping=(0, 2, 4), ndim_state=6)

# --- Initialization ---
# Your best guess of target [x, vx, y, vy, z, vz]
x = np.array([100.0, 0.0, 200.0, 0.0, 50.0, 0.0])
P = np.eye(6) * 100.0  # large initial uncertainty

# --- Loop ---
# Each iteration the ownship has maneuvered to a new position.
# sensor_pos comes from your navigation solution (INS/GPS).
# azimuth/elevation come from the sensor measurement.
for azimuth, elevation, sensor_pos in your_measurement_source:
    # Predict target state forward (skip on first iteration to use prior directly)
    x, P = predictor.predict(x, P, dt)

    # Update with measurement from the sensor's current position
    z = np.array([azimuth, elevation])
    x, P = updater.update(
        x,
        P,
        z,
        sensor_position=sensor_pos,
        kalman_gain_method="solve",  # optional; default is "inv"
    )

    # x now holds the best estimate of the TARGET state, not the ownship
```

## Files to Extract

To port the 3D EKF to another project, copy these two files:

```
src/ekf3d/ekf_predictor.py   # EKFPredictor3D + ConstantVelocityModel + CombinedCVModel3D
src/ekf3d/ekf_updater.py     # EKFUpdater3D + AzimuthElevationMeasurementModel + normalize_angles
```

Both files depend only on NumPy.

## Tuning Notes

**Process noise (`noise_diff_coeff`)**: This is the variance of the assumed random acceleration per axis. Set it to match the expected target maneuverability. Too high and the velocity estimate drifts (range error grows unbounded). Too low and the filter can't track maneuvers. A good starting point is the square of the expected acceleration standard deviation.

**Measurement noise (`noise_covariance`)**: Set to the sensor's stated accuracy. This is `R` in the EKF equations.

**Initial covariance (`P`)**: Set diagonal entries large enough to cover your uncertainty in the initial state. Typical values are 100-10000 depending on units and how rough your initial estimate is.

## Performance Testing

Performance tests are opt-in so normal unit test runs stay fast.

### Microbenchmarks (predict/update latency)

Run:

```bash
mkdir -p perf && uv run --group dev pytest --runperf tests/perf --benchmark-only \
  --benchmark-min-rounds=20 \
  --benchmark-json perf/microbench.json
```

This measures:
- `EKFPredictor3D.predict()` latency
- `EKFUpdater3D.update()` latency across:
  - nominal
  - near-singularity
  - stress covariance
  - with/without sensor pose transforms
  - `kalman_gain_method="inv"` and `"solve"`

### Realtime Frame-Time Benchmark (deadline analysis)

Run:

```bash
uv run --group dev python scripts/benchmark_realtime.py \
  --measurement-intervals 0.033333,0.02,0.01,0.005 \
  --steps 20000 \
  --scenario nominal \
  --kalman-gain-method inv \
  --with-sensor-pose \
  --json-out perf/realtime_nominal.json
```

The script reports frame-time metrics versus measurement interval (`dt`), including:
- `miss_%`: percentage of frames that exceed the measurement deadline
- `worst_overrun_ms`: maximum deadline miss
- `mean_slack_ms` / `p05_slack_ms`: margin before deadline
- `util_p95_%`: p95 compute time as a percentage of interval budget
- `est_headroom_hz`: approximate sustainable update rate based on p95 frame time
