# MPCC Track Configuration Guide

## Two Types of Circle Tracks

MPCC supports two different ways to generate a circle track:

### 1. **MPCC Built-in Circle** (`mpcc_utils.create_circle()`)
- **Source**: `safe_control_gym/controllers/mpcc/mpcc_utils.py`
- **Configuration**: `track_name: 'circle'` in MPCC config
- **Parameters**: `track_scale` (radius)
- **When to use**: When you want MPCC's own path-based circle generator

### 2. **Environment Circle** (`env.X_GOAL` from `benchmark_env.py`)
- **Source**: `safe_control_gym/envs/benchmark_env.py` → `_circle()` method
- **Configuration**: `task_info.trajectory_type: 'circle'` in environment config
- **Parameters**: `trajectory_scale`, `num_cycles`, `trajectory_plane`, `trajectory_position_offset`
- **When to use**: When you want to use the same trajectory as MPC/LQR/PID controllers

---

## Configuration Examples

### Option A: Use MPCC Built-in Circle (Current Default)

**File**: `mpcc_quadrotor_2D_tracking.yaml`
```yaml
algo_config:
  track_name: 'circle'  # Uses mpcc_utils.create_circle()
  track_scale: 5.0       # Circle radius
  # use_env_trajectory: false  # Default
```

**File**: `quadrotor_2D_tracking.yaml`
```yaml
task_config:
  task: traj_tracking
  task_info: null  # Ignored when using built-in track
```

**Result**: MPCC generates its own circle using `create_circle(radius=5.0)`

---

### Option B: Use Environment Trajectory (env.X_GOAL)

**File**: `mpcc_quadrotor_2D_tracking_env_circle.yaml`
```yaml
algo_config:
  use_env_trajectory: true  # Enable env.X_GOAL conversion
  env_trajectory_sample_rate: 10  # Waypoints per second
  # track_name is ignored when use_env_trajectory=true
```

**File**: `quadrotor_2D_tracking_env_circle.yaml`
```yaml
task_config:
  task: traj_tracking
  task_info:
    trajectory_type: circle  # Environment's circle generator
    num_cycles: 2.5
    trajectory_plane: 'xy'
    trajectory_position_offset: [0, 1.0]
    trajectory_scale: 0.9
```

**Result**: Environment generates `env.X_GOAL` using `benchmark_env._circle()`, then MPCC converts it to waypoints

---

## How to Run

### Use Built-in Circle (Current):
```bash
python3 ./mpcc_experiment.py \
    --task quadrotor \
    --algo mpcc \
    --overrides \
        ./config_overrides/quadrotor_2D/quadrotor_2D_tracking.yaml \
        ./config_overrides/quadrotor_2D/mpcc_quadrotor_2D_tracking.yaml
```

### Use Environment Circle:
```bash
python3 ./mpcc_experiment.py \
    --task quadrotor \
    --algo mpcc \
    --overrides \
        ./config_overrides/quadrotor_2D/quadrotor_2D_tracking_env_circle.yaml \
        ./config_overrides/quadrotor_2D/mpcc_quadrotor_2D_tracking_env_circle.yaml
```

---

## Key Differences

| Feature | MPCC Built-in | Environment Trajectory |
|---------|---------------|------------------------|
| **Source** | `mpcc_utils.create_circle()` | `benchmark_env._circle()` |
| **Parameter** | `track_scale` (radius) | `trajectory_scale` (radius) |
| **Config Location** | `algo_config` | `task_config.task_info` |
| **Time-based?** | No (path-based) | Yes (converted to path) |
| **Compatible with MPC?** | No | Yes (same trajectory) |
| **Speed Control** | Yes (v_theta) | Yes (v_theta) |

---

## Priority Order

When MPCC initializes, it checks in this order:

1. **track_file** (if specified) → Load from CSV file
2. **waypoints** (if provided) → Use provided waypoints array
3. **env.X_GOAL** (if `use_env_trajectory=true`) → Convert environment trajectory
4. **track_name** (default) → Use built-in generator (`create_circle()`, etc.)

---

## Debugging

Enable verbose mode to see which track source is used:
```yaml
algo_config:
  verbose: true
```

You'll see messages like:
- `[MPCC] Converting env.X_GOAL to waypoints` → Using environment trajectory
- `[MPCC] Generating built-in track: circle` → Using MPCC built-in

