"""Utility functions for 2D MPCC (Model Predictive Contouring Control).

This module provides:
- Track generation from waypoints (Bezier interpolation)
- Track loading and management
- State conversion utilities
- Control conversion utilities

Based on: [https://github.com/mlab-upenn/mpcc](https://github.com/mlab-upenn/mpcc)
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

# Optional matplotlib for debugging
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════
# Track Generation (Bezier Interpolation)
# ═══════════════════════════════════════════════════════════

def interpolate_bezier(waypoints):
    """Interpolate waypoints with cubic Bezier curves (cyclic boundary condition).
    
    Creates a smooth closed curve through all waypoints using cubic Bezier splines
    with C2 continuity (continuous curvature) at all waypoints.
    
    Args:
        waypoints (ndarray): Nx2 array of waypoints [x, y]
            For closed curves, waypoints should NOT duplicate the first point at the end.
            The cyclic boundary condition handles the closure automatically.
        
    Returns:
        tuple: (a, b) control points
            a (ndarray): Control points A (2xN)
            b (ndarray): Control points B (2xN)
    """
    n = len(waypoints)
    M = np.zeros([n, n])
    
    # Build M matrix (cyclic tridiagonal)
    # For cyclic cubic spline with C2 continuity, we need:
    #
    # Row 0:   [4, 1, 0, 0, ..., 0, 0, 1]  <- wraps: M[0, n-1] = 1
    # Row 1:   [1, 4, 1, 0, ..., 0, 0, 0]
    # Row 2:   [0, 1, 4, 1, ..., 0, 0, 0]
    # ...
    # Row n-2: [0, 0, ..., 0, 1, 4, 1, 0]
    # Row n-1: [1, 0, 0, ..., 0, 0, 1, 4]  <- wraps: M[n-1, 0] = 1
    
    # Fill middle rows (1 to n-2) with [1, 4, 1] pattern
    for idx in range(1, n - 1):
        M[idx, idx - 1] = 1
        M[idx, idx] = 4
        M[idx, idx + 1] = 1
    
    # First row: [4, 1, 0, ..., 0, 1]
    M[0, 0] = 4
    M[0, 1] = 1
    M[0, n - 1] = 1  # Cyclic wrap-around
    
    # Last row: [1, 0, ..., 0, 1, 4]
    M[n - 1, 0] = 1  # Cyclic wrap-around
    M[n - 1, n - 2] = 1
    M[n - 1, n - 1] = 4
    
    # Build solution vector
    # s[i] = 2 * (2 * P[i] + P[i+1]) for continuity conditions
    s = np.zeros([n, 2])
    for idx in range(n - 1):
        s[idx, :] = 2 * (2 * waypoints[idx, :] + waypoints[idx + 1, :])
    # Last entry wraps around: s[n-1] uses P[n-1] and P[0]
    s[n - 1, :] = 2 * (2 * waypoints[n - 1, :] + waypoints[0, :])
    
    # Solve for control points A
    Ax = np.linalg.solve(M, s[:, 0])
    Ay = np.linalg.solve(M, s[:, 1])
    
    a = np.vstack([Ax, Ay])
    
    # Compute control points B from A
    # B[i] = 2 * P[i+1] - A[i+1]
    b = np.zeros([2, n])
    for idx in range(n - 1):
        b[:, idx] = 2 * waypoints[idx + 1, :] - a[:, idx + 1]
    # Last B wraps around: B[n-1] = 2 * P[0] - A[0]
    b[:, n - 1] = 2 * waypoints[0, :] - a[:, 0]
    
    return a, b


def eval_bezier(waypoints, a, b, t):
    """Evaluate Bezier curve at parameter t.
    
    Matches benchmarked implementation from Bezier.py (eval_raw).
    
    Args:
        waypoints (ndarray): Waypoints
        a (ndarray): Bezier control points A
        b (ndarray): Bezier control points B
        t (float): Parameter (can be > n for multiple laps)
        
    Returns:
        ndarray: [x, y] coordinates at parameter t
    """
    n = len(waypoints)
    t = np.mod(t, n)
    segment = np.floor(t)
    segment = int(segment)
    
    if segment >= n:
        t = n - 0.0001
        segment = n - 1
    elif t < 0:
        t = 0
    
    t_val = t - segment
    
    # Cubic Bezier formula (matches benchmarked version exactly)
    coords = np.power(1 - t_val, 3) * waypoints.T[:, segment] + 3 * np.power(1 - t_val, 2) * t_val * a[:, segment]\
    + 3 * (1 - t_val) * np.power(t_val, 2) * b[:, segment] + np.power(t_val, 3) * waypoints.T[:, int(np.mod(segment+1,n))]
    
    return coords


def get_angle_bezier(waypoints, a, b, t):
    """Get tangent angle at parameter t.
    
    Matches benchmarked implementation from Bezier.py (getangle_raw).
    
    Args:
        waypoints (ndarray): Waypoints
        a (ndarray): Bezier control points A
        b (ndarray): Bezier control points B
        t (float): Parameter
        
    Returns:
        float: Tangent angle (yaw) in radians
    """
    der = eval_bezier(waypoints, a, b, t + 0.1) - eval_bezier(waypoints, a, b, t)
    phi = np.arctan2(der[1], der[0])
    return phi


def fit_arc_length_to_parameter(waypoints, a, b):
    """Fit arc length s to parameter t using cubic spline.
    
    Matches benchmarked implementation from Bezier.py (fit_st function).
    Uses two revolutions to account for horizon overshooting end of lap.
    
    Args:
        waypoints (ndarray): Waypoints
        a (ndarray): Bezier control points A
        b (ndarray): Bezier control points B
        
    Returns:
        tuple: (ts_inverse, smax)
            ts_inverse (CubicSpline): Spline function s -> t
            smax (float): Maximum arc length (track length)
    """
    # Using two revolutions to account for horizon overshooting end of lap
    nwp = len(waypoints)
    npoints = 20 * nwp
    
    # Compute approximate max distance
    tvals = np.linspace(0, nwp, npoints + 1)
    coords = []
    for t in tvals:
        coords.append(eval_bezier(waypoints, a, b, t))
    coords = np.array(coords)
    
    dists = []
    dists.append(0)
    for idx in range(npoints):
        dists.append(np.sqrt(np.sum(np.square(coords[idx, :] - coords[np.mod(idx + 1, npoints - 1), :]))))
    dists = np.cumsum(np.array(dists))
    smax = dists[-1]
    
    # Fit the s-t relationship for two track revolutions
    npoints = 2 * 20 * nwp
    
    # Compute approx distance to arc param
    tvals = np.linspace(0, 2 * nwp, npoints + 1)
    
    coords = []
    for t in tvals:
        coords.append(eval_bezier(waypoints, a, b, np.mod(t, nwp)))
    coords = np.array(coords)
    
    distsr = []
    distsr.append(0)
    for idx in range(npoints):
        distsr.append(np.sqrt(np.sum(np.square(coords[idx, :] - coords[np.mod(idx + 1, npoints - 1), :]))))
    dists = np.cumsum(np.array(distsr))
    
    # Create inverse mapping: s -> t
    ts_inverse = CubicSpline(dists, tvals)
    
    return ts_inverse, smax


def generate_lookup_table_from_waypoints(waypoints, track_width=0.5, density=100):
    """Generate MPCC lookup table from waypoints.
    
    Args:
        waypoints (ndarray): Nx2 array of waypoints [x, y]
        track_width (float): Half-width of track (m)
        density (int): Points per meter in lookup table
        
    Returns:
        tuple: (table, smax)
            table (ndarray): Generated lookup table (Mx9)
            smax (float): Track length (m)
    """
    print(f"[Track Generation] Processing {len(waypoints)} waypoints")
    
    # Bezier interpolation
    a, b = interpolate_bezier(waypoints)
    
    # Fit arc length to parameter
    ts_inverse, smax = fit_arc_length_to_parameter(waypoints, a, b)
    
    # Generate lookup table
    lutable_density = density
    npoints = int(np.floor(2 * smax * lutable_density))
    
    print(f"[Track Generation] Generating lookup table with {npoints} points")
    
    svals = np.linspace(0, 2 * smax, npoints)
    tvals = ts_inverse(svals)
    
    # Compute table entries
    # Format: [s, t, x, y, phi, cos_phi, sin_phi, g_upper, g_lower]
    table = []
    for idx in range(npoints):
        # Track point
        track_point = eval_bezier(waypoints, a, b, tvals[idx])
        
        # Tangent angle
        phi = get_angle_bezier(waypoints, a, b, tvals[idx])
        
        # Normal vector
        n = [-np.sin(phi), np.cos(phi)]
        
        # Track bounds (for constraints)
        g_upper = track_width + track_point[0] * n[0] + track_point[1] * n[1]
        g_lower = -track_width + track_point[0] * n[0] + track_point[1] * n[1]
        
        table.append([
            svals[idx],      # Arc length
            tvals[idx],      # Parameter
            track_point[0],  # x
            track_point[1],  # y
            phi,             # Tangent angle
            np.cos(phi),     # cos(phi)
            np.sin(phi),     # sin(phi)
            g_upper,         # Upper bound
            g_lower          # Lower bound
        ])
    
    table = np.array(table)
    
    print(f"[Track Generation] Track length: {smax:.2f} m")
    
    return table, smax


# ═══════════════════════════════════════════════════════════
# Environment Trajectory Conversion
# ═══════════════════════════════════════════════════════════

def env_trajectory_to_waypoints(env, sample_rate=2):
    """Convert env.X_GOAL (time-based trajectory) to waypoints for MPCC.
    
    This function extracts x, y positions from the environment's time-based
    trajectory and samples them at a lower rate to create waypoints suitable
    for MPCC's path-based tracking system.
    
    Works for all trajectory types (circles, ovals, figure-eights, open paths, etc.).
    For closed trajectories, automatically ensures proper closure for Bezier interpolation.
    
    IMPORTANT: For 2D quadrotor with trajectory_plane='xy', we need to regenerate
    the trajectory because X_GOAL only stores x and z, but the actual trajectory
    is in x-y plane.
    
    Args:
        env: Environment with X_GOAL attribute (time-based trajectory)
        sample_rate (float): Number of waypoints per second (default: 2)
            Lower values (1-3) produce fewer, sparser waypoints suitable for Bezier interpolation.
            Higher values (5-10) produce denser waypoints.
    
    Returns:
        waypoints (ndarray): Nx2 array of [x, y] waypoints for MPCC
            For closed trajectories, the last waypoint matches the first for proper closure.
    
    Raises:
        ValueError: If env.X_GOAL is not available or has unexpected structure
    """
    if not hasattr(env, 'X_GOAL'):
        raise ValueError("Environment does not have X_GOAL attribute")
    
    # Check if we need to regenerate trajectory (for 2D quadrotor with xy plane)
    regenerate_needed = False
    if hasattr(env, 'QUAD_TYPE') and hasattr(env, 'TASK_INFO'):
        from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
        from safe_control_gym.envs.benchmark_env import Task
        
        if (env.QUAD_TYPE == QuadType.TWO_D and 
            env.TASK == Task.TRAJ_TRACKING and 
            env.TASK_INFO.get('trajectory_plane') == 'xy'):
            # For 2D quadrotor with xy plane, X_GOAL only has x and z
            # but the circle is in x-y, so z is constant. We need to regenerate.
            regenerate_needed = True
    
    if regenerate_needed:
        # Regenerate trajectory to get correct x, y coordinates
        if hasattr(env, '_generate_trajectory'):
            POS_REF, VEL_REF, _ = env._generate_trajectory(
                traj_type=env.TASK_INFO['trajectory_type'],
                traj_length=env.EPISODE_LEN_SEC,
                num_cycles=env.TASK_INFO['num_cycles'],
                traj_plane=env.TASK_INFO['trajectory_plane'],
                position_offset=np.array(env.TASK_INFO['trajectory_position_offset']),
                scaling=env.TASK_INFO['trajectory_scale'],
                sample_time=env.CTRL_TIMESTEP
            )
            # Extract x and y from POS_REF (which has full 3D info)
            x_positions = POS_REF[:, 0]  # x
            y_positions = POS_REF[:, 1]  # y (not z!)
        else:
            raise ValueError("Cannot regenerate trajectory: env._generate_trajectory not available")
    
    else:
        # Use X_GOAL directly (normal case)
        traj = env.X_GOAL  # Shape: (num_timesteps, state_dim)
        
        if traj.ndim != 2:
            raise ValueError(f"X_GOAL must be 2D array, got shape {traj.shape}")
        
        # Extract x, y positions based on environment type
        if hasattr(env, 'STATE_LABELS'):
            # Use state labels to find correct indices (most robust)
            labels = env.STATE_LABELS
            try:
                x_idx = labels.index('x')
                # For 2D quadrotor, 'y' in MPCC corresponds to 'z' in environment
                # For 3D quadrotor, use 'y' directly
                if 'y' in labels:
                    y_idx = labels.index('y')
                elif 'z' in labels:
                    y_idx = labels.index('z')  # 2D quadrotor: z becomes y in MPCC
                else:
                    raise ValueError("Cannot find y or z in STATE_LABELS")
                
                x_positions = traj[:, x_idx]
                y_positions = traj[:, y_idx]
                
            except ValueError as e:
                raise ValueError(f"Cannot extract x, y from STATE_LABELS: {e}")
        
        elif hasattr(env, 'QUAD_TYPE'):
            # Fallback: Use quadrotor-specific structure
            from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
            
            if env.QUAD_TYPE == QuadType.TWO_D:
                # X_GOAL structure: [x, x_dot, z, z_dot, 0, 0]
                # For xz plane: x and z vary
                # For xy plane: x varies, z is constant (need to regenerate - handled above)
                x_positions = traj[:, 0]  # x
                y_positions = traj[:, 2]  # z (becomes y in MPCC)
            elif env.QUAD_TYPE == QuadType.THREE_D:
                # X_GOAL structure: [x, x_dot, y, y_dot, z, z_dot, ...]
                x_positions = traj[:, 0]  # x
                y_positions = traj[:, 2]  # y
            else:
                raise ValueError(f"Unsupported QUAD_TYPE: {env.QUAD_TYPE}")
        
        else:
            # Last resort: Assume first two columns are x, y
            if traj.shape[1] < 2:
                raise ValueError(f"X_GOAL must have at least 2 columns, got {traj.shape[1]}")
            x_positions = traj[:, 0]
            y_positions = traj[:, 1]
    
    # Sample at lower rate to get waypoints
    # General sampling approach that works for all trajectory types
    num_timesteps = len(x_positions)
    ctrl_freq = getattr(env, 'CTRL_FREQ', 50)
    
    # Check if trajectory appears to be closed (for proper closure in Bezier interpolation)
    first_point = np.array([x_positions[0], y_positions[0]])
    last_point = np.array([x_positions[-1], y_positions[-1]])
    distance_first_last = np.linalg.norm(first_point - last_point)
    is_closed = distance_first_last < 0.05  # Tighter threshold for closed loop
    
    # For periodic trajectories (like circles with multiple cycles), we need to extract
    # exactly one cycle. Check if trajectory is periodic by finding where first point repeats
    cycle_end_idx = None
    if not is_closed and num_timesteps > 10:
        # Estimate single cycle length based on num_cycles if available
        num_cycles = getattr(env, 'TASK_INFO', {}).get('num_cycles', 1)
        estimated_cycle_length = int(num_timesteps / num_cycles) if num_cycles > 1 else num_timesteps
        
        # Search around the estimated cycle end for where first point repeats
        # Use tighter threshold for matching
        search_start = max(10, int(estimated_cycle_length * 0.8))
        search_end = min(num_timesteps, int(estimated_cycle_length * 1.2))
        
        min_dist = float('inf')
        best_idx = None
        for i in range(search_start, search_end):
            current_point = np.array([x_positions[i], y_positions[i]])
            dist_to_start = np.linalg.norm(current_point - first_point)
            if dist_to_start < min_dist:
                min_dist = dist_to_start
                best_idx = i
        
        # Only accept if we found a close enough match
        if best_idx is not None and min_dist < 0.05:
            # DON'T include the endpoint - it's a near-duplicate of start
            # The cyclic Bezier will handle the wrap-around naturally
            cycle_end_idx = best_idx
            print(f"[env_trajectory_to_waypoints] Found cycle end at idx {best_idx}, "
                  f"dist to start: {min_dist:.6f}m")
    
    # If we found a cycle, extract only one cycle for waypoint generation
    if cycle_end_idx is not None:
        # Extract points from 0 to cycle_end_idx-1 (NOT including the near-duplicate endpoint)
        x_positions = x_positions[:cycle_end_idx]
        y_positions = y_positions[:cycle_end_idx]
        num_timesteps = len(x_positions)
        is_closed = True  # Mark as closed for cyclic Bezier
        print(f"[env_trajectory_to_waypoints] Extracted one cycle: {num_timesteps} points")
    
    # Calculate how many waypoints to generate
    # For Bezier interpolation, 20-30 waypoints is typically ideal
    episode_length_sec = num_timesteps / ctrl_freq
    num_waypoints = max(4, min(30, int(episode_length_sec * sample_rate)))
    
    # For closed trajectories, sample uniformly around the loop
    # We want N waypoints at angles: 0, 360/N, 2*360/N, ..., (N-1)*360/N
    # So we sample indices: 0, L/N, 2L/N, ..., (N-1)*L/N where L = num_timesteps
    if is_closed:
        # Sample N points uniformly, NOT including the endpoint (which equals start)
        indices = np.linspace(0, num_timesteps, num_waypoints, endpoint=False, dtype=int)
        indices = np.clip(indices, 0, num_timesteps - 1)
    else:
        # Open trajectory: include both endpoints
        indices = np.linspace(0, num_timesteps - 1, num_waypoints, dtype=int)
    
    waypoints = np.column_stack([
        x_positions[indices],
        y_positions[indices]
    ])
    
    # Ensure we have at least 4 waypoints
    if len(waypoints) < 4:
        if is_closed:
            indices = np.linspace(0, num_timesteps, 4, endpoint=False, dtype=int)
        else:
            indices = np.linspace(0, num_timesteps - 1, 4, dtype=int)
        indices = np.clip(indices, 0, num_timesteps - 1)
        waypoints = np.column_stack([
            x_positions[indices],
            y_positions[indices]
        ])
    
    # Final check: for closed trajectories, ensure first and last waypoints are NOT duplicates
    # (they should be different points on the circle, cyclic Bezier handles the connection)
    if is_closed:
        waypoints_distance = np.linalg.norm(waypoints[0] - waypoints[-1])
        if waypoints_distance < 0.01:
            # Too close - remove the last waypoint
            waypoints = waypoints[:-1]
            print(f"[env_trajectory_to_waypoints] Removed near-duplicate endpoint, "
                  f"now {len(waypoints)} waypoints")
    
    print(f"[env_trajectory_to_waypoints] Generated {len(waypoints)} waypoints "
          f"(closed={is_closed})")
    
    return waypoints


# ═══════════════════════════════════════════════════════════
# Example Track Generators
# ═══════════════════════════════════════════════════════════

def create_simple_oval(scale=10):
    """Create simple oval track waypoints.
    
    Args:
        scale (float): Scaling factor
        
    Returns:
        ndarray: Nx2 array of waypoints
    """
    trackx = scale * np.array([
        0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
        0.8, 0.8, 0.8, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 
        0.3, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0
    ])
    tracky = scale * np.array([
        0.05, 0.3, 0.4, 0.2, 0.2, 0.0, 0.0, 0.0, 0.05,
        0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.5, 0.3, 0.3,
        0.5, 0.5, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1
    ])
    waypoints = np.vstack([trackx, tracky]).T
    return waypoints


def create_figure_eight(radius=5):
    """Create figure-8 track waypoints.
    
    Args:
        radius (float): Radius of loops
        
    Returns:
        ndarray: Nx2 array of waypoints
    """
    n = 20
    theta = np.linspace(0, 2 * np.pi, n)
    
    # Figure-8 parametric
    x = radius * np.sin(theta)
    y = radius * np.sin(theta) * np.cos(theta)
    
    waypoints = np.vstack([x, y]).T
    return waypoints


def create_circle(radius=5, n_points=20):
    """Create circular track waypoints.
    
    For cyclic Bezier interpolation, the last waypoint must match the first.
    So we sample (n_points-1) unique points, then add the first point again
    to get exactly n_points waypoints where the last matches the first.
    
    Args:
        radius (float): Circle radius
        n_points (int): Number of waypoints (will be n_points total with last matching first)
        
    Returns:
        ndarray: Nx2 array of waypoints where last matches first (N = n_points)
    """
    # Sample (n_points-1) unique points around the circle (excludes endpoint for closure)
    theta = np.linspace(0, 2 * np.pi, n_points - 1, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    waypoints_unique = np.vstack([x, y]).T
    
    # For cyclic Bezier interpolation, last waypoint must match first
    # Add first point again at the end to ensure proper closure
    # This gives exactly n_points waypoints where last exactly matches first
    waypoints = np.vstack([waypoints_unique, waypoints_unique[0:1]])
    
    return waypoints


# ═══════════════════════════════════════════════════════════
# Track Management
# ═══════════════════════════════════════════════════════════

class TrackManager:
    """Manages track lookup table for MPCC.
    
    Can either:
    1. Load pre-generated lookup table from CSV
    2. Generate lookup table from waypoints on-the-fly
    """
    
    def __init__(self, track_file=None, waypoints=None, track_width=0.5, 
                 density=100, closed=True, cache_dir='tracks'):
        """Initialize track manager.
        
        Args:
            track_file (str): Path to lookup table CSV (*_lutab.csv) OR waypoints CSV
            waypoints (ndarray): Nx2 array of waypoints (alternative to track_file)
            track_width (float): Half-width of track (m)
            density (int): Points per meter in lookup table
            closed (bool): Whether track is closed loop
            cache_dir (str): Directory to cache generated lookup tables
        """
        self.closed = closed
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        smax = None  # Initialize smax (one cycle length from generate_lookup_table_from_waypoints)
        
        # Determine how to load/generate track
        if track_file is not None:
            # Check if it's a lookup table or waypoints
            if track_file.endswith('_lutab.csv'):
                # Load existing lookup table
                print(f"[TrackManager] Loading lookup table: {track_file}")
                self.track_table = np.loadtxt(track_file, delimiter=',')
            else:
                # Load waypoints and generate lookup table
                print(f"[TrackManager] Loading waypoints: {track_file}")
                waypoints_loaded = np.loadtxt(track_file, delimiter=',')
                
                # Check if cached lookup table exists
                cache_file = self._get_cache_filename(track_file)
                if os.path.exists(cache_file):
                    print(f"[TrackManager] Using cached lookup table: {cache_file}")
                    self.track_table = np.loadtxt(cache_file, delimiter=',')
                else:
                    print(f"[TrackManager] Generating lookup table...")
                    self.track_table, smax = generate_lookup_table_from_waypoints(
                        waypoints_loaded, track_width, density
                    )
                    # Save to cache
                    np.savetxt(cache_file, self.track_table, delimiter=',')
                    print(f"[TrackManager] Cached to: {cache_file}")
        
        elif waypoints is not None:
            # Generate from provided waypoints
            print(f"[TrackManager] Generating lookup table from waypoints...")
            # Store original waypoints for debugging/visualization
            self.waypoints = waypoints.copy()
            self.track_table, smax = generate_lookup_table_from_waypoints(
                waypoints, track_width, density
            )
        
        else:
            raise ValueError("Must provide either track_file or waypoints")
        
        # Ensure 2D array
        if self.track_table.ndim == 1:
            self.track_table = self.track_table.reshape(1, -1)
        
        # Extract track properties
        self.n_points = len(self.track_table)
        # s_max should be the length of ONE cycle (not the full track_table length)
        # track_table is generated with 2 * smax to account for horizon overshooting,
        # but s_max (cycle length) should be half of that
        # Use smax if available (from generate_lookup_table_from_waypoints), otherwise estimate from table
        if 'smax' in locals():
            self.s_max = smax  # Use the returned smax (one cycle length)
        else:
            # For cached tables, estimate: track_table covers 2 cycles, so divide by 2
            full_table_length = self.track_table[-1, 0]
            self.s_max = full_table_length / 2.0  # One cycle length
        self.reference_path = self.track_table[:, 2:4]
        
        # Ensure waypoints are stored (if not already stored from waypoints parameter)
        if not hasattr(self, 'waypoints') or self.waypoints is None:
            # Extract waypoints from track_file if available
            if track_file is not None and not track_file.endswith('_lutab.csv'):
                # If it was a waypoints file, we should have already loaded it
                pass
            self.waypoints = None  # Waypoints not available
        
        print(f"[TrackManager] Track ready: {self.n_points} points, length={self.s_max:.2f}m")
    
    def _get_cache_filename(self, waypoints_file):
        """Get cache filename for waypoints file.
        
        Args:
            waypoints_file (str): Path to waypoints file
            
        Returns:
            str: Cache file path
        """
        base = os.path.splitext(os.path.basename(waypoints_file))[0]
        return os.path.join(self.cache_dir, f'{base}_lutab.csv')
    
    def get_track_info(self, theta):
        """Get track information at progress variable theta.
        
        Args:
            theta (float): Progress along path (arc length, same as 's')
            
        Returns:
            dict: Track information at theta with keys:
                - 's': arc length
                - 't': parameter
                - 'x', 'y': position
                - 'phi': tangent angle
                - 'cos_phi', 'sin_phi': angle components
                - 'g_upper', 'g_lower': track bounds
        """
        # Handle closed/open track
        if self.closed:
            theta = theta % self.s_max
        else:
            theta = np.clip(theta, 0, self.s_max)
        
        # Find nearest point in lookup table
        idx = np.argmin(np.abs(self.track_table[:, 0] - theta))
        info = self.track_table[idx]
        
        return {
            's': info[0],
            't': info[1],
            'x': info[2],
            'y': info[3],
            'phi': info[4],
            'cos_phi': info[5],
            'sin_phi': info[6],
            'g_upper': info[7] if len(info) > 7 else 0.5,
            'g_lower': info[8] if len(info) > 8 else -0.5,
        }
    
    def get_track_params(self, theta):
        """Get track parameters for ACADOS solver.
        
        Args:
            theta (float): Progress variable
            
        Returns:
            ndarray: [x_ref, y_ref, cos_phi, sin_phi] (4D)
        """
        info = self.get_track_info(theta)
        return np.array([
            info['x'],
            info['y'],
            info['cos_phi'],
            info['sin_phi']
        ])
    
    def compute_errors(self, pos_xy, theta): #for post analysis
        """Compute MPCC contouring and lag errors.
        
        Args:
            pos_xy (ndarray): Current 2D position [x, y]
            theta (float): Current progress variable
            
        Returns:
            tuple: (e_c, e_l)
                e_c (float): Contouring error (perpendicular, m)
                e_l (float): Lag error (tangential, m)
        """
        info = self.get_track_info(theta)
        
        dx = pos_xy[0] - info['x']
        dy = pos_xy[1] - info['y']
        
        # Contouring error (perpendicular to path)
        e_c = -dx * info['sin_phi'] + dy * info['cos_phi']
        
        # Lag error (tangential to path)
        e_l = dx * info['cos_phi'] + dy * info['sin_phi']
        
        return e_c, e_l


# ═══════════════════════════════════════════════════════════
# State/Control Conversion
# ═══════════════════════════════════════════════════════════

def obs_to_mpcc_state(obs, theta, env=None): #NOTE: regacy code: not used anymore
    """Convert environment observation to MPCC state (2D path following).
    
    MPCC State Vector (7D):
        [x, y, vx, vy, yaw, yaw_rate, theta]
        - x, y: position in MPCC frame (2D path plane)
        - vx, vy: velocities in MPCC frame
        - yaw, yaw_rate: heading and heading rate
        - theta: path progress variable (NOT drone pitch!)
    
    Frame Mapping:
        - 2D Quadrotor (xz-plane): drone x → MPCC x, drone z → MPCC y
        - 3D Quadrotor (xy-plane): drone x → MPCC x, drone y → MPCC y
    
    Supported observation formats:
        - 6D (2D Quadrotor): [x, x_dot, z, z_dot, pitch, pitch_dot]
        - 12D (3D Quadrotor): [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    
    Args:
        obs (ndarray): Environment observation
        theta (float): Current path progress variable (arc length along path)
        env: Environment (optional, for robust state indexing via STATE_LABELS)
        
    Returns:
        ndarray: MPCC state [x, y, vx, vy, yaw, yaw_rate, theta] (7D)
    """
    obs = np.atleast_1d(obs)
    
    # Option 1: Use environment's state labels (most robust)
    if env is not None and hasattr(env, 'STATE_LABELS'):
        labels = env.STATE_LABELS
        try:
            x = obs[labels.index('x')]
            
            # For 2D quadrotor: z becomes y in MPCC frame
            if 'y' in labels:
                y = obs[labels.index('y')]
            elif 'z' in labels:
                y = obs[labels.index('z')]  # 2D quadrotor: z → y
            else:
                raise ValueError("No 'y' or 'z' in STATE_LABELS")
            
            vx = obs[labels.index('x_dot')]
            
            # Velocity: y_dot or z_dot
            if 'y_dot' in labels:
                vy = obs[labels.index('y_dot')]
            elif 'z_dot' in labels:
                vy = obs[labels.index('z_dot')]  # 2D quadrotor: z_dot → vy
            else:
                raise ValueError("No 'y_dot' or 'z_dot' in STATE_LABELS")
            
            # Yaw: psi or theta (for 2D quadrotor, theta is pitch angle)
            if 'psi' in labels:
                yaw = obs[labels.index('psi')]
            elif 'theta' in labels:
                # For 2D quadrotor, theta is pitch which affects horizontal movement
                yaw = obs[labels.index('theta')]
            else:
                yaw = 0.0
            
            # Yaw rate might have different names
            if 'psi_dot' in labels:
                yaw_rate = obs[labels.index('psi_dot')]
            elif 'theta_dot' in labels:
                yaw_rate = obs[labels.index('theta_dot')]  # 2D quadrotor
            elif 'p' in labels:
                yaw_rate = obs[labels.index('p')]
            elif 'r' in labels:
                yaw_rate = obs[labels.index('r')]
            else:
                yaw_rate = 0.0
                
        except (ValueError, IndexError) as e:
            raise ValueError(f"Environment STATE_LABELS missing required fields: {e}")
    
    # Option 2: Auto-detect based on observation dimension
    elif len(obs) == 6:
        # ═══════════════════════════════════════════════════════════
        # 6D Observation: 2D Quadrotor in xz-plane
        # obs = [x, x_dot, z, z_dot, pitch, pitch_dot]
        #        0    1    2    3      4       5
        # 
        # MPCC Frame Mapping:
        #   drone x → MPCC x (horizontal forward)
        #   drone z → MPCC y (vertical, becomes MPCC's "lateral" direction)
        #   drone pitch → used as MPCC yaw (controls heading in xz plane)
        # ═══════════════════════════════════════════════════════════
        if env is not None and hasattr(env, 'QUAD_TYPE'):
            from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
            if env.QUAD_TYPE == QuadType.TWO_D:
                # 2D Quadrotor: [x, x_dot, z, z_dot, pitch, pitch_dot]
                x = obs[0]         # drone x → MPCC x
                y = obs[2]         # drone z → MPCC y
                vx = obs[1]        # drone x_dot → MPCC vx
                vy = obs[3]        # drone z_dot → MPCC vy
                yaw = obs[4]       # drone pitch → MPCC yaw (heading in xz plane)
                yaw_rate = obs[5]  # drone pitch_dot → MPCC yaw_rate
            else:
                # Assume generic planar: [x, y, vx, vy, yaw, yaw_rate]
                x = obs[0]
                y = obs[1]
                vx = obs[2]
                vy = obs[3]
                yaw = obs[4]
                yaw_rate = obs[5]
        else:
            # Default: assume 2D Quadrotor format [x, x_dot, z, z_dot, pitch, pitch_dot]
            # This is the most common 6D format in this codebase
            x = obs[0]         # drone x → MPCC x
            y = obs[2]         # drone z → MPCC y
            vx = obs[1]        # drone x_dot → MPCC vx
            vy = obs[3]        # drone z_dot → MPCC vy
            yaw = obs[4]       # drone pitch → MPCC yaw
            yaw_rate = obs[5]  # drone pitch_dot → MPCC yaw_rate
        
    elif len(obs) == 12:
        # ═══════════════════════════════════════════════════════════
        # 12D Observation: 3D Quadrotor
        # obs = [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        #        0  1  2   3   4   5    6     7     8    9  10  11
        # 
        # MPCC Frame Mapping:
        #   drone x → MPCC x (forward in world frame)
        #   drone y → MPCC y (lateral in world frame)
        #   drone z is controlled separately (fixed altitude)
        # ═══════════════════════════════════════════════════════════
        x = obs[0]         # drone x → MPCC x
        y = obs[1]         # drone y → MPCC y
        vx = obs[3]        # drone vx → MPCC vx
        vy = obs[4]        # drone vy → MPCC vy
        yaw = obs[8]       # drone yaw → MPCC yaw
        yaw_rate = obs[11] # drone wz → MPCC yaw_rate
        
    elif len(obs) == 10:
        # Another common format
        # [x, y, z, vx, vy, vz, yaw, pitch, roll, yaw_rate]
        x = obs[0]
        y = obs[1]
        vx = obs[3]
        vy = obs[4]
        yaw = obs[6]
        yaw_rate = obs[9]
        
    else:
        raise ValueError(
            f"Unsupported observation dimension: {len(obs)}. "
            f"Expected 6, 10, or 12. "
            f"If using custom environment, ensure env.STATE_LABELS is defined."
        )
    
    state = np.array([x, y, vx, vy, yaw, yaw_rate, theta])
    return state


def mpcc_control_to_action(u_mpcc, obs, env, fixed_altitude=1.0): #NOTE: regacy code: not used anymore
    """Convert MPCC control to environment action.
    
    MPCC outputs high-level control [ax, ay, yaw_rate_cmd, v_theta].
    This function converts it to environment-specific action.
    
    MPCC Frame Mapping:
    - 2D Quadrotor (xz plane): MPCC x = drone x, MPCC y = drone z
    - 3D Quadrotor (xy plane): MPCC x = drone x, MPCC y = drone y, z = fixed altitude
    
    Control Mapping:
    - ax_des: acceleration in MPCC x direction (drone's horizontal/forward)
    - ay_des: acceleration in MPCC y direction (drone's z for 2D, y for 3D)
    
    For 2D planar quadrotor: action = [thrust, pitch_torque]
    For 3D quadrotor: action = [thrust, tau_x, tau_y, tau_z]
    
    Args:
        u_mpcc (ndarray): MPCC control [ax, ay, yaw_rate_cmd, v_theta] (4D)
        obs (ndarray): Current observation (6D or 12D)
        env: Environment (for mass, gravity, action limits)
        fixed_altitude (float): Desired altitude (for 3D quadrotor z control)
        
    Returns:
        ndarray: Environment action (2D or 4D depending on env)
    """
    obs = np.atleast_1d(obs)
    
    # Extract MPCC commands
    # u_mpcc = [ax, ay, yaw_rate_cmd, v_theta]
    ax_des = u_mpcc[0]      # MPCC x acceleration (drone horizontal)
    ay_des = u_mpcc[1]      # MPCC y acceleration (drone z for 2D, drone y for 3D)
    yaw_rate_cmd = u_mpcc[2]
    # v_theta = u_mpcc[3]   # Path progress rate - not used in action conversion
    
    # Determine action space dimension
    action_dim = env.action_space.shape[0]
    
    # ═══════════════════════════════════════════════════════════
    # Case 1: 2D Planar Quadrotor (action_dim = 2)
    # obs = [x, x_dot, z, z_dot, theta, theta_dot]
    # MPCC frame: x → x, z → y (vertical becomes MPCC y)
    # 
    # 2D Quadrotor Dynamics (from quadrotor.py):
    #   x_ddot = sin(theta) * (T1 + T2) / m
    #   z_ddot = cos(theta) * (T1 + T2) / m - g
    #   theta_ddot = L * (T2 - T1) / Iyy / sqrt(2)
    #
    # Action: [T1, T2] = individual thrust forces (not [total_thrust, torque]!)
    #   T1 = thrust from motors 1 & 4
    #   T2 = thrust from motors 2 & 3
    # ═══════════════════════════════════════════════════════════
    if action_dim == 2:
        # Extract state from 2D quadrotor observation
        pitch = obs[4]       # Current pitch angle (theta)
        pitch_dot = obs[5]   # Current pitch rate (theta_dot)
        
        # Get physical parameters
        m = env.MASS
        g = env.GRAVITY_ACC
        L = env.L            # Arm length
        Iyy = env.J[1, 1]    # Moment of inertia about y-axis
        
        # ───────────────────────────────────────────────────────
        # 1. Total thrust to achieve desired vertical acceleration
        # z_ddot = cos(theta) * F_total / m - g
        # F_total = m * (g + az_des) / cos(theta)
        # ───────────────────────────────────────────────────────
        az_des = ay_des  # MPCC y → drone z
        
        # Prevent division by zero and limit extreme thrust
        cos_pitch = np.cos(pitch)
        cos_pitch = np.clip(cos_pitch, 0.5, 1.0)  # Limit when pitch > 60 deg
        
        F_total = m * (g + az_des) / cos_pitch
        F_total = np.clip(F_total, 0.0, 4.0 * m * g)  # Limit max thrust
        
        # ───────────────────────────────────────────────────────
        # 2. Desired pitch angle to achieve horizontal acceleration
        # x_ddot = sin(theta) * F_total / m
        # For small angle: ax_des ≈ theta_des * g (when hovering)
        # More accurate: theta_des = arcsin(ax_des * m / F_total)
        # ───────────────────────────────────────────────────────
        # Limit ax_des to prevent impossible pitch angles
        ax_des_clipped = np.clip(ax_des, -0.5 * g, 0.5 * g)
        
        # Compute desired pitch (positive pitch = positive x acceleration)
        if F_total > 0.1:
            sin_pitch_des = ax_des_clipped * m / F_total
            sin_pitch_des = np.clip(sin_pitch_des, -0.5, 0.5)  # Limit to ~30 deg
            pitch_des = np.arcsin(sin_pitch_des)
        else:
            pitch_des = 0.0
        
        # ───────────────────────────────────────────────────────
        # 3. PD control for pitch angle → angular acceleration
        # theta_ddot_des = kp * (theta_des - theta) - kd * theta_dot
        # ───────────────────────────────────────────────────────
        kp_pitch = 50.0   # Proportional gain (higher for faster response)
        kd_pitch = 10.0   # Derivative gain (damping)
        
        theta_ddot_des = kp_pitch * (pitch_des - pitch) - kd_pitch * pitch_dot
        
        # ───────────────────────────────────────────────────────
        # 4. Convert angular acceleration to torque
        # tau = Iyy * theta_ddot
        # ───────────────────────────────────────────────────────
        tau = Iyy * theta_ddot_des
        
        # ───────────────────────────────────────────────────────
        # 5. Convert total thrust + torque to T1, T2
        # From dynamics: theta_ddot = L * (T2 - T1) / Iyy / sqrt(2)
        # Therefore: T2 - T1 = Iyy * theta_ddot * sqrt(2) / L = tau * sqrt(2) / L
        # 
        # T1 + T2 = F_total
        # T2 - T1 = tau * sqrt(2) / L
        # 
        # T1 = (F_total - tau * sqrt(2) / L) / 2
        # T2 = (F_total + tau * sqrt(2) / L) / 2
        # ───────────────────────────────────────────────────────
        sqrt2 = np.sqrt(2)
        T_diff = tau * sqrt2 / L  # T2 - T1
        
        T1 = (F_total - T_diff) / 2.0
        T2 = (F_total + T_diff) / 2.0
        
        # Ensure non-negative thrust (motors can't reverse)
        T1 = np.clip(T1, 0.0, None)
        T2 = np.clip(T2, 0.0, None)
        
        action = np.array([T1, T2])
    
    # ═══════════════════════════════════════════════════════════
    # Case 2: 3D Quadrotor (action_dim = 4)
    # obs = [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    # MPCC frame: x → x, y → y (horizontal plane), z = fixed altitude
    # ═══════════════════════════════════════════════════════════
    elif action_dim == 4:
        # Extract state from 3D quadrotor observation
        z = obs[2]          # Current altitude
        vz = obs[5]         # Vertical velocity
        yaw = obs[8]        # Current yaw angle
        
        # Action: [thrust, tau_x, tau_y, tau_z]
        
        # 1. Altitude control (hold fixed altitude)
        kp_z = 10.0
        kd_z = 5.0
        thrust = env.MASS * env.GRAVITY_ACC + kp_z * (fixed_altitude - z) - kd_z * vz
        
        # 2. XY acceleration to body torques
        # ax_des, ay_des are in world frame, need to convert to body frame
        R_yaw = Rotation.from_euler('z', yaw)
        a_world = np.array([ax_des, ay_des, 0.0])
        a_body = R_yaw.inv().apply(a_world)
        
        # Body frame: x = forward, y = left
        # To accelerate forward: pitch down (negative pitch) → positive tau_y
        # To accelerate left: roll left (positive roll) → negative tau_x
        k_accel = 0.1
        tau_y = a_body[0] * k_accel   # Forward accel via pitch
        tau_x = -a_body[1] * k_accel  # Lateral accel via roll
        
        # 3. Yaw rate control
        k_yaw = 0.05
        tau_z = yaw_rate_cmd * k_yaw
        
        action = np.array([thrust, tau_x, tau_y, tau_z])
    
    else:
        raise ValueError(f"Unsupported action dimension: {action_dim}. Expected 2 or 4.")
    
    # ═══════════════════════════════════════════════════════════
    # Clip to action space limits
    # ═══════════════════════════════════════════════════════════
    if hasattr(env, 'action_space'):
        action = np.clip(action, env.action_space.low, env.action_space.high)
    
    return action

