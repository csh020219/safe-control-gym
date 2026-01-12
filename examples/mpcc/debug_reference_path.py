"""Debug script to visualize MPCC reference path generation.

This script loads the environment, generates waypoints, and plots:
1. Original environment trajectory (env.X_GOAL)
2. Generated waypoints
3. Bezier-interpolated reference path
4. MPCC lookup table reference path

Run: python examples/mpcc/debug_reference_path.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import read_file
import munch
from safe_control_gym.controllers.mpcc.mpcc_utils import (
    env_trajectory_to_waypoints,
    interpolate_bezier,
    eval_bezier,
    generate_lookup_table_from_waypoints
)


def debug_reference_path_generation():
    """Debug function to visualize reference path generation."""
    
    # Get the script's directory and find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir is examples/mpcc/, so project root is 2 levels up
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Load configuration from config file (relative to project root)
    config_file = os.path.join(project_root, 'examples', 'mpcc', 'config_overrides', 
                                'quadrotor_2D', 'quadrotor_2D_tracking_env_circle.yaml')
    
    print(f"Loading configuration from: {config_file}")
    config_dict = read_file(config_file)
    
    if config_dict is None:
        raise FileNotFoundError(f"Config file not found: {config_file}\n"
                              f"Expected at: {os.path.abspath(config_file)}")
    
    # Extract task_config from the loaded yaml
    task_config = config_dict.get('task_config', config_dict)
    task_name = 'quadrotor'  # Default task name
    
    # Make it munch so we can use attribute access
    task_config = munch.munchify(task_config)
    
    # Create environment
    env_func = partial(make, task_name, **task_config)
    env = env_func(gui=False)
    
    print(f"\n{'='*60}")
    print("MPCC Reference Path Generation Debug")
    print(f"{'='*60}\n")
    
    print(f"Environment trajectory type: {env.TASK_INFO.get('trajectory_type')}")
    print(f"Trajectory plane: {env.TASK_INFO.get('trajectory_plane')}")
    print(f"Position offset: {env.TASK_INFO.get('trajectory_position_offset')}")
    print(f"Scale: {env.TASK_INFO.get('trajectory_scale')}")
    print(f"Number of cycles: {env.TASK_INFO.get('num_cycles')}")
    print(f"X_GOAL shape: {env.X_GOAL.shape}\n")
    
    # Extract environment trajectory positions
    if hasattr(env, 'STATE_LABELS'):
        labels = env.STATE_LABELS
        x_idx = labels.index('x')
        if 'z' in labels:
            y_idx = labels.index('z')  # 2D quadrotor: z becomes y in MPCC
        else:
            y_idx = labels.index('y')
        env_x = env.X_GOAL[:, x_idx]
        env_y = env.X_GOAL[:, y_idx]
    else:
        env_x = env.X_GOAL[:, 0]
        env_y = env.X_GOAL[:, 2]  # 2D quadrotor uses z
    
    # Generate waypoints from environment trajectory
    # sample_rate=2 gives ~20 waypoints per cycle, ideal for Bezier interpolation
    print("Step 1: Generating waypoints from env.X_GOAL...")
    waypoints = env_trajectory_to_waypoints(env, sample_rate=2)
    print(f"Generated {len(waypoints)} waypoints\n")
    
    # Generate Bezier interpolation
    print("Step 2: Generating Bezier interpolation...")
    a, b = interpolate_bezier(waypoints)
    
    # Generate smooth Bezier curve for visualization
    n_bezier_points = 500
    n_wp = len(waypoints)
    t_bezier = np.linspace(0, n_wp, n_bezier_points)
    bezier_curve = []
    for t in t_bezier:
        bezier_curve.append(eval_bezier(waypoints, a, b, t))
    bezier_curve = np.array(bezier_curve)
    
    # Generate lookup table (MPCC reference path)
    print("Step 3: Generating lookup table (MPCC reference path)...")
    table, smax = generate_lookup_table_from_waypoints(waypoints, track_width=0.5, density=100)
    ref_path = table[:, 2:4]  # x, y columns
    print(f"Generated lookup table with {len(table)} points\n")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # === Left plot: Waypoints and Bezier interpolation ===
    ax1 = axes[0]
    
    # Plot original environment trajectory
    ax1.plot(env_x, env_y, 'm:', linewidth=1.5, alpha=0.5, label='Environment Trajectory (env.X_GOAL)')
    
    # Plot waypoints
    ax1.plot(waypoints[:, 0], waypoints[:, 1], 'bo-', markersize=8, linewidth=2, label='Waypoints', zorder=5)
    ax1.plot(waypoints[0, 0], waypoints[0, 1], 'go', markersize=15, label='First waypoint', zorder=6)
    ax1.plot(waypoints[-1, 0], waypoints[-1, 1], 'ro', markersize=15, label='Last waypoint', zorder=6)
    
    # Plot Bezier interpolated curve
    ax1.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'g-', linewidth=2, 
            label='Bezier Interpolated Path', alpha=0.7)
    
    # Check closure
    first_last_dist = np.linalg.norm(waypoints[0] - waypoints[-1])
    closure_status = "CLOSED ✓" if first_last_dist < 1e-6 else f"NOT CLOSED ✗ (dist={first_last_dist:.6f})"
    
    ax1.set_title(f'Waypoint Generation & Bezier Interpolation\n'
                 f'{len(waypoints)} waypoints - {closure_status}')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # === Right plot: MPCC Reference Path (Lookup Table) ===
    ax2 = axes[1]
    
    # Plot original environment trajectory
    ax2.plot(env_x, env_y, 'm:', linewidth=1.5, alpha=0.5, label='Environment Trajectory (env.X_GOAL)')
    
    # Plot waypoints
    ax2.plot(waypoints[:, 0], waypoints[:, 1], 'bo', markersize=5, alpha=0.5, label='Waypoints', zorder=4)
    
    # Plot MPCC reference path (from lookup table - what MPCC actually uses)
    ax2.plot(ref_path[:, 0], ref_path[:, 1], 'r--', linewidth=2, 
            label=f'MPCC Reference Path ({len(ref_path)} points)', zorder=3)
    
    # Check for discontinuities in reference path
    path_diff = np.diff(ref_path, axis=0)
    path_distances = np.linalg.norm(path_diff, axis=1)
    max_jump = np.max(path_distances)
    mean_jump = np.mean(path_distances)
    
    # Highlight jumps/discontinuities
    jump_threshold = mean_jump * 3  # 3x mean is suspicious
    jump_indices = np.where(path_distances > jump_threshold)[0]
    if len(jump_indices) > 0:
        for idx in jump_indices:
            ax2.plot([ref_path[idx, 0], ref_path[idx+1, 0]], 
                    [ref_path[idx, 1], ref_path[idx+1, 1]], 
                    'r-', linewidth=3, alpha=0.8, zorder=10)
        print(f"⚠️ WARNING: Found {len(jump_indices)} potential discontinuities in reference path!")
        print(f"   Max jump: {max_jump:.6f}m, Mean jump: {mean_jump:.6f}m")
        print(f"   Jump threshold: {jump_threshold:.6f}m\n")
    
    ax2.set_title(f'MPCC Reference Path (Lookup Table)\n'
                 f'Track length: {smax:.2f}m, Points: {len(ref_path)}\n'
                 f'Max jump: {max_jump:.6f}m, Mean jump: {mean_jump:.6f}m')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('mpcc_reference_path_debug.png', dpi=150, bbox_inches='tight')
    print("Debug plot saved to: mpcc_reference_path_debug.png")
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  - Environment trajectory points: {len(env_x)}")
    print(f"  - Generated waypoints: {len(waypoints)}")
    print(f"  - Waypoint closure: {first_last_dist:.6f} ({'✓' if first_last_dist < 1e-6 else '✗'})")
    print(f"  - Bezier curve points: {len(bezier_curve)}")
    print(f"  - MPCC reference path points: {len(ref_path)}")
    print(f"  - Track length: {smax:.2f}m")
    print(f"  - Reference path discontinuities: {len(jump_indices)}")
    print(f"{'='*60}\n")
    
    env.close()


if __name__ == '__main__':
    debug_reference_path_generation()

