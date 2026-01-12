"""Test script to visualize the raw environment trajectory (env.X_GOAL) before waypoint conversion.

This script creates a real quadrotor environment with a circle trajectory
and plots the original trajectory points from env.X_GOAL before they are
converted to waypoints by env_trajectory_to_waypoints.
"""

import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import read_file
import munch


def test_raw_env_trajectory():
    """Visualize the raw environment trajectory before waypoint conversion."""
    
    print("=" * 70)
    print("Visualizing Raw Environment Trajectory (env.X_GOAL)")
    print("=" * 70)
    
    # Create configuration from config file
    config_file = 'examples/mpcc/config_overrides/quadrotor_2D/quadrotor_2D_tracking_env_circle.yaml'
    
    print(f"\n1. Loading configuration from: {config_file}")
    config_dict = read_file(config_file)
    
    if config_dict is None:
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Extract task_config from the loaded yaml
    task_config = config_dict.get('task_config', config_dict)
    task_name = 'quadrotor'  # Default task name
    
    # Make it munch so we can use attribute access
    task_config = munch.munchify(task_config)
    
    print("2. Creating environment with circle trajectory...")
    env_func = partial(
        make,
        task_name,
        **task_config
    )
    
    # Create environment (no GUI for faster execution)
    env = env_func(gui=False)
    
    print(f"   Environment created successfully!")
    print(f"   Task: {getattr(env, 'TASK', 'N/A')}")
    
    # Check X_GOAL
    if not hasattr(env, 'X_GOAL') or env.X_GOAL is None:
        print("\n3. Error: env.X_GOAL is None or not available")
        env.close()
        return
    
    print(f"\n3. Environment X_GOAL (Raw Trajectory):")
    print(f"   Shape: {env.X_GOAL.shape}")
    print(f"   Type: {type(env.X_GOAL)}")
    print(f"   First point: {env.X_GOAL[0]}")
    print(f"   Last point: {env.X_GOAL[-1]}")
    
    # Get task info for reference
    task_info = getattr(env, 'TASK_INFO', {})
    if not isinstance(task_info, dict):
        # If task_info is munch object
        radius = getattr(task_info, 'trajectory_scale', 0.9)
        center_offset = getattr(task_info, 'trajectory_position_offset', [0, 1.0])
        num_cycles = getattr(task_info, 'num_cycles', 2.5)
        plane = getattr(task_info, 'trajectory_plane', 'xz')
    else:
        radius = task_info.get('trajectory_scale', 0.9)
        center_offset = task_info.get('trajectory_position_offset', [0, 1.0])
        num_cycles = task_info.get('num_cycles', 2.5)
        plane = task_info.get('trajectory_plane', 'xz')
    
    print(f"\n4. Trajectory Parameters:")
    print(f"   Type: circle")
    print(f"   Radius: {radius} m")
    print(f"   Center offset: {center_offset}")
    print(f"   Number of cycles: {num_cycles}")
    print(f"   Plane: {plane}")
    print(f"   Episode length: {getattr(env, 'EPISODE_LEN_SEC', 'N/A')} sec")
    print(f"   Control frequency: {getattr(env, 'CTRL_FREQ', 'N/A')} Hz")
    
    # Extract trajectory points based on environment type
    print(f"\n5. Extracting trajectory points...")
    
    # For 2D quadrotor in xz plane: X_GOAL structure is [x, x_dot, z, z_dot, theta, theta_dot]
    # Check state labels if available
    if hasattr(env, 'STATE_LABELS'):
        labels = env.STATE_LABELS
        print(f"   State labels: {labels}")
        try:
            x_idx = labels.index('x')
            # For 2D quadrotor, 'y' in MPCC corresponds to 'z' in environment
            if 'z' in labels:
                z_idx = labels.index('z')  # 2D quadrotor: z becomes y in MPCC
            elif 'y' in labels:
                z_idx = labels.index('y')  # 3D quadrotor
            else:
                raise ValueError("Cannot find y or z in STATE_LABELS")
            
            x_positions = env.X_GOAL[:, x_idx]
            z_positions = env.X_GOAL[:, z_idx]
        except ValueError as e:
            print(f"   Warning: Cannot extract from STATE_LABELS: {e}")
            # Fallback: assume standard structure
            x_positions = env.X_GOAL[:, 0]
            z_positions = env.X_GOAL[:, 2] if env.X_GOAL.shape[1] > 2 else env.X_GOAL[:, 1]
    else:
        # Fallback: assume standard structure for 2D quadrotor
        # [x, x_dot, z, z_dot, theta, theta_dot]
        x_positions = env.X_GOAL[:, 0]  # x
        if env.X_GOAL.shape[1] > 2:
            z_positions = env.X_GOAL[:, 2]  # z (becomes y in MPCC)
        else:
            z_positions = env.X_GOAL[:, 1]  # y
    
    print(f"   Extracted {len(x_positions)} trajectory points")
    print(f"   X range: [{x_positions.min():.4f}, {x_positions.max():.4f}] m")
    print(f"   Z range: [{z_positions.min():.4f}, {z_positions.max():.4f}] m")
    print(f"   First point: ({x_positions[0]:.4f}, {z_positions[0]:.4f})")
    print(f"   Last point: ({x_positions[-1]:.4f}, {z_positions[-1]:.4f})")
    
    # Check if trajectory is closed
    first_point = np.array([x_positions[0], z_positions[0]])
    last_point = np.array([x_positions[-1], z_positions[-1]])
    distance_first_last = np.linalg.norm(first_point - last_point)
    is_closed = distance_first_last < 0.1
    
    print(f"   Distance first to last: {distance_first_last:.6f} m")
    print(f"   Is closed: {is_closed}")
    
    # Create visualization
    print(f"\n6. Creating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Plot the raw trajectory (all points)
    ax.plot(x_positions, z_positions, 'b-', linewidth=2.5, 
            label=f'Raw trajectory ({len(x_positions)} points)', 
            alpha=0.8, zorder=2)
    
    # Plot trajectory points as dots (subsample for clarity)
    step = max(1, len(x_positions) // 500)  # Show max 500 points
    ax.scatter(x_positions[::step], z_positions[::step], 
              c='blue', s=20, zorder=3, alpha=0.6,
              label=f'Trajectory points (subsampled, every {step} points)')
    
    # Highlight first point
    ax.scatter(x_positions[0], z_positions[0], c='green', s=300, marker='o', 
              edgecolors='black', linewidths=3, zorder=6, 
              label='First point (t=0)')
    
    # Highlight last point
    ax.scatter(x_positions[-1], z_positions[-1], c='red', s=300, marker='s', 
              edgecolors='black', linewidths=3, zorder=6, 
              label='Last point')
    
    # Annotate first few points with time indices
    n_annotate = min(10, len(x_positions))
    for i in range(0, n_annotate, max(1, n_annotate // 10)):
        ax.annotate(f't={i}', (x_positions[i], z_positions[i]), 
                   xytext=(10, 10), textcoords='offset points', 
                   fontsize=9, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='yellow', alpha=0.8, 
                   edgecolor='black', linewidth=1))
    
    # Draw perfect circle reference
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    x_circle = center_offset[0] + radius * np.cos(theta_circle)
    y_circle = center_offset[1] + radius * np.sin(theta_circle)
    ax.plot(x_circle, y_circle, 'purple', linewidth=2, linestyle=':', 
           alpha=0.7, zorder=1, label='Perfect circle (reference)')
    
    # Draw circle center
    ax.scatter(center_offset[0], center_offset[1], c='purple', s=400, marker='+', 
              linewidths=5, zorder=7, label=f'Circle center ({center_offset[0]}, {center_offset[1]})')
    
    # Labels and title
    ax.set_xlabel('X (m) [from env]', fontsize=15, fontweight='bold')
    ax.set_ylabel('Z (m) [from env, becomes Y in MPCC]', fontsize=15, fontweight='bold')
    ax.set_title(
        f'Raw Environment Trajectory (env.X_GOAL)\n'
        f'Before env_trajectory_to_waypoints conversion\n'
        f'Total points: {len(x_positions)} | '
        f'Closed: {is_closed} | Plane: {plane}',
        fontsize=15, fontweight='bold'
    )
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, 
             fancybox=True, shadow=True)
    ax.set_aspect('equal', adjustable='box')
    
    # Add info box
    ctrl_freq = getattr(env, 'CTRL_FREQ', 50)
    episode_len = getattr(env, 'EPISODE_LEN_SEC', 30)
    info_text = (
        f'Trajectory Parameters:\n'
        f'Type: circle\n'
        f'Radius: {radius} m\n'
        f'Center: ({center_offset[0]}, {center_offset[1]})\n'
        f'Num cycles: {num_cycles}\n'
        f'Plane: {plane}\n'
        f'\nEnvironment Settings:\n'
        f'Episode length: {episode_len} sec\n'
        f'Control frequency: {ctrl_freq} Hz\n'
        f'Total points: {len(x_positions)}\n'
        f'Points per cycle: ~{len(x_positions) / num_cycles:.0f}\n'
        f'\nTrajectory Properties:\n'
        f'X range: [{x_positions.min():.3f}, {x_positions.max():.3f}]\n'
        f'Z range: [{z_positions.min():.3f}, {z_positions.max():.3f}]\n'
        f'Distance first-last: {distance_first_last:.6f} m\n'
        f'Closed: {is_closed}'
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=1.0', facecolor='wheat', 
                     alpha=0.95, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'raw_env_trajectory.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"   ✓ Plot saved as '{output_file}'")
    
    # Show plot
    print("\n7. Displaying plot...")
    plt.show()
    
    # Cleanup
    env.close()
    print("\n8. Visualization completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    try:
        test_raw_env_trajectory()
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\n\nVisualization failed with error: {e}")
        import traceback
        traceback.print_exc()

