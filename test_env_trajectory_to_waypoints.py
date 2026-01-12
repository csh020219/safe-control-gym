"""Test script to visualize waypoints from env_trajectory_to_waypoints with real environment.

This script creates a real quadrotor environment with a circle trajectory,
calls env_trajectory_to_waypoints to convert it to waypoints, and plots the result.
"""

import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from safe_control_gym.controllers.mpcc.mpcc_utils import env_trajectory_to_waypoints
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import read_file
import munch


def test_circle_waypoints_from_env():
    """Test env_trajectory_to_waypoints with a real circle trajectory environment."""
    
    print("=" * 70)
    print("Testing env_trajectory_to_waypoints with Circle Trajectory")
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
    print(f"   Task info: {getattr(env, 'TASK_INFO', 'N/A')}")
    
    # Check X_GOAL
    if hasattr(env, 'X_GOAL') and env.X_GOAL is not None:
        print(f"\n3. Environment X_GOAL:")
        print(f"   Shape: {env.X_GOAL.shape}")
        print(f"   First point: {env.X_GOAL[0]}")
        print(f"   Last point: {env.X_GOAL[-1]}")
    else:
        print("\n3. Warning: env.X_GOAL is None or not available")
        env.close()
        return
    
    # Extract original trajectory points for comparison
    # For 2D quadrotor in xz plane: [x, x_dot, z, z_dot, theta, theta_dot]
    x_original = env.X_GOAL[:, 0]  # x position
    z_original = env.X_GOAL[:, 2]  # z position (becomes y in MPCC)
    
    print(f"\n4. Original trajectory points: {len(x_original)} points")
    print(f"   X range: [{x_original.min():.3f}, {x_original.max():.3f}]")
    print(f"   Z range: [{z_original.min():.3f}, {z_original.max():.3f}]")
    
    # Call env_trajectory_to_waypoints
    print("\n5. Calling env_trajectory_to_waypoints with sample_rate=10...")
    try:
        waypoints = env_trajectory_to_waypoints(env, sample_rate=10)
        print(f"   ✓ Success! Generated {len(waypoints)} waypoints")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    print(f"\n6. Waypoint details:")
    print(f"   Shape: {waypoints.shape}")
    print(f"   First waypoint: [{waypoints[0, 0]:.4f}, {waypoints[0, 1]:.4f}]")
    print(f"   Last waypoint: [{waypoints[-1, 0]:.4f}, {waypoints[-1, 1]:.4f}]")
    
    # Check closure
    distance_first_last = np.linalg.norm(waypoints[0] - waypoints[-1])
    is_closed = distance_first_last < 0.1
    print(f"   Distance first to last: {distance_first_last:.6f}")
    print(f"   Is properly closed: {is_closed}")
    
    # Get trajectory parameters for reference
    task_info = getattr(env, 'TASK_INFO', {})
    if isinstance(task_info, dict):
        radius = task_info.get('trajectory_scale', 0.9)
        center_offset = task_info.get('trajectory_position_offset', [0, 1.0])
    else:
        # If task_info is not a dict (might be munch object)
        radius = getattr(task_info, 'trajectory_scale', 0.9)
        center_offset = getattr(task_info, 'trajectory_position_offset', [0, 1.0])
    
    # Create visualization
    print("\n7. Creating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Plot original trajectory (all points) - thin gray line
    ax.plot(x_original, z_original, 'gray', linewidth=0.5, alpha=0.3, 
            linestyle='--', label=f'Original trajectory ({len(x_original)} points)', 
            zorder=0)
    
    # Plot waypoints as connected path - thick blue line
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=3.5, 
            label='Waypoint path', alpha=0.8, zorder=2)
    
    # Plot waypoints as red dots
    ax.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=90, zorder=5, 
               label='Waypoints', edgecolors='darkred', linewidths=1)
    
    # Highlight first waypoint (green circle)
    ax.scatter(waypoints[0, 0], waypoints[0, 1], c='green', s=300, marker='o', 
               edgecolors='black', linewidths=3, zorder=7, 
               label='First waypoint (idx=0)')
    
    # Highlight last waypoint (orange square)
    if len(waypoints) > 1:
        last_idx = -1 if len(waypoints) > 1 else 0
        ax.scatter(waypoints[last_idx, 0], waypoints[last_idx, 1], 
                  c='orange', s=300, marker='s', 
                  edgecolors='black', linewidths=3, zorder=7, 
                  label=f'Last waypoint (idx={last_idx})')
    
    # Annotate first several waypoints with indices
    n_annotate = min(12, len(waypoints))
    for i in range(n_annotate):
        ax.annotate(f'{i}', (waypoints[i, 0], waypoints[i, 1]), 
                   xytext=(12, 12), textcoords='offset points', 
                   fontsize=11, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.5', 
                   facecolor='yellow', alpha=0.85, 
                   edgecolor='black', linewidth=1.5))
    
    # Draw perfect circle reference
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    x_circle = center_offset[0] + radius * np.cos(theta_circle)
    y_circle = center_offset[1] + radius * np.sin(theta_circle)
    ax.plot(x_circle, y_circle, 'purple', linewidth=2, linestyle=':', 
           alpha=0.6, zorder=1, label='Perfect circle (reference)')
    
    # Draw circle center
    ax.scatter(center_offset[0], center_offset[1], c='purple', s=300, marker='+', 
              linewidths=5, zorder=6, label=f'Circle center ({center_offset[0]}, {center_offset[1]})')
    
    # Labels and title
    ax.set_xlabel('X (m)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Y (m) [from Z in env]', fontsize=15, fontweight='bold')
    ax.set_title(
        f'Circle Trajectory Waypoints from env_trajectory_to_waypoints\n'
        f'(Real Environment, sample_rate=10)\n'
        f'Waypoints: {len(waypoints)} | Original points: {len(x_original)} | '
        f'Closed: {is_closed}',
        fontsize=15, fontweight='bold'
    )
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
    ax.legend(loc='best', fontsize=11, framealpha=0.95, 
             fancybox=True, shadow=True)
    ax.set_aspect('equal', adjustable='box')
    
    # Get task info values safely
    if isinstance(task_info, dict):
        num_cycles = task_info.get("num_cycles", 1)
        plane = task_info.get("trajectory_plane", "xz")
    else:
        num_cycles = getattr(task_info, "num_cycles", 1)
        plane = getattr(task_info, "trajectory_plane", "xz")
    
    # Add info box
    info_text = (
        f'Trajectory Parameters:\n'
        f'Radius: {radius} m\n'
        f'Center: ({center_offset[0]}, {center_offset[1]})\n'
        f'Num cycles: {num_cycles}\n'
        f'Plane: {plane}\n'
        f'\nSampling:\n'
        f'Sample rate: 10 Hz\n'
        f'Waypoints: {len(waypoints)}\n'
        f'Original: {len(x_original)}\n'
        f'Reduction: {len(x_original) // len(waypoints):.1f}x\n'
        f'\nClosure:\n'
        f'Distance: {distance_first_last:.6f} m\n'
        f'Closed: {is_closed}'
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=1.0', facecolor='wheat', 
                     alpha=0.95, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'circle_waypoints_from_env.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"   ✓ Plot saved as '{output_file}'")
    
    # Show plot
    print("\n8. Displaying plot...")
    plt.show()
    
    # Cleanup
    env.close()
    print("\n9. Test completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    try:
        test_circle_waypoints_from_env()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

