"""Visualize how waypoints look after env_trajectory_to_waypoints processes a circle.

This simulates the function's behavior without requiring the full environment.
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_env_trajectory_to_waypoints_circle(radius=0.9, center=[0, 1.0], 
                                                 num_cycles=2.5, sample_rate=10,
                                                 episode_len_sec=30, ctrl_freq=50):
    """Simulate what env_trajectory_to_waypoints does for a circle trajectory.
    
    This simulates the general sampling approach without circle-specific logic.
    """
    # Generate original trajectory points (time-based, like env.X_GOAL)
    num_timesteps = int(episode_len_sec * ctrl_freq)
    
    # Circle trajectory in xz plane (env), which becomes xy for MPCC
    theta = np.linspace(0, 2 * np.pi * num_cycles, num_timesteps)
    x_positions = center[0] + radius * np.cos(theta)
    y_positions = center[1] + radius * np.sin(theta)  # z in env becomes y in MPCC
    
    # General sampling approach (as in the updated function)
    # Check if trajectory appears to be closed
    first_point = np.array([x_positions[0], y_positions[0]])
    last_point = np.array([x_positions[-1], y_positions[-1]])
    distance_first_last = np.linalg.norm(first_point - last_point)
    is_closed = distance_first_last < 0.1
    
    # Sample waypoints uniformly based on sample_rate
    num_waypoints = max(4, int(episode_len_sec * sample_rate))
    step = max(1, num_timesteps // num_waypoints)
    
    waypoints = np.column_stack([
        x_positions[::step],
        y_positions[::step]
    ])
    
    # Ensure we have at least 4 waypoints
    if len(waypoints) < 4:
        indices = np.linspace(0, num_timesteps - 1, max(4, num_waypoints), dtype=int)
        waypoints = np.column_stack([
            x_positions[indices],
            y_positions[indices]
        ])
    
    # For closed trajectories, ensure proper closure
    if is_closed:
        if np.linalg.norm(waypoints[0] - waypoints[-1]) > 1e-6:
            waypoints = np.vstack([waypoints, waypoints[0:1]])
    
    return waypoints, x_positions, y_positions


def visualize_waypoints():
    """Visualize the waypoints from the simulated function."""
    
    # Circle parameters (matching the config file)
    radius = 0.9
    center = [0, 1.0]  # [offset_x, offset_z] -> [x, y] in MPCC
    num_cycles = 2.5
    sample_rate = 10
    episode_len_sec = 30
    
    print("Simulating env_trajectory_to_waypoints for circle trajectory...")
    print(f"Parameters: radius={radius}, center={center}, num_cycles={num_cycles}")
    print(f"sample_rate={sample_rate}, episode_len_sec={episode_len_sec}")
    
    waypoints, x_original, y_original = simulate_env_trajectory_to_waypoints_circle(
        radius=radius, center=center, num_cycles=num_cycles, 
        sample_rate=sample_rate, episode_len_sec=episode_len_sec
    )
    
    print(f"\nGenerated {len(waypoints)} waypoints")
    print(f"First waypoint: [{waypoints[0, 0]:.4f}, {waypoints[0, 1]:.4f}]")
    print(f"Last waypoint: [{waypoints[-1, 0]:.4f}, {waypoints[-1, 1]:.4f}]")
    distance_first_last = np.linalg.norm(waypoints[0] - waypoints[-1])
    print(f"Distance first to last: {distance_first_last:.6f}")
    print(f"Is properly closed: {distance_first_last < 0.1}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot original trajectory points (thin gray line, all points)
    ax.plot(x_original, y_original, 'gray', linewidth=0.5, alpha=0.3, 
            linestyle='--', label=f'Original trajectory ({len(x_original)} points)', 
            zorder=0)
    
    # Plot waypoints as connected path
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=3, 
            label='Waypoint path', alpha=0.8, zorder=2)
    
    # Plot waypoints as points
    ax.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=80, zorder=5, 
               label='Waypoints', edgecolors='darkred', linewidths=1)
    
    # Highlight first waypoint (green circle)
    ax.scatter(waypoints[0, 0], waypoints[0, 1], c='green', s=250, marker='o', 
               edgecolors='black', linewidths=3, zorder=7, 
               label='First waypoint (idx=0)')
    
    # Highlight last waypoint (orange square) - should match first if closed
    if len(waypoints) > 1:
        last_idx = -1 if len(waypoints) > 1 else 0
        ax.scatter(waypoints[last_idx, 0], waypoints[last_idx, 1], 
                  c='orange', s=250, marker='s', 
                  edgecolors='black', linewidths=3, zorder=7, 
                  label=f'Last waypoint (idx={last_idx})')
    
    # Annotate first several waypoints with indices
    n_annotate = min(10, len(waypoints))
    for i in range(n_annotate):
        ax.annotate(f'{i}', (waypoints[i, 0], waypoints[i, 1]), 
                   xytext=(10, 10), textcoords='offset points', 
                   fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.4', 
                   facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1))
    
    # Draw center of circle
    ax.scatter(center[0], center[1], c='purple', s=200, marker='+', 
              linewidths=4, zorder=6, label=f'Circle center ({center[0]}, {center[1]})')
    
    # Draw circle outline for reference
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    x_circle = center[0] + radius * np.cos(theta_circle)
    y_circle = center[1] + radius * np.sin(theta_circle)
    ax.plot(x_circle, y_circle, 'purple', linewidth=1.5, linestyle=':', 
           alpha=0.5, zorder=1, label='Perfect circle (reference)')
    
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m) [from Z in env]', fontsize=14, fontweight='bold')
    ax.set_title(f'Circle Trajectory Waypoints\n'
                 f'(Simulated: env_trajectory_to_waypoints, sample_rate={sample_rate})\n'
                 f'Waypoints: {len(waypoints)} | Original points: {len(x_original)} | '
                 f'Closed: {distance_first_last < 0.1}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    
    # Add info box
    info_text = (f'Parameters:\n'
                 f'Radius: {radius} m\n'
                 f'Center: ({center[0]}, {center[1]})\n'
                 f'Cycles: {num_cycles}\n'
                 f'Sample rate: {sample_rate} Hz\n'
                 f'Waypoints: {len(waypoints)}\n'
                 f'Closed: {distance_first_last < 0.1}')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', 
                     alpha=0.9, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    output_file = 'circle_waypoints_plot.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Plot saved as '{output_file}'")
    print("Displaying plot...")
    plt.show()


if __name__ == '__main__':
    visualize_waypoints()

