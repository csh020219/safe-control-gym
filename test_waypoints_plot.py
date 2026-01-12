"""Temporary test script to visualize waypoints from env_trajectory_to_waypoints.

This script creates a mock environment with a circle trajectory and shows
how the waypoints look after env_trajectory_to_waypoints processes them.
"""

import numpy as np
import matplotlib.pyplot as plt


# Mock environment class to simulate what env_trajectory_to_waypoints needs
class MockEnv:
    """Mock environment for testing waypoint generation."""
    
    def __init__(self):
        # Circle trajectory parameters
        self.trajectory_type = 'circle'
        self.num_cycles = 2.5
        self.trajectory_plane = 'xz'
        self.position_offset = [0, 1.0]  # [offset_x, offset_z]
        self.radius = 0.9
        
        self.EPISODE_LEN_SEC = 30
        self.CTRL_FREQ = 50
        self.CTRL_TIMESTEP = 1.0 / self.CTRL_FREQ
        
        # Generate trajectory data (time-based)
        num_timesteps = int(self.EPISODE_LEN_SEC * self.CTRL_FREQ)
        
        # Generate circle trajectory in xz plane
        # In xz plane: x = offset_x + radius * cos(theta), z = offset_z + radius * sin(theta)
        # For MPCC xy: x from env becomes x in MPCC, z from env becomes y in MPCC
        theta = np.linspace(0, 2 * np.pi * self.num_cycles, num_timesteps)
        x_traj = self.position_offset[0] + self.radius * np.cos(theta)
        z_traj = self.position_offset[1] + self.radius * np.sin(theta)
        
        # Create X_GOAL-like structure for 2D quadrotor
        # Format: [x, x_dot, z, z_dot, 0, 0]
        self.X_GOAL = np.zeros((num_timesteps, 6))
        self.X_GOAL[:, 0] = x_traj  # x
        self.X_GOAL[:, 2] = z_traj  # z (becomes y in MPCC)
        
        # Calculate derivatives
        dt = self.CTRL_TIMESTEP
        self.X_GOAL[1:, 1] = np.diff(x_traj) / dt  # x_dot
        self.X_GOAL[1:, 3] = np.diff(z_traj) / dt  # z_dot
        
        # Task info (using xz plane to avoid regeneration path)
        self.TASK = 'traj_tracking'
        self.TASK_INFO = {
            'trajectory_type': 'circle',
            'num_cycles': self.num_cycles,
            'trajectory_plane': self.trajectory_plane,  # 'xz' - won't trigger regeneration
            'trajectory_position_offset': self.position_offset,
            'trajectory_scale': self.radius,
        }
        
        # State labels for 2D quadrotor (allows direct extraction)
        self.STATE_LABELS = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']


def test_circle_waypoints():
    """Test circle trajectory waypoints visualization."""
    
    # Import the function
    from safe_control_gym.controllers.mpcc.mpcc_utils import env_trajectory_to_waypoints
    
    # Create mock environment
    print("Creating mock environment with circle trajectory...")
    env = MockEnv()
    
    print(f"Environment X_GOAL shape: {env.X_GOAL.shape}")
    print(f"Trajectory parameters: radius={env.radius}, num_cycles={env.num_cycles}")
    print(f"Position offset: {env.position_offset}")
    
    # Call env_trajectory_to_waypoints
    print("\nCalling env_trajectory_to_waypoints with sample_rate=10...")
    waypoints = env_trajectory_to_waypoints(env, sample_rate=10)
    
    print(f"\nGenerated {len(waypoints)} waypoints")
    print(f"Waypoints shape: {waypoints.shape}")
    print(f"First waypoint: [{waypoints[0, 0]:.4f}, {waypoints[0, 1]:.4f}]")
    print(f"Last waypoint: [{waypoints[-1, 0]:.4f}, {waypoints[-1, 1]:.4f}]")
    distance_first_last = np.linalg.norm(waypoints[0] - waypoints[-1])
    print(f"Distance first to last: {distance_first_last:.6f}")
    print(f"Is closed: {distance_first_last < 0.1}")
    
    # Plot the waypoints
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot waypoints as connected points
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=2.5, 
            label='Waypoint path', alpha=0.7, zorder=1)
    ax.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=60, zorder=5, 
               label='Waypoints', edgecolors='darkred', linewidths=0.5)
    
    # Highlight first and last waypoints
    ax.scatter(waypoints[0, 0], waypoints[0, 1], c='green', s=200, marker='o', 
               edgecolors='black', linewidths=2.5, zorder=6, label='First waypoint')
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], c='orange', s=200, marker='s', 
               edgecolors='black', linewidths=2.5, zorder=6, label='Last waypoint')
    
    # Annotate first few waypoints with indices
    for i in range(min(8, len(waypoints))):
        ax.annotate(f'{i}', (waypoints[i, 0], waypoints[i, 1]), 
                   xytext=(8, 8), textcoords='offset points', fontsize=9, 
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='yellow', alpha=0.7))
    
    # Plot original trajectory points (thinner, for comparison)
    x_original = env.X_GOAL[:, 0]
    y_original = env.X_GOAL[:, 2]  # z becomes y in MPCC
    ax.plot(x_original, y_original, 'gray', linewidth=0.5, alpha=0.3, 
            linestyle='--', label='Original trajectory (all points)', zorder=0)
    
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m) [from Z in env]', fontsize=14, fontweight='bold')
    ax.set_title(f'Circle Trajectory Waypoints\n'
                 f'(Generated by env_trajectory_to_waypoints, sample_rate=10)\n'
                 f'Total waypoints: {len(waypoints)} | '
                 f'Original trajectory points: {len(x_original)}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    # Add text box with info
    info_text = (f'Circle Parameters:\n'
                 f'Radius: {env.radius} m\n'
                 f'Center: ({env.position_offset[0]}, {env.position_offset[1]})\n'
                 f'Cycles: {env.num_cycles}\n'
                 f'Waypoints: {len(waypoints)}\n'
                 f'Closed: {distance_first_last < 0.1}')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('circle_waypoints_plot.png', dpi=200, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'circle_waypoints_plot.png'")
    print("Showing plot...")
    plt.show()


if __name__ == '__main__':
    test_circle_waypoints()
