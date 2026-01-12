"""An MPCC (Model Predictive Contouring Control) example."""

import os
import pickle
from functools import partial
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def calculate_cycle_duration(trajs_data, env, ctrl=None, algo_name=''):
    """Calculate actual cycle duration using controller-specific methods.
    
    For MPCC: Uses theta (path progress) from error_history to detect cycle completions
    For MPC: Uses reference trajectory period or step-based calculation
    
    Args:
        trajs_data: Trajectory data from experiment
        env: Environment object
        ctrl: Controller object (optional, needed for MPCC)
        algo_name: Name of the algorithm (for display purposes)
    
    Returns:
        cycle_duration (float): Average cycle duration in seconds, or None if not applicable
    """
    if not hasattr(env, 'TASK') or env.TASK != Task.TRAJ_TRACKING:
        return None
    
    if not hasattr(env, 'TASK_INFO') or 'num_cycles' not in env.TASK_INFO:
        return None
    
    # Calculate expected cycle period
    episode_len_sec = env.EPISODE_LEN_SEC
    num_cycles = env.TASK_INFO['num_cycles']
    expected_cycle_period = episode_len_sec / num_cycles
    
    # Get trajectory data
    if 'state' not in trajs_data or len(trajs_data['state']) == 0:
        return None
    
    states = trajs_data['state'][0]  # First episode
    timestamps = trajs_data.get('timestamp', [[]])[0] if 'timestamp' in trajs_data else None
    
    if len(states) < 10:  # Need minimum steps
        return None
    
    # ═══════════════════════════════════════════════════════════
    # MPCC: Use theta (path progress) from error_history
    # ═══════════════════════════════════════════════════════════
    if algo_name == 'MPCC' and ctrl is not None:
        error_history = None
        if hasattr(ctrl, 'error_history') and ctrl.error_history:
            error_history = ctrl.error_history
        elif hasattr(ctrl, '_last_error_history') and ctrl._last_error_history:
            error_history = ctrl._last_error_history
        
        if error_history and hasattr(ctrl, 'track_manager') and hasattr(ctrl.track_manager, 's_max'):
            s_max = ctrl.track_manager.s_max
            thetas = [e.get('theta', 0) for e in error_history]
            
            if len(thetas) > 1:
                # Calculate total cycles completed from final theta
                final_theta = thetas[-1]
                total_cycles_completed = final_theta / s_max
                
                if total_cycles_completed > 0:
                    # Use actual timestamps if available
                    if timestamps is not None and len(timestamps) > 1:
                        total_duration = timestamps[-1] - timestamps[0]
                    else:
                        total_duration = len(thetas) / env.CTRL_FREQ
                    
                    # Average cycle duration = total duration / number of cycles completed
                    actual_cycle_duration = total_duration / total_cycles_completed
                else:
                    # No progress made, use expected value
                    actual_cycle_duration = expected_cycle_period
            else:
                # Not enough data, use expected value
                actual_cycle_duration = expected_cycle_period
        else:
            # No error_history available, use expected value
            actual_cycle_duration = expected_cycle_period
    
    # ═══════════════════════════════════════════════════════════
    # MPC: Use reference trajectory period or step-based calculation
    # ═══════════════════════════════════════════════════════════
    else:
        # For MPC, reference trajectory period is known
        # But we want actual execution time
        
        # Use actual timestamps if available
        if timestamps is not None and len(timestamps) > 1:
            total_duration = timestamps[-1] - timestamps[0]
            actual_cycle_duration = total_duration / num_cycles
        else:
            # Fallback: use step count and control frequency
            total_steps = len(states)
            total_duration = total_steps / env.CTRL_FREQ
            actual_cycle_duration = total_duration / num_cycles
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"CYCLE DURATION ANALYSIS ({algo_name})")
    print(f"{'='*60}")
    print(f"Episode length (expected): {episode_len_sec:.2f} sec")
    if timestamps is not None and len(timestamps) > 1:
        total_duration = timestamps[-1] - timestamps[0]
        print(f"Episode length (actual): {total_duration:.4f} sec")
    print(f"Number of cycles: {num_cycles}")
    print(f"Expected cycle period: {expected_cycle_period:.4f} sec")
    print(f"Actual cycle duration: {actual_cycle_duration:.4f} sec")
    print(f"Difference: {actual_cycle_duration - expected_cycle_period:.4f} sec")
    if expected_cycle_period > 0:
        print(f"Percentage difference: {((actual_cycle_duration - expected_cycle_period) / expected_cycle_period * 100):.2f}%")
    print(f"{'='*60}\n")
    
    return actual_cycle_duration


def run(gui=False, plot=True, n_episodes=1, n_steps=None, save_data=False):
    """The main function running MPCC experiments.

    Args:
        gui (bool): Whether to display the gui.
        plot (bool): Whether to plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    """

    # Create the configuration dictionary
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Extract gui from config and remove it to avoid keyword argument conflict with partial
    task_config_dict = dict(config.task_config)
    gui_value = task_config_dict.pop('gui', gui)  # Use config gui if available, otherwise use function parameter
    
    # Create an environment
    env_func = partial(make,
                       config.task,
                       **task_config_dict
                       )
    env = env_func(gui=gui_value)

    # Create MPCC controller
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )

    # MPCC-specific initialization
    print(f"\n{'='*50}")
    print(f"MPCC Experiment")
    print(f"Task: {config.task}")
    print(f"Horizon: {config.algo_config.get('horizon', 'N/A')}")
    print(f"Contouring weight (Qc): {config.algo_config.get('q_contour', 'N/A')}")
    print(f"Lag weight (Ql): {config.algo_config.get('q_lag', 'N/A')}")
    print(f"{'='*50}\n")

    # Run the experiment
    experiment = BaseExperiment(env=env, ctrl=ctrl)
    trajs_data, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)

    # Calculate and display cycle duration
    calculate_cycle_duration(trajs_data, env, ctrl=ctrl, algo_name='MPCC')

    if plot:
        for i in range(len(trajs_data['obs'])):
            post_analysis(
                trajs_data['obs'][i], 
                trajs_data['action'][i], 
                ctrl.env,
                ctrl,  # MPCC controller 전달
                trajs_data=trajs_data,  # Pass trajs_data for Excel export
                episode_idx=i  # Episode index for Excel filename
            )

    ctrl.close()
    env.close()

    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))
    
    # MPCC-specific metrics
    if hasattr(ctrl, 'get_mpcc_metrics'):
        mpcc_metrics = ctrl.get_mpcc_metrics()
        print('\nMPCC-SPECIFIC METRICS:')
        print(f"  Avg Contouring Error: {mpcc_metrics.get('avg_contour_error', 'N/A'):.4f} m")
        print(f"  Avg Lag Error: {mpcc_metrics.get('avg_lag_error', 'N/A'):.4f} m")
        print(f"  Max Contouring Error: {mpcc_metrics.get('max_contour_error', 'N/A'):.4f} m")
        print(f"  Max Lag Error: {mpcc_metrics.get('max_lag_error', 'N/A'):.4f} m")
        print(f"  RMS Contouring Error: {mpcc_metrics.get('rms_contour_error', 'N/A'):.4f} m")
        print(f"  RMS Lag Error: {mpcc_metrics.get('rms_lag_error', 'N/A'):.4f} m")
        print(f"  Path Progress: {mpcc_metrics.get('path_progress_pct', 'N/A'):.2f}%")


def post_analysis(state_stack, input_stack, env, ctrl=None, trajs_data=None, episode_idx=0):
    """Plots the input and states to determine MPCC's success.

    Args:
        state_stack (ndarray): The list of observations of MPCC in the latest run.
        input_stack (ndarray): The list of inputs of MPCC in the latest run.
        env: The environment.
        ctrl: The MPCC controller (optional, for additional plotting).
        trajs_data (dict): Full trajectory data dictionary (optional, for Excel export).
        episode_idx (int): Episode index (optional, for Excel filename).
    """
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Create figure with more subplots for MPCC-specific info
    fig = plt.figure(figsize=(15, 10))
    
    # ===== State Trajectories =====
    ax_states = fig.add_subplot(2, 2, 1)
    for k in range(model.nx):
        ax_states.plot(times, np.array(state_stack)[:plot_length, k], 
                      label=f'{env.STATE_LABELS[k]}')
    ax_states.set_title('State Trajectories')
    ax_states.set_xlabel('time (sec)')
    ax_states.set_ylabel('states')
    ax_states.legend()
    ax_states.grid(True)

    # ===== Input Trajectories =====
    ax_inputs = fig.add_subplot(2, 2, 2)
    for k in range(model.nu):
        ax_inputs.plot(times, np.array(input_stack)[:plot_length, k],
                      label=f'{env.ACTION_LABELS[k]}')
    ax_inputs.set_title('Input Trajectories')
    ax_inputs.set_xlabel('time (sec)')
    ax_inputs.set_ylabel('inputs')
    ax_inputs.legend()
    ax_inputs.grid(True)

    # ===== MPCC-specific: Path Following (XY plot) =====
    ax_path = fig.add_subplot(2, 2, 3)
    
    # Get x and y indices safely
    # For 2D quadrotor: STATE_LABELS = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
    # MPCC uses x-y plane where drone z maps to MPCC y
    try:
        if hasattr(env, 'STATE_LABELS'):
            labels = env.STATE_LABELS
            x_idx = labels.index('x') if 'x' in labels else 0
            # For 2D quadrotor, 'z' in env = 'y' in MPCC plot
            if 'z' in labels:
                y_idx = labels.index('z')  # 2D quadrotor: z becomes y in MPCC
            elif 'y' in labels:
                y_idx = labels.index('y')  # 3D quadrotor: use y directly
            else:
                y_idx = 2  # Fallback
        else:
            x_idx, y_idx = 0, 2  # Default: x=0, z=2 for 2D quadrotor
    except (AttributeError, ValueError):
        # Fallback to hardcoded indices (2D Quadrotor: x=0, z=2)
        x_idx, y_idx = 0, 2
    
    # Plot actual trajectory
    actual_x = np.array(state_stack)[:plot_length, x_idx]
    actual_y = np.array(state_stack)[:plot_length, y_idx]
    ax_path.plot(actual_x, actual_y, 'b-', linewidth=2, label='Actual Path')
    
    # Plot environment trajectory (original circle from env.X_GOAL)
    if hasattr(env, 'X_GOAL') and env.X_GOAL is not None:
        env_traj = env.X_GOAL
        # Extract x, y positions from environment trajectory
        if hasattr(env, 'STATE_LABELS'):
            labels = env.STATE_LABELS
            try:
                x_idx = labels.index('x')
                if 'z' in labels:
                    y_idx = labels.index('z')  # 2D quadrotor: z becomes y in MPCC
                elif 'y' in labels:
                    y_idx = labels.index('y')
                else:
                    y_idx = x_idx + 2  # Fallback
                env_x = env_traj[:, x_idx]
                env_y = env_traj[:, y_idx]
                ax_path.plot(env_x, env_y, 'm:', linewidth=1.5, alpha=0.6, label='Environment Trajectory (env.X_GOAL)')
            except (ValueError, IndexError):
                pass
        elif env_traj.shape[1] >= 3:
            # Fallback: assume first and third columns are x and z
            env_x = env_traj[:, 0]
            env_y = env_traj[:, 2]
            ax_path.plot(env_x, env_y, 'm:', linewidth=1.5, alpha=0.6, label='Environment Trajectory (env.X_GOAL)')
    
    # Plot MPCC waypoints if available
    if hasattr(ctrl, 'waypoints') and ctrl.waypoints is not None:
        waypoints = ctrl.waypoints
        if waypoints.ndim == 2 and waypoints.shape[1] >= 2:
            ax_path.plot(waypoints[:, 0], waypoints[:, 1], 'go', 
                        markersize=6, alpha=0.7, label='MPCC Waypoints', zorder=5)
            # Highlight first and last waypoints
            ax_path.plot(waypoints[0, 0], waypoints[0, 1], 'g*', 
                        markersize=15, label='First/Last Waypoint', zorder=6)
    
    # Plot MPCC reference path (from lookup table - Bezier interpolated)
    if hasattr(ctrl, 'reference_path') and ctrl.reference_path is not None:
        ref_path = ctrl.reference_path
        # Verify shape before plotting
        if ref_path.ndim == 2 and ref_path.shape[1] >= 2:
            ax_path.plot(ref_path[:, 0], ref_path[:, 1], 'r--', 
                        linewidth=2, label='MPCC Reference Path (Bezier)', zorder=4)
        else:
            print(f"[Warning] reference_path has unexpected shape: {ref_path.shape}")
    
    ax_path.set_title('Path Following (XY Plane)')
    ax_path.set_xlabel('X position [m]')
    ax_path.set_ylabel('Y position [m]')
    ax_path.legend()
    ax_path.grid(True)
    ax_path.axis('equal')

    # ===== MPCC-specific: Contouring and Lag Errors =====
    ax_errors = fig.add_subplot(2, 2, 4)
    
    # Extract contouring and lag errors if available
    # Also check _last_error_history as backup (in case current episode failed)
    error_history = None
    if hasattr(ctrl, 'error_history') and ctrl.error_history:
        error_history = ctrl.error_history
    elif hasattr(ctrl, '_last_error_history') and ctrl._last_error_history:
        error_history = ctrl._last_error_history
        print("[Plot] Using backup error history from previous episode")
    
    if error_history:
        # Use actual simulation time based on state_stack length and stepsize
        # error_history should have same length as state_stack (one entry per control step)
        error_history_length = len(error_history)
        actual_length = min(error_history_length, plot_length)
        
        # Calculate time array matching actual simulation steps
        # Use same time base as state/input plots for consistency
        error_times = np.linspace(0, stepsize * actual_length, actual_length)
        
        # Trim error_history to match actual_length if needed
        contour_errors = [e['contour'] for e in error_history[:actual_length]]
        lag_errors = [e['lag'] for e in error_history[:actual_length]]
        
        ax_errors.plot(error_times, contour_errors, 'r-', label='Contouring Error')
        ax_errors.plot(error_times, lag_errors, 'b-', label='Lag Error')
        ax_errors.set_title('MPCC Tracking Errors')
        ax_errors.set_xlabel('time (sec)')
        ax_errors.set_ylabel('error [m]')
        ax_errors.legend()
        ax_errors.grid(True)
    else:
        ax_errors.text(0.5, 0.5, 'Error history not available', 
                      ha='center', va='center', transform=ax_errors.transAxes)
        ax_errors.set_title('MPCC Tracking Errors')

    plt.tight_layout()
    plt.show()
    
    # ═══════════════════════════════════════════════════════════
    # Export data to Excel files
    # ═══════════════════════════════════════════════════════════
    if trajs_data is not None:
        export_to_excel(state_stack, input_stack, env, ctrl, trajs_data, episode_idx)


def export_to_excel(state_stack, input_stack, env, ctrl, trajs_data, episode_idx=0):
    """Export trajectory data and optimized theta to Excel files.
    
    Args:
        state_stack (ndarray): State observations.
        input_stack (ndarray): Input actions.
        env: The environment.
        ctrl: The MPCC controller.
        trajs_data (dict): Full trajectory data dictionary.
        episode_idx (int): Episode index for filename.
    """
    # Create output directory
    output_dir = './excel_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'mpcc_episode_{episode_idx}_{timestamp}'
    
    model = env.symbolic
    stepsize = model.dt
    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)
    
    # ═══════════════════════════════════════════════════════════
    # 1. Export trajectory data (states, inputs, etc.)
    # ═══════════════════════════════════════════════════════════
    traj_df_data = {
        'time': times[:plot_length]
    }
    
    # Add state columns
    if hasattr(env, 'STATE_LABELS'):
        state_labels = env.STATE_LABELS
    else:
        state_labels = [f'state_{i}' for i in range(model.nx)]
    
    state_array = np.array(state_stack)[:plot_length, :]
    for k in range(model.nx):
        traj_df_data[state_labels[k]] = state_array[:, k]
    
    # Add input columns
    if hasattr(env, 'ACTION_LABELS'):
        action_labels = env.ACTION_LABELS
    else:
        action_labels = [f'action_{i}' for i in range(model.nu)]
    
    input_array = np.array(input_stack)[:plot_length, :]
    for k in range(model.nu):
        traj_df_data[action_labels[k]] = input_array[:, k]
    
    # Add reference trajectory if available
    if hasattr(env, 'X_GOAL') and env.X_GOAL is not None:
        reference = env.X_GOAL
        if env.TASK == Task.STABILIZATION:
            reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))
        
        ref_length = min(len(reference), plot_length)
        for k in range(model.nx):
            if ref_length > 0:
                ref_col = np.full(plot_length, np.nan)
                ref_col[:ref_length] = reference[:ref_length, k]
                traj_df_data[f'ref_{state_labels[k]}'] = ref_col
            else:
                traj_df_data[f'ref_{state_labels[k]}'] = np.full(plot_length, np.nan)
    
    # Add additional data from trajs_data if available
    if 'state' in trajs_data and len(trajs_data['state']) > episode_idx:
        state_data = trajs_data['state'][episode_idx]
        if len(state_data) >= plot_length:
            for k in range(min(model.nx, state_data.shape[1])):
                traj_df_data[f'state_data_{state_labels[k]}'] = state_data[:plot_length, k]
    
    if 'reward' in trajs_data and len(trajs_data['reward']) > episode_idx:
        reward_data = trajs_data['reward'][episode_idx]
        if len(reward_data) >= plot_length:
            traj_df_data['reward'] = reward_data[:plot_length]
    
    if 'timestamp' in trajs_data and len(trajs_data['timestamp']) > episode_idx:
        timestamp_data = trajs_data['timestamp'][episode_idx]
        if len(timestamp_data) >= plot_length:
            traj_df_data['timestamp'] = timestamp_data[:plot_length]
    
    traj_df = pd.DataFrame(traj_df_data)
    traj_excel_path = os.path.join(output_dir, f'{base_filename}_trajectory.xlsx')
    traj_df.to_excel(traj_excel_path, index=False, engine='openpyxl')
    print(f"[Excel Export] Trajectory data saved to: {traj_excel_path}")
    
    # ═══════════════════════════════════════════════════════════
    # 2. Export optimized theta (path progress) data
    # ═══════════════════════════════════════════════════════════
    error_history = None
    if hasattr(ctrl, 'error_history') and ctrl.error_history:
        error_history = ctrl.error_history
    elif hasattr(ctrl, '_last_error_history') and ctrl._last_error_history:
        error_history = ctrl._last_error_history
    
    if error_history:
        error_history_length = len(error_history)
        actual_length = min(error_history_length, plot_length)
        error_times = np.linspace(0, stepsize * actual_length, actual_length)
        
        theta_df_data = {
            'time': error_times[:actual_length],
            'theta': [e.get('theta', 0) for e in error_history[:actual_length]],
            'v_theta': [e.get('v_theta', 0) for e in error_history[:actual_length]],
            'contour_error': [e.get('contour', 0) for e in error_history[:actual_length]],
            'lag_error': [e.get('lag', 0) for e in error_history[:actual_length]]
        }
        
        # Add theta cycles if s_max is available
        if hasattr(ctrl, 'track_manager') and hasattr(ctrl.track_manager, 's_max'):
            s_max = ctrl.track_manager.s_max
            theta_df_data['theta_cycles'] = [theta / s_max for theta in theta_df_data['theta']]
            theta_df_data['s_max'] = [s_max] * actual_length
        
        theta_df = pd.DataFrame(theta_df_data)
        theta_excel_path = os.path.join(output_dir, f'{base_filename}_theta.xlsx')
        theta_df.to_excel(theta_excel_path, index=False, engine='openpyxl')
        print(f"[Excel Export] Theta data saved to: {theta_excel_path}")
    else:
        print("[Excel Export] Warning: error_history not available, skipping theta export")


def post_analysis_detailed(state_stack, input_stack, env):
    """Detailed plotting similar to original MPCC python_sim.py.
    
    Args:
        state_stack (ndarray): The list of observations.
        input_stack (ndarray): The list of inputs.
        env: The environment.
    """
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states (detailed version)
    fig, axs = plt.subplots(model.nx)
    if model.nx == 1:
        axs = [axs]
        
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if k != model.nx - 1:
            axs[k].set_xticks([])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')

    # Plot inputs (detailed version)
    fig2, axs2 = plt.subplots(model.nu)
    if model.nu == 1:
        axs2 = [axs2]
        
    for k in range(model.nu):
        axs2[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        axs2[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs2[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs2[0].set_title('Input Trajectories')
    axs2[-1].set(xlabel='time (sec)')

    plt.show()


if __name__ == '__main__':
    run(gui = True, plot = True)
