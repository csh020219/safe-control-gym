'''An MPC and Linear MPC example.'''

import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def calculate_cycle_duration(trajs_data, env, algo_name=''):
    """Calculate actual cycle duration by finding all x=0 crossings and averaging intervals.
    
    Args:
        trajs_data: Trajectory data from experiment
        env: Environment object
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
    
    # Get trajectory plane coordinates
    if 'state' not in trajs_data or len(trajs_data['state']) == 0:
        return None
    
    states = trajs_data['state'][0]  # First episode
    timestamps = trajs_data.get('timestamp', [[]])[0] if 'timestamp' in trajs_data else None
    
    if len(states) < 10:  # Need minimum steps
        return None
    
    # Get trajectory plane (e.g., 'xz' for 2D quadrotor)
    traj_plane = env.TASK_INFO.get('trajectory_plane', 'xz')
    
    # Find position indices from STATE_LABELS
    # State: [x, x_dot, z, z_dot, theta, theta_dot] for 2D quadrotor
    state_labels = env.STATE_LABELS if hasattr(env, 'STATE_LABELS') else []
    x_idx = state_labels.index(traj_plane[0]) if traj_plane[0] in state_labels else 0
    
    # Use actual timestamps or step count to calculate real cycle duration
    if timestamps is not None and len(timestamps) > 1:
        # Use actual timestamps if available
        total_duration = timestamps[-1] - timestamps[0]
        actual_cycle_duration = total_duration / num_cycles
        num_detected_cycles = num_cycles
    else:
        # Fallback: use step count and control frequency
        total_steps = len(states)
        total_duration = total_steps / env.CTRL_FREQ
        actual_cycle_duration = total_duration / num_cycles
        num_detected_cycles = num_cycles
    
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
    '''The main function running MPC and Linear MPC experiments.

    Args:
        gui (bool): Whether to display the gui.
        plot (bool): Whether to plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
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

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )

    # Run the experiment.
    experiment = BaseExperiment(env=env, ctrl=ctrl)
    trajs_data, metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)

    # Calculate and display cycle duration
    calculate_cycle_duration(trajs_data, env, algo_name='MPC')

    if plot:
        for i in range(len(trajs_data['obs'])):
            post_analysis(trajs_data['obs'][i], trajs_data['action'][i], ctrl.env)

    ctrl.close()
    env.close()

    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        with open(f'./temp-data/{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))


def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to determine MPC's success.

    Args:
        state_stack (ndarray): The list of observations of MPC in the latest run.
        input_stack (ndarray): The list of inputs of MPC in the latest run.
    '''
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))

    # Plot states
    fig, axs = plt.subplots(model.nx)
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

    # Plot inputs
    _, axs = plt.subplots(model.nu)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        axs[k].set(ylabel=f'input {k}')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')

    plt.show()


if __name__ == '__main__':
    run(gui=True)
