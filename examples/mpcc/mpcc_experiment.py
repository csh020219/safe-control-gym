"""An MPCC (Model Predictive Contouring Control) example."""

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

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    env = env_func(gui=gui)

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

    if plot:
        for i in range(len(trajs_data['obs'])):
            post_analysis(
                trajs_data['obs'][i], 
                trajs_data['action'][i], 
                ctrl.env,
                ctrl  # MPCC controller 전달
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
        print(f"  Avg Contouring Error: {mpcc_metrics.get('avg_contour_error', 'N/A'):.4f}")
        print(f"  Avg Lag Error: {mpcc_metrics.get('avg_lag_error', 'N/A'):.4f}")
        print(f"  Path Progress: {mpcc_metrics.get('path_progress', 'N/A'):.2f}%")


def post_analysis(state_stack, input_stack, env, ctrl=None):
    """Plots the input and states to determine MPCC's success.

    Args:
        state_stack (ndarray): The list of observations of MPCC in the latest run.
        input_stack (ndarray): The list of inputs of MPCC in the latest run.
        env: The environment.
        ctrl: The MPCC controller (optional, for additional plotting).
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
    # Plot actual trajectory
    actual_x = np.array(state_stack)[:plot_length, 0]  # Assuming x is first state
    actual_y = np.array(state_stack)[:plot_length, 1]  # Assuming y is second state
    ax_path.plot(actual_x, actual_y, 'b-', linewidth=2, label='Actual Path')
    
    # Plot reference path if available
    if hasattr(ctrl, 'reference_path') and ctrl.reference_path is not None:
        ref_path = ctrl.reference_path
        ax_path.plot(ref_path[:, 0], ref_path[:, 1], 'r--', 
                    linewidth=1, label='Reference Path')
    
    ax_path.set_title('Path Following (XY Plane)')
    ax_path.set_xlabel('X position [m]')
    ax_path.set_ylabel('Y position [m]')
    ax_path.legend()
    ax_path.grid(True)
    ax_path.axis('equal')

    # ===== MPCC-specific: Contouring and Lag Errors =====
    ax_errors = fig.add_subplot(2, 2, 4)
    
    # Extract contouring and lag errors if available
    if hasattr(ctrl, 'error_history'):
        error_history = ctrl.error_history
        error_times = np.linspace(0, stepsize * len(error_history), len(error_history))
        
        contour_errors = [e['contour'] for e in error_history]
        lag_errors = [e['lag'] for e in error_history]
        
        ax_errors.plot(error_times, contour_errors, 'r-', label='Contouring Error')
        ax_errors.plot(error_times, lag_errors, 'b-', label='Lag Error')
        ax_errors.set_title('MPCC Tracking Errors')
        ax_errors.set_xlabel('time (sec)')
        ax_errors.set_ylabel('error')
        ax_errors.legend()
        ax_errors.grid(True)
    else:
        ax_errors.text(0.5, 0.5, 'Error history not available', 
                      ha='center', va='center', transform=ax_errors.transAxes)
        ax_errors.set_title('MPCC Tracking Errors')

    plt.tight_layout()
    plt.show()


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
    run(gui=True, plot=True)
