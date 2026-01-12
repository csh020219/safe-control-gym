"""2D Model Predictive Contouring Control (MPCC) using ACADOS.

Track lookup table is automatically generated from waypoints.
Based on: [https://github.com/mlab-upenn/mpcc](https://github.com/mlab-upenn/mpcc)
"""

import os
import numpy as np
import casadi as ca

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.mpcc.mpcc_utils import (
    TrackManager,
    create_simple_oval,
    create_figure_eight,
    create_circle,
    env_trajectory_to_waypoints
)


class MPCC(BaseController):
    """2D Model Predictive Contouring Control for path following."""
    
    def __init__(
            self,
            env_func,
            # Track configuration
            track_file=None,
            waypoints=None,
            track_name='circle',
            track_scale=10,
            track_width=0.5,
            track_density=100,
            track_closed=True,
            track_cache_dir='tracks',
            fixed_altitude=1.0,
            use_env_trajectory=False,
            env_trajectory_sample_rate=2,
            # MPC parameters
            horizon=20,
            # MPCC cost weights
            q_contour=1.0,
            q_lag=100.0,
            q_vel=None,
            q_yaw=0.1,
            q_yaw_rate=0.01,
            q_v_theta=0.1,
            # Control cost weights
            r_accel=None,
            r_yaw_rate=0.1,
            # Control limits
            accel_limits=None,
            yaw_rate_limits=None,
            v_theta_limits=None,
            # Solver options
            use_terminal_cost=True,
            qp_solver='PARTIAL_CONDENSING_HPIPM',
            integrator_type='ERK',
            nlp_solver_type='SQP',
            nlp_solver_max_iter=20,
            qp_solver_iter_max=50,
            verbose=False,
            **kwargs
    ):
        """Initialize 2D MPCC controller.
        
        Args:
            env_func: Function to instantiate environment.
            track_file: Path to track CSV file (waypoints or lookup table).
            waypoints: Nx2 array of waypoints (alternative to track_file).
            track_name: Built-in track name ('simple_oval', 'figure_eight', 'circle').
            track_scale: Scaling factor for built-in tracks.
            track_width: Half-width of track bounds (m).
            track_density: Points per meter in lookup table.
            track_closed: Whether track is a closed loop.
            track_cache_dir: Directory to cache generated lookup tables.
            fixed_altitude: Fixed altitude for 2D flight (m).
            use_env_trajectory: If True, use env.X_GOAL to generate waypoints.
            env_trajectory_sample_rate: Waypoints per second when using env trajectory.
            horizon: MPC prediction horizon.
            q_contour: Contouring error weight.
            q_lag: Lag error weight.
            q_vel: Velocity weights [vx, vy].
            q_yaw: Yaw angle weight.
            q_yaw_rate: Yaw rate weight.
            q_v_theta: Progress rate weight.
            r_accel: Acceleration control weights [ax, ay].
            r_yaw_rate: Yaw rate control weight.
            accel_limits: Acceleration limits [min, max] (m/s²).
            yaw_rate_limits: Yaw rate limits [min, max] (rad/s).
            v_theta_limits: Progress rate limits [min, max] (m/s).
            use_terminal_cost: Use terminal cost in OCP.
            qp_solver: ACADOS QP solver.
            integrator_type: ACADOS integrator type.
            nlp_solver_type: ACADOS NLP solver type.
            nlp_solver_max_iter: Maximum NLP iterations.
            qp_solver_iter_max: Maximum QP iterations.
            verbose: Print debug information.
        """
        if not ACADOS_AVAILABLE:
            raise ImportError("ACADOS required. Install: https://docs.acados.org/installation/")
        
        super().__init__(env_func, **kwargs)
        self.env = env_func()
        
        # Store all configuration (store verbose early so we can use it)
        self.track_file = track_file
        self.track_name = track_name
        self.track_scale = track_scale
        self.track_width = track_width
        self.track_density = track_density
        self.track_closed = track_closed
        self.track_cache_dir = track_cache_dir
        self.fixed_altitude = fixed_altitude
        self.horizon = horizon
        self.verbose = verbose
        self.env_trajectory_sample_rate = env_trajectory_sample_rate
        
        # Store environment trajectory (similar to MPC pattern)
        self.use_env_trajectory = use_env_trajectory
        if use_env_trajectory:
            from safe_control_gym.envs.benchmark_env import Task
            # Check if env.X_GOAL exists and is valid
            if hasattr(self.env, 'X_GOAL') and self.env.X_GOAL is not None:
                if self.env.TASK == Task.STABILIZATION:
                    self.x_goal = self.env.X_GOAL
                elif self.env.TASK == Task.TRAJ_TRACKING:
                    self.x_goal = self.env.X_GOAL
                    if self.verbose:
                        print(f"[MPCC] env.X_GOAL shape: {self.env.X_GOAL.shape}")
                        print(f"[MPCC] env.TASK: {self.env.TASK}, task_info: {getattr(self.env, 'TASK_INFO', None)}")
                else:
                    self.x_goal = None
                    if self.verbose:
                        print(f"[MPCC] Warning: Unknown task type: {self.env.TASK}")
            else:
                self.x_goal = None
                if self.verbose:
                    print(f"[MPCC] Warning: env.X_GOAL not available or None")
                    print(f"[MPCC] env has X_GOAL attr: {hasattr(self.env, 'X_GOAL')}")
                    if hasattr(self.env, 'X_GOAL'):
                        print(f"[MPCC] env.X_GOAL value: {self.env.X_GOAL}")
        else:
            self.x_goal = None
        
        # Cost weights
        self.q_contour = q_contour
        self.q_lag = q_lag
        self.q_vel = np.array(q_vel if q_vel is not None else [0.1, 0.1])
        self.q_yaw = q_yaw
        self.q_yaw_rate = q_yaw_rate
        self.q_v_theta = q_v_theta
        self.r_accel = np.array(r_accel if r_accel is not None else [0.1, 0.1])
        self.r_yaw_rate = r_yaw_rate
        self.use_terminal_cost = use_terminal_cost
        
        # Control limits
        self.accel_limits = accel_limits if accel_limits is not None else [-3.0, 3.0]
        self.yaw_rate_limits = yaw_rate_limits if yaw_rate_limits is not None else [-1.5, 1.5]
        self.v_theta_limits = v_theta_limits if v_theta_limits is not None else [0.01, 2.0]
        
        # Solver options
        self.qp_solver = qp_solver
        self.integrator_type = integrator_type
        self.nlp_solver_type = nlp_solver_type
        self.nlp_solver_max_iter = nlp_solver_max_iter
        self.qp_solver_iter_max = qp_solver_iter_max
        
        # ═══════════════════════════════════════════════════════════
        # Track setup
        # ═══════════════════════════════════════════════════════════
        waypoints_to_use = self._get_track_waypoints(waypoints)
        
        self.track_manager = TrackManager(
            track_file=track_file if track_file is not None else None,
            waypoints=waypoints_to_use if waypoints_to_use is not None else None,
            track_width=track_width,
            density=track_density,
            closed=track_closed,
            cache_dir=track_cache_dir
        )
        
        # MPCC state (initialized in reset_before_run)
        self.theta = 0.0
        self.error_history = []
        self.reference_path = self.track_manager.reference_path
        
        # ═══════════════════════════════════════════════════════════
        # Setup constraints from environment (same as MPC!)
        # ═══════════════════════════════════════════════════════════
        from safe_control_gym.controllers.mpc.mpc_utils import reset_constraints
        
        if hasattr(self.env, 'constraints') and self.env.constraints is not None:
            self.constraints, self.state_constraints_sym, self.input_constraints_sym = \
                reset_constraints(self.env.constraints.constraints)
            if verbose:
                print(f"[MPCC] Loaded {len(self.constraints.state_constraints)} state constraints, "
                      f"{len(self.constraints.input_constraints)} input constraints from env")
        else:
            self.constraints = None
            self.state_constraints_sym = []
            self.input_constraints_sym = []
            if verbose:
                print("[MPCC] No environment constraints found")
        
        if verbose:
            print(f"[MPCC] Initialized: N={horizon}, altitude={fixed_altitude}m")
            if self.use_env_trajectory:
                print(f"[MPCC] Track: Environment trajectory (env.X_GOAL) - {len(self.track_manager.reference_path)} points")
            else:
                print(f"[MPCC] Track: {track_name if track_file is None else track_file}")
            print(f"[MPCC] Weights: Qc={q_contour}, Ql={q_lag}, Q_theta={q_v_theta}")
    
    def _get_track_waypoints(self, waypoints_arg=None):
        """Get track waypoints based on track_name, track_file, waypoints argument, or env.X_GOAL.
        
        Priority order:
        1. track_file (if specified)
        2. waypoints argument (if provided)
        3. env.X_GOAL (if use_env_trajectory=True)
        4. built-in track_name
        
        Args:
            waypoints_arg: Waypoints passed as constructor argument (or None).
            
        Returns:
            waypoints (ndarray): Nx2 array of (x, y) waypoints, or None if track_file is used.
        """
        # Priority 1: Use track_file if specified
        if self.track_file is not None:
            if self.verbose:
                print(f"[MPCC] Loading track from file: {self.track_file}")
            # TrackManager will handle file loading
            return None
        
        # Priority 2: Use waypoints argument if provided
        if waypoints_arg is not None:
            if self.verbose:
                print(f"[MPCC] Using provided waypoints (shape: {waypoints_arg.shape})")
            return waypoints_arg
        
        # Priority 3: Use env.X_GOAL if use_env_trajectory is True
        if self.use_env_trajectory:
            if self.verbose:
                print(f"[MPCC] use_env_trajectory=True, checking env.X_GOAL...")
                print(f"[MPCC] self.x_goal is None: {self.x_goal is None}")
                if hasattr(self.env, 'X_GOAL'):
                    print(f"[MPCC] env.X_GOAL exists, shape: {self.env.X_GOAL.shape if self.env.X_GOAL is not None else 'None'}")
            
            if self.x_goal is not None:
                if self.verbose:
                    print(f"[MPCC] Converting env.X_GOAL to waypoints (sample_rate={self.env_trajectory_sample_rate})")
                try:
                    waypoints = env_trajectory_to_waypoints(
                        self.env, 
                        sample_rate=self.env_trajectory_sample_rate
                    )
                    if self.verbose:
                        print(f"[MPCC] Generated {len(waypoints)} waypoints from env trajectory")
                    return waypoints
                except Exception as e:
                    if self.verbose:
                        print(f"[MPCC] Error: Failed to convert env.X_GOAL: {e}")
                        import traceback
                        traceback.print_exc()
                    print(f"[MPCC] Falling back to track_name: {self.track_name}")
                    # Fall through to Priority 4
            else:
                if self.verbose:
                    print(f"[MPCC] Warning: use_env_trajectory=True but env.X_GOAL is None or not available")
                    print(f"[MPCC] Falling back to track_name: {self.track_name}")
                # Fall through to Priority 4
        
        # Priority 4: Generate from built-in track_name
        if self.verbose:
            print(f"[MPCC] Generating built-in track: {self.track_name} (scale={self.track_scale})")
            print(f"[MPCC] Note: This is MPCC's built-in track generator, NOT env.X_GOAL")
        
        if self.track_name == 'simple_oval':
            waypoints = create_simple_oval(scale=self.track_scale)
        elif self.track_name == 'figure_eight':
            waypoints = create_figure_eight(radius=self.track_scale)
        elif self.track_name == 'circle_builtin':
            waypoints = create_circle(radius=self.track_scale)
            if self.verbose:
                print(f"[MPCC] Using MPCC built-in circle (radius={self.track_scale})")
        else:
            raise ValueError(
                f"Unknown track_name: '{self.track_name}'. "
                f"Options: 'simple_oval', 'figure_eight', 'circle'. "
                f"To use env.X_GOAL, set use_env_trajectory=True"
            )
        
        return waypoints
    
    def setup_acados_model(self):
        """Create ACADOS model for 2D MPCC using ENVIRONMENT SYMBOLIC MODEL.
        
        Uses env.symbolic.fc_func directly (same as MPC_ACADOS) to ensure
        perfect consistency with environment dynamics!
        
        MPCC extends the environment model with path progress variable:
            State:  env_state + [theta_path] = 7D
            Input:  env_action + [v_theta] = 3D
            
        Frame mapping for MPCC cost:
            MPCC x = drone x (horizontal)
            MPCC y = drone z (vertical)
        
        Returns:
            model (AcadosModel): ACADOS model object.
        """
        model_name = 'quadrotor_mpcc_2d'
        
        # ═══════════════════════════════════════════════════════════
        # Get environment's symbolic model (same as MPC_ACADOS!)
        # ═══════════════════════════════════════════════════════════
        env_model = self.env.symbolic
        env_x_sym = env_model.x_sym      # 6D: [x, x_dot, z, z_dot, pitch, pitch_dot]
        env_u_sym = env_model.u_sym      # 2D: [T1, T2]
        env_fc_func = env_model.fc_func  # dynamics: x_dot = f(x, u)
        
        # Get physical parameters for cost function
        m = self.env.MASS
        g = self.env.GRAVITY_ACC
        
        # ═══════════════════════════════════════════════════════════
        # MPCC State (7D) = Environment state (6D) + theta_path
        # [x, x_dot, z, z_dot, pitch, pitch_dot, theta_path]
        # NOTE: Use MX type to match environment's symbolic model!
        # ═══════════════════════════════════════════════════════════
        theta_path = ca.MX.sym('theta_path')  # Path progress variable (MX!)
        state_mpcc = ca.vertcat(env_x_sym, theta_path)
        
        # Extract individual states for cost function
        x = env_x_sym[0]           # Horizontal position
        x_dot = env_x_sym[1]       # Horizontal velocity
        z = env_x_sym[2]           # Vertical position (altitude)
        z_dot = env_x_sym[3]       # Vertical velocity
        pitch = env_x_sym[4]       # Pitch angle
        pitch_dot = env_x_sym[5]   # Pitch rate
        
        # ═══════════════════════════════════════════════════════════
        # MPCC Control (3D) = Environment action (2D) + v_theta
        # [T1, T2, v_theta]
        # NOTE: Use MX type to match environment's symbolic model!
        # ═══════════════════════════════════════════════════════════
        v_theta = ca.MX.sym('v_theta')  # Path progress rate (MX!)
        u_mpcc = ca.vertcat(env_u_sym, v_theta)
        
        # Extract thrust for cost function
        T1 = env_u_sym[0]
        T2 = env_u_sym[1]
        
        # ═══════════════════════════════════════════════════════════
        # MPCC Dynamics = Environment dynamics + theta_path dynamics
        # Uses env.symbolic.fc_func DIRECTLY for consistency!
        # ═══════════════════════════════════════════════════════════
        env_dynamics = env_fc_func(env_x_sym, env_u_sym)  # 6D: x_dot from env
        theta_path_dot = v_theta
        
        f_expl_mpcc = ca.vertcat(env_dynamics, theta_path_dot)  # 7D
        
        # ═══════════════════════════════════════════════════════════
        # Implicit formulation (MX type!)
        # ═══════════════════════════════════════════════════════════
        x_dot_sym = ca.MX.sym('x_dot', 7)  # MX!
        f_impl = x_dot_sym - f_expl_mpcc
        
        # ═══════════════════════════════════════════════════════════
        # Parameters (track reference) - MX type to match state/control!
        # ═══════════════════════════════════════════════════════════
        x_ref = ca.MX.sym('x_ref')       # Reference x position
        y_ref = ca.MX.sym('y_ref')       # Reference y position (= drone z)
        cos_phi_ref = ca.MX.sym('cos_phi_ref')  # cos(path tangent angle)
        sin_phi_ref = ca.MX.sym('sin_phi_ref')  # sin(path tangent angle)
        p = ca.vertcat(x_ref, y_ref, cos_phi_ref, sin_phi_ref)
        
        # ═══════════════════════════════════════════════════════════
        # MPCC Cost (EXTERNAL type)
        # Frame mapping: MPCC x = drone x, MPCC y = drone z
        # ═══════════════════════════════════════════════════════════
        # Position in MPCC frame
        x_mpcc = x      # Horizontal = MPCC x
        y_mpcc = z      # Vertical (drone z) = MPCC y
        
        # Contouring error (perpendicular to path)
        e_c = -(x_mpcc - x_ref) * sin_phi_ref + (y_mpcc - y_ref) * cos_phi_ref
        
        # Lag error (tangential to path)
        e_l = (x_mpcc - x_ref) * cos_phi_ref + (y_mpcc - y_ref) * sin_phi_ref
        
        # Hover thrust for reference
        T_hover = m * g / 2.0  # Each motor provides half of hover thrust
        
        # ═══════════════════════════════════════════════════════════
        # Traditional MPCC Stage Cost (pure MPCC formulation)
        # J = Qc * e_c² + Ql * e_l² - Q_theta * v_theta + R * u²
        # Note: Negative sign before Q_theta*v_theta MAXIMIZES progress rate
        # (cost minimization = -Q*v minimization = v maximization)
        # ═══════════════════════════════════════════════════════════
        stage_cost = (
            self.q_contour * e_c**2 +           # Contouring error penalty
            self.q_lag * e_l**2 -               # Lag error penalty  
            self.q_v_theta * v_theta +          # Progress rate MAXIMIZATION (negative sign in formula)
            self.r_accel[0] * (T1 - T_hover)**2 +  # Control effort (T1)
            self.r_accel[1] * (T2 - T_hover)**2    # Control effort (T2)
        )
        
        # Terminal cost (path errors only)
        terminal_cost = (
            self.q_contour * 2 * e_c**2 +       # Terminal contouring error
            self.q_lag * 2 * e_l**2             # Terminal lag error
        )
        
        # ═══════════════════════════════════════════════════════════
        # Create ACADOS Model
        # ═══════════════════════════════════════════════════════════
        model = AcadosModel()
        model.name = model_name
        model.x = state_mpcc
        model.xdot = x_dot_sym
        model.u = u_mpcc
        model.p = p
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl_mpcc
        model.cost_expr_ext_cost = stage_cost
        model.cost_expr_ext_cost_e = terminal_cost
        
        return model

    
    def setup_optimizer(self):
        """Setup ACADOS OCP solver."""
        model = self.setup_acados_model()
        
        nx = 7  # [x, x_dot, z, z_dot, pitch, pitch_dot, theta_path]
        nu = 3  # [T1, T2, v_theta]
        
        ocp = AcadosOcp()
        ocp.model = model
        
        # ═══════════════════════════════════════════════════════════
        # Traditional MPCC: EXTERNAL cost type
        # ═══════════════════════════════════════════════════════════
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        
        # ═══════════════════════════════════════════════════════════
        # Control Constraints: [T1, T2, v_theta]
        # Get thrust limits from environment's physical action bounds
        # ═══════════════════════════════════════════════════════════
        T_min, T_max = self.env.physical_action_bounds
        T_min = float(T_min[0])  # Min thrust per "motor group"
        T_max = float(T_max[0])  # Max thrust per "motor group"
        
        if self.verbose:
            print(f"[MPCC] Thrust bounds: T_min={T_min:.4f}, T_max={T_max:.4f}")
        
        ocp.constraints.lbu = np.array([T_min, T_min, self.v_theta_limits[0]])
        ocp.constraints.ubu = np.array([T_max, T_max, self.v_theta_limits[1]])
        ocp.constraints.idxbu = np.array([0, 1, 2])
        
        # ═══════════════════════════════════════════════════════════
        # State Constraints (same as MPC - from environment!)
        # State: [x, x_dot, z, z_dot, pitch, pitch_dot, theta_path]
        # ═══════════════════════════════════════════════════════════
        # Try to use environment constraints (like MPC does)
        from safe_control_gym.envs.constraints import BoundedConstraint
        
        env_nx = 6  # Environment state dimension
        state_constraints_applied = False
        
        if hasattr(self, 'constraints') and self.constraints is not None:
            for state_constraint in self.constraints.state_constraints:
                if isinstance(state_constraint, BoundedConstraint):
                    # Extend environment constraints (6D) to MPCC state (7D)
                    # Add theta_path bounds: [0, +inf] (path progress is non-negative)
                    env_lbx = state_constraint.lower_bounds
                    env_ubx = state_constraint.upper_bounds
                    
                    # MPCC state bounds = env bounds + theta_path bounds
                    lbx = np.append(env_lbx, 0.0)      # theta_path >= 0
                    ubx = np.append(env_ubx, 1e6)     # theta_path < inf (large number)
                    
                    ocp.constraints.lbx = lbx
                    ocp.constraints.ubx = ubx
                    ocp.constraints.idxbx = np.arange(nx)  # All 7 states
                    
                    # Terminal state constraints
                    ocp.constraints.lbx_e = lbx
                    ocp.constraints.ubx_e = ubx
                    ocp.constraints.idxbx_e = np.arange(nx)
                    
                    state_constraints_applied = True
                    if self.verbose:
                        print(f"[MPCC] Using environment state constraints: lbx={lbx}, ubx={ubx}")
                    break
        
        # Fallback: manual pitch/pitch_rate constraints if no env constraints
        if not state_constraints_applied:
            pitch_max = 0.5  # ~30 degrees max pitch
            pitch_rate_max = 3.0  # rad/s max pitch rate
            
            ocp.constraints.lbx = np.array([-pitch_max, -pitch_rate_max])
            ocp.constraints.ubx = np.array([pitch_max, pitch_rate_max])
            ocp.constraints.idxbx = np.array([4, 5])  # pitch, pitch_dot indices
            
            if self.verbose:
                print(f"[MPCC] Using fallback pitch constraints: ±{pitch_max} rad")
        
        # Initial state constraint
        ocp.constraints.x0 = np.zeros(nx)
        
        # Parameter initialization
        ocp.parameter_values = np.zeros(4)
        
        # ═══════════════════════════════════════════════════════════
        # Solver options (tuned for stability)
        # ═══════════════════════════════════════════════════════════
        ocp.solver_options.N_horizon = self.horizon
        ocp.solver_options.tf = self.horizon * self.env.CTRL_TIMESTEP
        
        # QP solver settings
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver_iter_max = 100  # Increased from 50
        ocp.solver_options.qp_solver_cond_N = self.horizon // 2
        
        # NLP solver settings
        ocp.solver_options.nlp_solver_type = 'SQP'  # Full SQP for stability
        ocp.solver_options.nlp_solver_max_iter = 50  # Increased
        
        # Hessian approximation - use GAUSS_NEWTON with regularization for stability
        # (EXACT can be unstable for highly nonlinear problems)
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.regularize_method = 'CONVEXIFY'
        ocp.solver_options.levenberg_marquardt = 1e-2  # Regularization for stability
        
        # Integration
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4  # RK4
        ocp.solver_options.sim_method_num_steps = 3   # Multiple integration steps
        
        # Tolerances
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.qp_tol = 1e-4
        
        ocp.code_export_directory = self.output_dir + '/mpcc_c_generated_code'
        
        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_mpcc.json')
        
        # Store bounds for reference
        self.T_min = T_min
        self.T_max = T_max
        self.T_hover = self.env.MASS * self.env.GRAVITY_ACC / 2.0
        
        if self.verbose:
            print(f"[MPCC] ACADOS solver created: nx={nx}, nu={nu}, N={self.horizon}")
            print(f"[MPCC] T_hover={self.T_hover:.4f} N")



    
    def reset(self):
        """Prepares MPCC for a new experiment (called once before evaluation).
        
        This is called by BaseExperiment.reset() at the start of an experiment.
        DO NOT initialize theta or error_history here - they need obs from reset_before_run().
        """
        if self.verbose:
            print("[MPCC] Resetting controller (experiment-level)")

        
        # Setup or reset ACADOS solver
        if not hasattr(self, 'acados_ocp_solver'):
            if self.verbose:
                print("[MPCC] Creating ACADOS solver...")
            self.setup_optimizer()
        else:
            self.acados_ocp_solver.reset()
            if self.verbose:
                print("[MPCC] ACADOS solver reset")
    
    def reset_before_run(self, obs=None, info=None, env=None):
        """Reinitialize before each episode (called every episode).
        
        This is called by BaseExperiment._evaluation_reset() at the start of each episode.
        """
        super().reset_before_run(obs, info, env)
        
        if self.verbose:
            print("[MPCC] Resetting before episode")
        
        # Episode-level state reset
        self.theta = 0.0
        
        # Save previous error_history if not empty (for plotting)
        # Only clear if this is the first episode or previous had data
        if hasattr(self, 'error_history') and self.error_history:
            self._last_error_history = self.error_history.copy()
        self.error_history = []
        
        # Initialize theta if obs is provided
        if obs is not None and hasattr(self, 'acados_ocp_solver'):
            self._initialize_theta(obs)
        elif self.verbose and obs is None:
            print("[MPCC] Warning: No obs provided, skipping theta initialization")
    
    def _obs_to_mpcc_state(self, obs, theta_path):
        """Convert environment observation to MPCC state.
        
        MPCC State: [x, x_dot, z, z_dot, pitch, pitch_dot, theta_path] (7D)
        
        2D Quadrotor obs: [x, x_dot, z, z_dot, theta, theta_dot] (6D)
        (Note: 'theta' in env is pitch angle, 'theta_path' in MPCC is path progress)
        
        Args:
            obs: Environment observation (6D for 2D quadrotor)
            theta_path: Current path progress variable
            
        Returns:
            x0: MPCC state vector (7D)
        """
        obs = np.atleast_1d(obs)
        
        if len(obs) == 6:
            # 2D Quadrotor: [x, x_dot, z, z_dot, pitch, pitch_dot]
            x0 = np.array([
                obs[0],      # x
                obs[1],      # x_dot
                obs[2],      # z
                obs[3],      # z_dot
                obs[4],      # pitch
                obs[5],      # pitch_dot
                theta_path   # theta_path (MPCC path progress)
            ])
        else:
            raise ValueError(f"Unsupported obs dimension: {len(obs)}. Expected 6 for 2D quadrotor.")
        
        return x0
    
    def _initialize_theta(self, obs):
        """Initialize theta and solver by solving OCP multiple times.
        
        This properly initializes:
        1. The path progress variable (theta_path)
        2. The solver's internal state/control trajectories for warm-starting
        
        Args:
            obs (ndarray): Initial observation from environment
        """
        x0 = self._obs_to_mpcc_state(obs, self.theta)
        
        n_init_iter = 5  # Number of initialization iterations
        dt = self.env.CTRL_TIMESTEP
        v_theta_est = (self.v_theta_limits[0] + self.v_theta_limits[1]) * 0.5
        
        # Hover thrust (each motor provides half of weight)
        T_hover = self.env.MASS * self.env.GRAVITY_ACC / 2.0
        
        # ═══════════════════════════════════════════════════════════
        # Initialize solver trajectories with reasonable guesses
        # This is critical for the first episode!
        # ═══════════════════════════════════════════════════════════
        for i in range(self.horizon + 1):
            # Initial state guess: propagate from x0 with estimated velocity
            x_init = x0.copy()
            x_init[6] = self.theta + i * v_theta_est * dt  # theta_path increases along horizon
            self.acados_ocp_solver.set(i, 'x', x_init)
            
            # Initial control guess: hover thrust + progress rate
            # u = [T1, T2, v_theta]
            if i < self.horizon:
                u_init = np.array([T_hover, T_hover, v_theta_est])
                self.acados_ocp_solver.set(i, 'u', u_init)
        
        # ═══════════════════════════════════════════════════════════
        # Iteratively solve to find good theta and trajectories
        # ═══════════════════════════════════════════════════════════
        for iter_idx in range(n_init_iter):
            # Fix initial state
            self.acados_ocp_solver.set(0, 'lbx', x0)
            self.acados_ocp_solver.set(0, 'ubx', x0)
            
            # Set track parameters for horizon
            for i in range(self.horizon + 1):
                theta_pred = self.theta + i * v_theta_est * dt
                track_params = self.track_manager.get_track_params(theta_pred)
                self.acados_ocp_solver.set(i, 'p', track_params)
            
            # Solve OCP
            status = self.acados_ocp_solver.solve()
            
            if status not in [0, 2]:
                if self.verbose:
                    print(f"[MPCC] Warning: Init solve failed at iter {iter_idx}, status={status}")
                break
            
            # Extract next theta from optimal solution
            x_next = self.acados_ocp_solver.get(1, 'x')
            theta_new = x_next[6]
            
            # Check convergence (skip first iteration - always update)
            theta_diff = abs(theta_new - self.theta)
            self.theta = theta_new
            
            if iter_idx > 0 and theta_diff < 0.01:  # Skip first iteration check
                if self.verbose:
                    print(f"[MPCC] Theta converged after {iter_idx+1} iterations: theta={self.theta:.3f}")
                break
        else:
            # Loop completed without break
            if self.verbose:
                print(f"[MPCC] Theta initialization completed: theta={self.theta:.3f}")
    
    def select_action(self, obs, info=None):
        """Select action by solving MPCC OCP.
        
        Uses ENVIRONMENT DYNAMICS directly, so MPCC output [T1, T2, v_theta]
        can be used as environment action [T1, T2] WITHOUT conversion!
        
        Args:
            obs (ndarray): Current observation from environment
            info (dict): Current info dict
            
        Returns:
            action (ndarray): Action for environment (2D: [T1, T2])
        """
        # Convert obs to MPCC state: [x, x_dot, z, z_dot, pitch, pitch_dot, theta_path]
        x0 = self._obs_to_mpcc_state(obs, self.theta)
        
        # ═══════════════════════════════════════════════════════════
        # Warm-starting: Shift previous solution as initial guess
        # This improves solver convergence by providing a good starting point
        # ═══════════════════════════════════════════════════════════
        # Store warm-started states for track parameter calculation (avoid redundant get calls)
        x_warmstart_list = [None] * (self.horizon + 1)
        
        for i in range(self.horizon):
            # Shift state: x_init[i] = x*[i+1] from previous solution
            x_prev = self.acados_ocp_solver.get(i + 1, 'x')
            self.acados_ocp_solver.set(i, 'x', x_prev)
            x_warmstart_list[i] = x_prev  # Store for track parameter calculation
            
            # Shift control: u_init[i] = u*[i+1] from previous solution
            # For last control, repeat u*[N-1]
            if i < self.horizon - 1:
                u_prev = self.acados_ocp_solver.get(i + 1, 'u')
            else:
                u_prev = self.acados_ocp_solver.get(self.horizon - 1, 'u')
            self.acados_ocp_solver.set(i, 'u', u_prev)
        
        # Terminal state: use last state from previous solution
        x_terminal = self.acados_ocp_solver.get(self.horizon, 'x')
        self.acados_ocp_solver.set(self.horizon, 'x', x_terminal)
        x_warmstart_list[self.horizon] = x_terminal  # Store for track parameter calculation
        
        # ═══════════════════════════════════════════════════════════
        # Fix initial state constraint (overrides warm-start at k=0)
        # ═══════════════════════════════════════════════════════════
        self.acados_ocp_solver.set(0, 'lbx', x0)
        self.acados_ocp_solver.set(0, 'ubx', x0)
        x_warmstart_list[0] = x0  # Use x0 for i=0 (constraint overrides warm-start)
        
        # ═══════════════════════════════════════════════════════════
        # Set track parameters using warm-started theta trajectory
        # Use theta_path from warm-started states (x[6] contains theta_path)
        # ═══════════════════════════════════════════════════════════
        for i in range(self.horizon + 1):
            # Use stored warm-started state (no redundant get call)
            theta_pred = x_warmstart_list[i][6]  # theta_path is state[6]
            
            track_params = self.track_manager.get_track_params(theta_pred)
            self.acados_ocp_solver.set(i, 'p', track_params)
        
        # ═══════════════════════════════════════════════════════════
        # Solve OCP
        # ═══════════════════════════════════════════════════════════
        status = self.acados_ocp_solver.solve()
        if status not in [0, 2] and self.verbose:
            print(f"[MPCC] Solver status {status}")
        
        # ═══════════════════════════════════════════════════════════
        # Extract optimal control
        # u_opt = [T1, T2, v_theta]
        # Environment action = [T1, T2] (first 2 elements)
        # ═══════════════════════════════════════════════════════════
        u_opt = self.acados_ocp_solver.get(0, 'u')
        v_theta_opt = u_opt[2]  # Optimal path progress rate
        
        # Update theta_path from next state prediction
        theta_prev = self.theta
        x_next = self.acados_ocp_solver.get(1, 'x')
        self.theta = x_next[6]
        theta_change = self.theta - theta_prev
        
        # Debug: Track theta changes (including backward movement detection)
        if len(self.error_history) % 50 == 0 or abs(theta_change) > 0.5 or theta_change < 0:
            cycles_completed = self.theta / self.track_manager.s_max if hasattr(self, 'track_manager') and hasattr(self.track_manager, 's_max') else 0
            status = "OK"
            if theta_change < -0.01:  # Significant backward movement
                status = "BACKWARD!"
            elif abs(theta_change) > 0.5:
                status = "LARGE_JUMP"
            print(f"[MPCC DEBUG] step={len(self.error_history)}, theta={self.theta:.4f}, "
                  f"prev={theta_prev:.4f}, delta={theta_change:+.4f}, "
                  f"cycles={cycles_completed:.2f}, v_theta={v_theta_opt:.4f}, status={status}")
        
        # Always warn about backward movement
        if theta_change < -0.001:
            print(f"[MPCC WARNING] Theta decreased! step={len(self.error_history)}, "
                  f"theta: {theta_prev:.4f} -> {self.theta:.4f} (delta={theta_change:.4f})")
        
        # ═══════════════════════════════════════════════════════════
        # Compute and record errors for diagnostics
        # MPCC frame: x = drone x, y = drone z
        # ═══════════════════════════════════════════════════════════
        pos_mpcc = np.array([x0[0], x0[2]])  # [drone_x, drone_z] = [MPCC_x, MPCC_y]
        e_c, e_l = self.track_manager.compute_errors(pos_mpcc, self.theta)
        self.error_history.append({
            'contour': e_c,
            'lag': e_l,
            'theta': self.theta,
            'v_theta': v_theta_opt,  # Record v_theta for analysis
            'theta_change': theta_change
        })
        
        # ═══════════════════════════════════════════════════════════
        # Return environment action directly (NO CONVERSION NEEDED!)
        # action = [T1, T2]
        # ═══════════════════════════════════════════════════════════
        action = np.array([u_opt[0], u_opt[1]])
        return action
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'acados_ocp_solver'):
            del self.acados_ocp_solver
    
    def get_mpcc_metrics(self):
        """Get MPCC-specific metrics.
        
        Returns:
            dict: Dictionary of MPCC metrics
        """
        if not self.error_history:
            return {
                'avg_contour_error': 0.0,
                'avg_lag_error': 0.0,
                'path_progress_pct': 0.0,
                'final_theta': 0.0
            }
        
        contour_errors = [e['contour'] for e in self.error_history]
        lag_errors = [e['lag'] for e in self.error_history]
        
        return {
            'avg_contour_error': np.mean(np.abs(contour_errors)),
            'avg_lag_error': np.mean(np.abs(lag_errors)),
            'max_contour_error': np.max(np.abs(contour_errors)),
            'max_lag_error': np.max(np.abs(lag_errors)),
            'rms_contour_error': np.sqrt(np.mean(np.square(contour_errors))),
            'rms_lag_error': np.sqrt(np.mean(np.square(lag_errors))),
            'path_progress_pct': (self.theta / self.track_manager.s_max) * 100,
            'final_theta': self.theta
        }
