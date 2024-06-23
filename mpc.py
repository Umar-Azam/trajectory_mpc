# %%

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Define velocity constraints
MAX_LINEAR_VELOCITY = 1.0  # m/s
MIN_LINEAR_VELOCITY = -1.0  # m/s
MAX_ANGULAR_VELOCITY = 1.0  # rad/s
MIN_ANGULAR_VELOCITY = -1.0  # rad/s

def system_dynamics(state, control, dt):
    X, Y, theta, v, omega = state
    a, alpha = control
    
    X_next = X + v * np.cos(theta) * dt
    Y_next = Y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    v_next = v + a * dt
    omega_next = omega + alpha * dt
    
    return np.array([X_next, Y_next, theta_next, v_next, omega_next])

def predict_trajectory(initial_state, controls, dt, N):
    trajectory = np.zeros((N+1, 5))
    trajectory[0] = initial_state
    
    for i in range(N):
        trajectory[i+1] = system_dynamics(trajectory[i], controls[i], dt)
    
    return trajectory


def velocity_constraint_penalty(trajectory):
    v_penalty = np.sum(np.maximum(0, trajectory[:, 3] - MAX_LINEAR_VELOCITY)**2 +
                       np.maximum(0, MIN_LINEAR_VELOCITY - trajectory[:, 3])**2)
    omega_penalty = np.sum(np.maximum(0, trajectory[:, 4] - MAX_ANGULAR_VELOCITY)**2 +
                           np.maximum(0, MIN_ANGULAR_VELOCITY - trajectory[:, 4])**2)
    return v_penalty + omega_penalty


def cost_function(controls, initial_state, reference_trajectory, dt, N):
    trajectory = predict_trajectory(initial_state, controls.reshape(-1, 2), dt, N)
    
    # State error cost (increased weight)
    state_cost = 10 * np.sum((trajectory[:, :2] - reference_trajectory[:, :2])**2)
    
    # Control input cost (reduced weight)
    control_cost = 0.01 * np.sum(controls**2)

    # Velocity constraint penalty
    velocity_penalty = 1000 * velocity_constraint_penalty(trajectory)
    
    # # Add a cost for final state error
    # final_state_cost = 100 * np.sum((trajectory[-1, :2] - reference_trajectory[-1, :2])**2)
    
    # return state_cost + control_cost + final_state_cost


    return state_cost + control_cost + velocity_penalty

# Works fine up until here

# %%

def mpc_control(initial_state, reference_trajectory, dt, N):
    n_controls = 2  # Number of control inputs (a, alpha)
    
    # Initial guess for controls
    controls_guess = np.zeros(N * n_controls)
    
    # Optimization
    result = minimize(
        cost_function,
        controls_guess,
        args=(initial_state, reference_trajectory, dt, N),
        method='SLSQP',
        bounds=[(-2, 2)] * (N * n_controls)  # Control Boundary
    )
    
    optimal_controls = result.x.reshape(-1, 2)
    return optimal_controls[0]  # Return only the first control input

def generate_reference_trajectory(N):
    t = np.linspace(0, 2*np.pi, N+1)
    X = np.cos(t)
    Y = np.sin(t)
    return np.column_stack((X, Y, np.zeros((N+1, 3))))  # Padding with zeros for theta, v, omega

# %%

# MPC parameters
N = 20  # Prediction horizon
dt = 0.05  # Time step


# Initial state
initial_state = np.array([1, 0, np.pi/2, 0, 0])  # [X, Y, theta, v, omega]

# Simulation
num_steps = 200  # Increased number of steps for full circle
actual_trajectory = np.zeros((num_steps+1, 5))
actual_trajectory[0] = initial_state

# Generate reference trajectory
reference_trajectory = generate_reference_trajectory(num_steps)

# Padding for a receding horizon window 
padding = np.tile(reference_trajectory[-1],(20,1))
reference_trajectory = np.vstack((reference_trajectory, padding))

# %%

control_list = np.zeros(shape = (num_steps,2))

for i in range(num_steps):
    # Compute optimal control
    optimal_control = mpc_control(actual_trajectory[i], reference_trajectory[i:i+1+N], dt, N)
    
    # Save optimal control parameters
    control_list[i] = optimal_control

    # Apply control and update state
    actual_trajectory[i+1] = system_dynamics(actual_trajectory[i], optimal_control, dt)

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(reference_trajectory[:, 0], reference_trajectory[:, 1], 'r--', label='Reference')
plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'b-', label='MPC')
plt.title('MPC Trajectory Tracking')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
# %%


# %%
