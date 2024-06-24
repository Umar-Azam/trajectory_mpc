# %%

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json

# Define velocity constraints
MAX_LINEAR_VELOCITY = 1.0  # m/s
MIN_LINEAR_VELOCITY = -1.0  # m/s
MAX_ANGULAR_VELOCITY = 1.0  # rad/s
MIN_ANGULAR_VELOCITY = -1.0  # rad/s
MAX_ACCEL = 2.0  # m/s^2
MIN_ACCEL = -2.0  # m/s^2

def system_dynamics(state, control, dt):
    X, Y, theta, v, omega = state
    a, alpha = control
    
    X_next = X + v * np.cos(theta) * dt
    Y_next = Y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    if (theta_next >= 2*np.pi):
        theta_next = theta_next - 2*np.pi
    if (theta_next < 0):
        theta_next = theta_next + 2*np.pi

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
    position_state_cost = 50 * np.sum((trajectory[:, :2] - reference_trajectory[:, :2])**2)
    

    angle_cost = np.sum((1 - np.cos(trajectory[:, 2] - reference_trajectory[:, 2]))**2)
    
    # Control input cost (reduced weight)
    control_cost = 0.01 * np.sum(controls**2)

    # Velocity constraint penalty
    velocity_penalty = 1000 * velocity_constraint_penalty(trajectory)
    
    # # Add a cost for final state error
    # final_state_cost = 100 * np.sum((trajectory[-1, :2] - reference_trajectory[-1, :2])**2)
    
    # return state_cost + control_cost + final_state_cost

    print(position_state_cost + angle_cost + control_cost + velocity_penalty )
    return position_state_cost + angle_cost + control_cost + velocity_penalty 

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
        bounds=[(MIN_ACCEL, MAX_ACCEL)] * (N * n_controls)  # Control Boundary
    )
    
    optimal_controls = result.x.reshape(-1, 2)
    return optimal_controls[0]  # Return only the first control input

def generate_reference_trajectory(N):
    t = np.linspace(0, 2*np.pi, N+1)
    X = np.cos(t)
    Y = np.sin(t)
    theta = np.arctan2(np.cos(t), -np.sin(t))
    for i in range(len(theta)):
        if (theta[i] >= 2*np.pi):
            theta[i] = theta[i]  - 2*np.pi
        if (theta[i]  < 0.0):
            theta[i]  = theta[i] + 2*np.pi
    return np.column_stack((X, Y,theta,  np.zeros((N+1, 2))))  # Padding with zeros for theta, v, omega

# %%

# # MPC parameters
# N = 20  # Prediction horizon
# dt = 0.05  # Time step


# # Initial state
# initial_state = np.array([1, 0, np.pi/2, 0, 0])  # [X, Y, theta, v, omega]

# # Simulation
# num_steps = 200  # Increased number of steps for full circle
# actual_trajectory = np.zeros((num_steps+1, 5))
# actual_trajectory[0] = initial_state

# # Generate reference trajectory
# reference_trajectory = generate_reference_trajectory(num_steps)

# # Padding for a receding horizon window 
# padding = np.tile(reference_trajectory[-1],(20,1))
# reference_trajectory = np.vstack((reference_trajectory, padding))

# # %%

# control_list = np.zeros(shape = (num_steps,2))

# for i in range(num_steps):
#     # Compute optimal control
#     optimal_control = mpc_control(actual_trajectory[i], reference_trajectory[i:i+1+N], dt, N)
    
#     # Save optimal control parameters
#     control_list[i] = optimal_control

#     # Apply control and update state
#     actual_trajectory[i+1] = system_dynamics(actual_trajectory[i], optimal_control, dt)

# # Plotting
# plt.figure(figsize=(10, 8))
# plt.plot(reference_trajectory[:, 0], reference_trajectory[:, 1], 'r--', label='Reference')
# plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'b-', label='MPC')
# plt.title('MPC Trajectory Tracking')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.show()
# # %%


# %%


# MPC parameters
N = 20  # Prediction horizon
dt = 0.05  # Time step


# Initial state
initial_state = np.array([0, 0, 0, 0, 0])  # [X, Y, theta, v, omega]

# Simulation
max_num_steps = 500  # Increased number of steps for full circle
actual_trajectory = np.zeros((max_num_steps+1, 5))
actual_trajectory[0] = initial_state

# # Generate reference trajectory
# reference_trajectory = generate_reference_trajectory(num_steps)

# # Padding for a receding horizon window 
# padding = np.tile(reference_trajectory[-1],(20,1))
# reference_trajectory = np.vstack((reference_trajectory, padding))

# # Break down trajectory to remove local minima in cost function
# # Issue seen when robot is at X,Y,Theta = (0,0,0) and needs to be at (0,1,0)
# # and can only move in theta direction. 

def reference_generator(goal_point, horizon_legth):
    return np.tile(goal_point,(horizon_legth,1))

def trajectory_parser(ref_trajectory):
    
    # Calculate the number of intermediate points
    num_intermediate = len(ref_trajectory) - 1
    
    # Initialize the decomposed trajectory with double the size minus 1
    decomposed_trajectory = np.zeros((2 * len(ref_trajectory) - 1, 5))
    
    for i in range(num_intermediate):
        # Copy the current reference point
        decomposed_trajectory[2*i] = ref_trajectory[i]
        
        # Calculate the intermediate point
        x1, y1 = ref_trajectory[i][:2]
        x2, y2 = ref_trajectory[i+1][:2]
        
        # Calculate the angle between the two points
        angle = np.arctan2(y2 - y1, x2 - x1)

        if (angle  >= 2*np.pi):
            angle  = angle  - 2*np.pi
        if (angle  < 0):
            angle  = angle  + 2*np.pi
        
        # Set the intermediate point
        decomposed_trajectory[2*i + 1] = [
            (x1 + x2) / 2,  # X: midpoint
            (y1 + y2) / 2,  # Y: midpoint
            angle,          # theta: angle between points
            0,              # v: velocity (set to 0 for intermediate)
            0               # w: angular velocity (set to 0 for intermediate)
        ]
    
    # Add the last reference point
    decomposed_trajectory[-1] = ref_trajectory[-1]
    
    return decomposed_trajectory 

# %%

reference_trajectory = np.zeros((5,5))
reference_trajectory[0,:] = np.array([0,0,0,0,0])
reference_trajectory[1,:] = np.array([1,0,0,0,0])
reference_trajectory[2,:] = np.array([1,1,0,0,0])
reference_trajectory[3,:] = np.array([1,1,np.pi/2,0,0])
reference_trajectory[4,:] = np.array([1,1,0,0,0])

reference_trajectory = trajectory_parser(reference_trajectory)


control_list = np.zeros(shape = (max_num_steps,2))
threshold = 0.05

ind = 0

for i in range(4):

  
    input_traj = reference_generator(reference_trajectory[i,:],N+1)
    #mpc_control(actual_trajectory[ind], input_traj , dt, N)

    while(np.sum((actual_trajectory[ind] - reference_trajectory[i,:])**2) > threshold):

        optimal_control = mpc_control(actual_trajectory[ind], input_traj , dt, N)
        actual_trajectory[ind+1] = system_dynamics(actual_trajectory[ind], optimal_control, dt)
        ind = ind + 1

    

# %%



plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'b-', label='MPC')
plt.title('MPC Trajectory Tracking')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()



# %%



for i in range(len(reference_trajectory[:,0])):

    # Index for the actual trajectory when seeking a goal
    ind = 0


    # Compute optimal control
    optimal_control = mpc_control(actual_trajectory[ind], reference_generator(reference_trajectory[i,:],N+1), dt, N)
    

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