import pathlib
import numpy as np
from math import pi
from warnings import filterwarnings

import hidden_pipe_environment_single_agent

filterwarnings("ignore", category=RuntimeWarning)

# Initial exploration rate
epsilon_0 = 0.3

# Maximum turning angle of the agent per timestep
theta_max = 3 * pi / 16

# Absolute value of the speed of the agent
v0 = 0.3

# Radius of the agent's area of vision; used to determine the agent's neighbours
R = 4

# Number of possible "neighbours" states (32 + the "no neighbours" state)
k_s = 32

# Number of possible "pipe" states
k_s_pipe = 5

# Number of possible turning angles [= number of possible actions]
k_a = 7

# Half of the agent's angle of view. (Total angle will be 2*phi)
phi = 0.5

# Discount factor (survival probability)
gamma = 0.9995

# Number of episodes
n_episodes = 1600

# T_star epsilon (Timestep in the learning at which the exploration rate starts to decrease)
# Can be different from t_star_lr
t_star_epsilon = 600

# T_star learning rate (Timestep in the learning at which the learning rate starts to decrease).
# Can be different from t_star_epsilon
t_star_lr = 600

t_stop = 1500

# Initial learning rate
alpha_0 = 0.005

# Parameters describing the pipe to be followed (in this case a straight line, pointing to the right)
slope_pipe = 0
offset_pipe = 0

# Mean and standard deviation for the gaussian noise on the position
mean_position_noise = 0
std_dev_position_noise = 0.01

# Mean and standard deviation for the gaussian noise on the velocity angle
mean_velocity_noise = 0
# Computed to assure that the noise on the velocity won't cause the loss of a neighbour in less than 5 timesteps
std_dev_velocity_noise = np.sqrt((phi ** 2) / 10) / 2

std_dev_measure_pipe = pi/16.

prob_end_surge = 1/15.

forgetting_factor = 0.99

visibility_pipe = 0.6

pipe_recognition_probability = 0.95

AF = hidden_pipe_environment_single_agent.HiddenPipeEnvironmentSingleAgent(
    theta_max,
    v0,
    R,
    k_s,
    k_s_pipe,
    k_a,
    alpha_0,
    phi,
    n_episodes,
    t_star_epsilon,
    t_star_lr,
    t_stop,
    epsilon_0,
    slope_pipe,
    offset_pipe,
    mean_velocity_noise,
    mean_position_noise,
    std_dev_velocity_noise,
    std_dev_position_noise,
    gamma,
    std_dev_measure_pipe,
    prob_end_surge,
    forgetting_factor,
    visibility_pipe,
    pipe_recognition_probability
)

AF.reset_position_and_velocity()

output_directory = './data_trace_6_states/longer_no_step_no_scale_old_exp_6_states_%.2f_noise_%.3f_forgetting_factor_%.2f_4' % (prob_end_surge, std_dev_measure_pipe, forgetting_factor)
pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

AF.complete_simulation(50, output_directory)
