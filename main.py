import pathlib
import numpy as np
from math import pi
import hidden_pipe_environment
from warnings import filterwarnings

filterwarnings("ignore", category=RuntimeWarning)

# Number of agents
# n_agents = 16

# Initial exploration rate
epsilon_0 = 0.3

# Maximum turning angle of the agent per timestep
theta_max = 3 * pi / 16

# Absolute value of the speed of the agent
v0 = 0.3

# Radius of the agent's area of vision; used to determine the agent's neighbours
R = 4

# Number of possible "neighbours" states (32 + the "no neighbours" state)
k_s = 33

# Number of possible "pipe" states
k_s_pipe = 6

# Number of possible turning angles [= number of possible actions]
k_a = 7

# Half of the agent's angle of view. (Total angle will be 2*phi)
phi = 0.5

# Discount factor (survival probability)
gamma = 0.9999

# Number of episodes
n_episodes = 1600

# T_star epsilon (Timestep in the learning at which the exploration rate starts to decrease)
# Can be different from t_star_lr
t_star_epsilon = 300

# T_star learning rate (Timestep in the learning at which the learning rate starts to decrease).
# Can be different from t_star_epsilon
t_star_lr = 500

t_stop = 1500

# Initial learning rate
alpha_0 = 0.005

# Parameters describing the pipe to be followed (in this case a straight line, pointing to the right)
slope_pipe = 0
offset_pipe = 0

# Mean and standard deviation for the gaussian noise on the position
mean_position_noise = 0
std_dev_position_noise = 0.01
# std_dev_position_noise = v0/8
# std_dev_position_noise = 0

# Mean and standard deviation for the gaussian noise on the velocity angle
mean_velocity_noise = 0
# Computed to assure that the noise on the velocity won't cause the loss of a neighbour in less than 10 timesteps
std_dev_velocity_noise = np.sqrt((phi ** 2) / 10) / 2
# std_dev_velocity_noise = 0

# Distance from the pipe that defines the region in which the agent receives a reward
# distance_from_pipe = R*np.sin(phi/2)

# Flag that defines how the positions and velocities of agents are reset at the beginning of an episode
# reset_type = "area"
reset_type = "line"

# pipe_recognition_probability = 1

flag_spatially_uncorrelated_case = False


std_dev_measure_pipe = pi/16.

print(std_dev_measure_pipe)

prob_end_surge = 1/15.

forgetting_factor = 0.99

weight_smart_agent = 0.8

visibility_pipe = 0.5

# reward_follow_smart_agent = 0.8

for j in [1,2,4]:
    print(j)
    AF = hidden_pipe_environment.HiddenPipeEnvironment(
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
        reset_type,
        gamma,
        flag_spatially_uncorrelated_case,
        std_dev_measure_pipe,
        prob_end_surge,
        forgetting_factor,
        weight_smart_agent,
        visibility_pipe
    )

    for i in range(j):
        AF.add_agent(i*0.5, 0, np.array([1, 0]))
    if reset_type == "line":
        AF.reset_position_and_velocities_in_line()
    else:
        AF.reset_position_and_velocities_in_area()

    output_directory = './data_swarming_behavior_6_states/in_line_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d/%d_agents' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, j)
    print(output_directory)
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    AF.complete_simulation(50, output_directory)
