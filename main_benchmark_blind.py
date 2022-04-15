import pathlib
import numpy as np
from math import pi
from hidden_pipe_environment_benchmark_blind import HiddenPipeEnvironmentBenchmark
from warnings import filterwarnings

filterwarnings("ignore", category=RuntimeWarning)

# Maximum turning angle of the agent per timestep
theta_max = 3 * pi / 16

# Absolute value of the speed of the agent
v0 = 0.3

# Radius of the agent's area of vision; used to determine the agent's neighbours
R = 4

# Number of possible "neighbours" states (32 + the "no neighbours" state)
k_s = 33

# Number of possible "pipe" states
k_s_pipe = 5

# Number of possible turning angles [= number of possible actions]
k_a = 7

# Half of the agent's angle of view. (Total angle will be 2*phi)
phi = 0.5

# Discount factor (survival probability)
gamma = 0.9995

# Number of episodes
n_episodes = 1000

# Parameters describing the pipe to be followed (in this case a straight line, pointing to the right)
slope_pipe = 0
offset_pipe = 0

# Mean and standard deviation for the gaussian noise on the position
mean_position_noise = 0
std_dev_position_noise = 0.01

# Mean and standard deviation for the gaussian noise on the velocity angle
mean_velocity_noise = 0
# Computed to assure that the noise on the velocity won't cause the loss of a neighbour in less than 10 timesteps
std_dev_velocity_noise = np.sqrt((phi ** 2) / 10) / 2

# Flag that defines how the positions and velocities of agents are reset at the beginning of an episode
reset_type = "area"
# reset_type = "line"

pipe_recognition_probability = 0.95

std_dev_measure_pipe = pi/16.

prob_end_surge = 1/15.

forgetting_factor = 0.99

visibility_pipe = 0.6

forgetting_factor_neigh = 0.9

n_agents = 4

t_star_lr = 24000

print(n_agents)
AF = HiddenPipeEnvironmentBenchmark(
    theta_max,
    v0,
    R,
    k_s,
    k_s_pipe,
    k_a,
    phi,
    n_episodes,
    slope_pipe,
    offset_pipe,
    mean_velocity_noise,
    mean_position_noise,
    std_dev_velocity_noise,
    std_dev_position_noise,
    reset_type,
    gamma,
    std_dev_measure_pipe,
    prob_end_surge,
    forgetting_factor,
    visibility_pipe,
    pipe_recognition_probability
)

epsilon_0 = 0.3

input_directory = './data_multiple_runs/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/%d_agents/2' % (visibility_pipe, gamma, reset_type, t_star_lr, n_agents)

# input_directory = "./data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_longer/%d_agents" % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents)

data_for_plots = np.load('%s/data_for_plots.npz' % input_directory)

Q_matrices = data_for_plots["Q_matrices"]

for i in range(n_agents):
    AF.add_agent(i*0.5, 0, np.array([1, 0]), Q_matrices[i])
if reset_type == "line":
    AF.reset_position_and_velocities_in_line()
else:
    AF.reset_position_and_velocities_in_area()

output_directory = './data_benchmark_multiple_swarm/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/%d_agents/2' % (visibility_pipe, gamma, reset_type, t_star_lr, n_agents)
print(output_directory)
pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

AF.complete_simulation(100, output_directory)
