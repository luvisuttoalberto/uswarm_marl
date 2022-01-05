import numpy as np
from plot_functions import plot_maximum_distance, plot_policy, plot_Q_matrices, plot_average_highest_reward, plot_fraction_visited_pipes, plot_average_fraction_visited_pipes
from auxiliary_functions import compute_rotation_matrix
from math import pi

np.set_printoptions(threshold=np.inf)

n_agents = 1

pipe_recognition_probability = 1/32.

reward_follow_smart_agent = 0.5

weight_smart_agent = 0.8

std_dev_measure_pipe = pi/16.

visibility_pipe = 1

data_for_plots = np.load("./data_swarming_behavior_6_states/slower_lr_exp_func_weight_%.2f_noise_%.2f_visibility_%.2f/%d_agents/data_for_plots.npz" % (weight_smart_agent, std_dev_measure_pipe,visibility_pipe, n_agents))

# plot_maximum_distance(data_for_plots["maximum_distance_towards_objective"])
# plot_average_highest_reward(data_for_plots["average_highest_reward"])
# plot_fraction_visited_pipes(data_for_plots["fraction_of_seen_sections_of_pipe"])
plot_average_fraction_visited_pipes(data_for_plots["average_fraction_pipe"])

K_a = data_for_plots["K_a"]
K_s = data_for_plots["K_s"]
K_s_pipe = data_for_plots["K_s_pipe"]
theta_max = data_for_plots["theta_max"]

standard_action_arrow = [1., 0.]
arrows_action = np.zeros((K_a, 2))
possible_actions = [x for x in np.linspace(-theta_max, theta_max, K_a)]
for i in range(K_a):
    arrows_action[i] = np.dot(compute_rotation_matrix(possible_actions[i] + pi / 2), standard_action_arrow)

Q_matrices = data_for_plots["Q_matrices"]
Q_visits = data_for_plots["Q_visits"]

Q_visits[Q_visits > 0] = np.log(Q_visits[Q_visits > 0])
Q_visits = Q_visits/(np.max(Q_visits))

# for i in range(1):
#     plot_policy(K_s, K_s_pipe, arrows_action, Q_matrices[i], Q_visits[i], i)

# plot_Q_matrices(K_s_pipe, n_agents, Q_matrices, K_s)
