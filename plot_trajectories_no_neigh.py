import numpy as np
from plot_functions import plot_maximum_distance, plot_fraction_visited_pipes, plot_policy, plot_Q_matrices, \
    plot_Q_matrix_no_neigh_version, plot_policy_no_neigh
from auxiliary_functions import compute_rotation_matrix
from math import pi

np.set_printoptions(threshold=np.inf)

n_agents = 8

data_for_plots = np.load("./data_pipe_5_m_sections_without_neighbors/%d_agents/data_for_plots.npz" % n_agents)

plot_maximum_distance(data_for_plots["maximum_distance_towards_objective"])
plot_fraction_visited_pipes(data_for_plots["fraction_of_seen_sections_of_pipe"])

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

Q_visits = Q_visits/(np.max(Q_visits))
Q_visits[Q_visits > 0] += 0.1
Q_visits[Q_visits > 0.9] -= 0.1

for i in range(n_agents):
    plot_policy_no_neigh(K_s, K_s_pipe, arrows_action, Q_matrices[i], Q_visits[i], i)

for i in range(K_s_pipe):
    plot_Q_matrix_no_neigh_version(i, n_agents, Q_matrices, K_s)
