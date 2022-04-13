import numpy as np
from plot_functions import plot_maximum_distance, plot_Q_matrix_no_neigh_version, plot_policy_no_neigh, plot_average_highest_reward, plot_fraction_visited_pipes, plot_average_fraction_visited_pipes
from auxiliary_functions import compute_rotation_matrix
from math import pi

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

np.set_printoptions(threshold=np.inf)

visibility_pipe = 0.9

gamma = 0.9995

epsilon_0 = 0.3

flag_single_agent = True

n_agents = 4

reset_type = "area"

data_for_plots = np.load('./data_baseline_new_reward/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))

# data_for_plots = np.load("./data_trace_6_states/no_step_no_scale_old_exp_6_states_%.2f_prob_%.2f_noise_%.3f_forgetting_factor_%.2f_2/data_for_plots.npz" % (prob_end_surge, prob_no_switch_state, std_dev_measure_pipe, forgetting_factor))

plot_maximum_distance(data_for_plots["maximum_distance_towards_objective"])
plot_average_highest_reward(data_for_plots["average_highest_reward"])
plot_fraction_visited_pipes(data_for_plots["fraction_of_seen_sections_of_pipe"])
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

for i in range(n_agents):
    plot_policy_no_neigh(K_s, K_s_pipe, arrows_action, Q_matrices[i], Q_visits[i], i)

for i in range(K_s_pipe):
    plot_Q_matrix_no_neigh_version(i, n_agents, Q_matrices, K_s)

V_matrices = data_for_plots["global_state_action_rate_visits"]
# V_matrices[V_matrices > 0] = np.log(V_matrices[V_matrices > 0])
# V_matrices = V_matrices/(np.max(V_matrices))

# plot_visit_matrices(K_s_pipe, n_agents, V_matrices, K_s)
