import numpy as np
from plot_functions import plot_policy
from auxiliary_functions import compute_rotation_matrix
from math import pi
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
np.set_printoptions(threshold=np.inf)

n_agents = 16
visibility_pipe = 0.6
gamma = 0.9995
epsilon_0 = 0.3
reset_type = "area"
prob_end_lost = 1/50.
std_dev_measure_pipe = pi/64.
t_star = 24000
data_for_plots = np.load('./data_multiple_runs/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/%d_agents/2/data_for_plots.npz' % (visibility_pipe, gamma, reset_type, t_star, n_agents))
# data_for_plots = np.load('./data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_prob_end_lost_%.3f/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, prob_end_lost, n_agents))
# data_for_plots = np.load('./data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_noise_%.2f/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, std_dev_measure_pipe, n_agents))
# data_for_plots = np.load('./data_benchmark_swarm/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_benchmark_neigh/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))

K_a = data_for_plots["K_a"]
K_s = data_for_plots["K_s"]
K_s_pipe = data_for_plots["K_s_pipe"]
theta_max = data_for_plots["theta_max"]
standard_action_arrow = [1., 0.]
arrows_action = np.zeros((K_a, 2))
# theta_max = pi/2
possible_actions = [x for x in np.linspace(-theta_max, theta_max, K_a)]
for i in range(K_a):
    arrows_action[i] = np.dot(compute_rotation_matrix(possible_actions[i] + pi / 2), standard_action_arrow)

Q_matrices = data_for_plots["Q_matrices"]
Q_visits = data_for_plots["Q_visits"]

Q_visits[Q_visits > 0] = np.log(Q_visits[Q_visits > 0])
Q_visits = Q_visits/(np.max(Q_visits))

# average_Q = np.zeros(Q_matrices[0].shape)
# for i in range(n_agents):
#     average_Q += Q_matrices[i]/n_agents
#
# average_Q_visits = np.zeros(Q_visits[0].shape)
# for i in range(n_agents):
#     average_Q_visits += Q_visits[i]/n_agents
#
# plot_policy(K_s, K_s_pipe, arrows_action, average_Q, average_Q_visits, 0)
#
#
for i in range(1):
    plot_policy(K_s, K_s_pipe, arrows_action, Q_matrices[i], Q_visits[i], i)
