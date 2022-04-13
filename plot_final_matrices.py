import numpy as np
from plot_functions import plot_Q_matrices, plot_visit_matrices
from auxiliary_functions import compute_rotation_matrix
from math import pi
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
np.set_printoptions(threshold=np.inf)

n_agents = 8
visibility_pipe = 0.6
gamma = 0.9995
epsilon_0 = 0.3
reset_type = "area"
prob_end_lost = 1/50.
std_dev_measure_pipe = pi/64.
t_star = 24000
data_for_plots = np.load('./data_multiple_runs/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/%d_agents/2/data_for_plots.npz' % (visibility_pipe, gamma, reset_type, t_star, n_agents))
# data_for_plots = np.load('./data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_closer/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_closer/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_t_star_%d/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, t_star, n_agents))
# data_for_plots = np.load('./data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_noise_%.2f/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, std_dev_measure_pipe, n_agents))
# data_for_plots = np.load('./data_benchmark_swarm/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_benchmark_neigh/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))

K_s = data_for_plots["K_s"]
K_s_pipe = data_for_plots["K_s_pipe"]
Q_matrices = data_for_plots["Q_matrices"]
V_matrices = data_for_plots["global_state_action_rate_visits"]
# V_matrices[V_matrices > 0] = np.log(V_matrices[V_matrices > 0])
# V_matrices = V_matrices/(np.max(V_matrices))

average_Q = np.zeros(Q_matrices.shape)
for i in range(n_agents):
    average_Q[0] += Q_matrices[i]/n_agents

# plot_Q_matrices(K_s_pipe, n_agents, Q_matrices, K_s)
plot_Q_matrices(K_s_pipe, 1, average_Q, K_s)

# plot_visit_matrices(K_s_pipe, n_agents, V_matrices, K_s)