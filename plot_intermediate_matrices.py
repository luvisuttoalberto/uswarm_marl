import numpy as np
from plot_functions import plot_Q_matrices, plot_visit_matrices
from math import pi
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
np.set_printoptions(threshold=np.inf)

n_agents = 4
visibility_pipe = 0.6
gamma = 0.998
epsilon_0 = 0.3
reset_type = "line"
episode_to_be_plotted = 30000
# episode_to_be_plotted = 31999

t_star = 3000

data_for_intermediate_matrices_plots = np.load('./data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_t_star_%d/%d_agents/episode_%d.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, t_star, n_agents, episode_to_be_plotted))

Q_matrices_episode = data_for_intermediate_matrices_plots["Q_matrices"]
K_s = 33
K_s_pipe = 5
V_matrices = data_for_intermediate_matrices_plots["global_state_action_rate_visits"]
# V_matrices[V_matrices > 0] = np.log(V_matrices[V_matrices > 0])
# V_matrices = V_matrices/(np.max(V_matrices))

# plot_Q_matrices(K_s_pipe, n_agents, Q_matrices_episode, K_s)
#
plot_visit_matrices(K_s_pipe, n_agents, V_matrices, K_s)
