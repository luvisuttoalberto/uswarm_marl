import numpy as np
from plot_functions import plot_maximum_distance, plot_average_highest_reward, plot_fraction_visited_pipes, plot_average_fraction_visited_pipes
from math import pi
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
np.set_printoptions(threshold=np.inf)

n_agents = 4
visibility_pipe = 0.6
gamma = 0.9995
epsilon_0 = 0.3
reset_type = "line"
prob_end_lost = 1/50.
t_star = 3000

# data_for_plots = np.load('./data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_longer/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
data_for_plots = np.load('./data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_t_star_%d/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, t_star, n_agents))
# data_for_plots = np.load('./data_constant_recognition_extended/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_prob_end_lost_%.3f/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, prob_end_lost, n_agents))
# data_for_plots = np.load('./data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_noise_%.2f/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, std_dev_measure_pipe, n_agents))
# data_for_plots = np.load('./data_benchmark_swarm/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_benchmark_neigh/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents))

plot_maximum_distance(data_for_plots["maximum_distance_towards_objective"])
plot_average_highest_reward(data_for_plots["average_highest_reward"])
plot_fraction_visited_pipes(data_for_plots["fraction_of_seen_sections_of_pipe"])
plot_average_fraction_visited_pipes(data_for_plots["average_fraction_pipe"])

# data_for_plots = np.load('./vecchi data/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f/%d_agents/data_for_plots.npz' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, n_agents))
# data_for_plots = np.load('./data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f/%d_agents/data_for_plots.npz' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0, n_agents))
# data_for_plots = np.load('./data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f_reset_%s/%d_agents/data_for_plots.npz' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0, reset_type, n_agents))
# data_for_plots = np.load('./data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f/%d_agents/data_for_plots.npz' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, n_agents))
# data_for_plots = np.load('./data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f/%d_agents/data_for_plots.npz' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, n_agents))
# data_for_plots = np.load('./data_swarming_behavior_6_states/new_exp_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f/%d_agents/data_for_plots.npz' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, n_agents))
# data_for_plots = np.load('./data_swarming_behavior_6_states/new_exp_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%f/%d_agents/data_for_plots.npz' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, n_agents))
# data_for_plots = np.load('./data_swarming_behavior_6_states/new_exp_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d/%d_agents/data_for_plots.npz' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, n_agents))
# data_for_plots = np.load("./data_swarming_behavior_6_states/in_line_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d/%d_agents/data_for_plots.npz" % (weight_smart_agent, std_dev_measure_pipe,visibility_pipe, t_star_lr, n_agents))
# data_for_plots = np.load("./data_swarming_behavior_6_states/average_neigh_in_line_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d/%d_agents/data_for_plots.npz" % (weight_smart_agent, std_dev_measure_pipe,visibility_pipe, t_star_lr, n_agents))
# data_for_plots = np.load("./data_swarming_behavior_6_states/slower_lr_exp_func_weight_%.2f_noise_%.2f_visibility_%.2f/%d_agents/data_for_plots.npz" % (weight_smart_agent, std_dev_measure_pipe,visibility_pipe, n_agents))
