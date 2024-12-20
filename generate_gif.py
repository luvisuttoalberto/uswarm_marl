from numpy import load
from plot_functions import generate_gif_initial, plot_whole_trajectories, generate_gif_final

n_agents = 1

# episode_to_be_plotted = 10000
# episode_to_be_plotted = 19999

# episode_to_be_plotted = 30000
episode_to_be_plotted = 15999
# episode_to_be_plotted = 0
# episode_to_be_plotted = 25000
# episode_to_be_plotted =
# episode_to_be_plotted = 63999
# episode_to_be_plotted = 31999

visibility_pipe = 0.6

gamma = 0.9995

epsilon_0 = 0.3

# reset_type = "area"
reset_type = "area"

t_star = 24000

phi=1
# directory = "./data_new_angle/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_t_star_%d_phi_%d/%d_agents" % (visibility_pipe, gamma, epsilon_0, reset_type, t_star, 2*phi, n_agents)
# directory = "./data_benchmark_average_policy/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d_fixed_length_shorter/%d_agents/2" % (visibility_pipe, gamma, reset_type, t_star, n_agents)


# directory = './data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_t_star_%d_longer/%d_agents' % (visibility_pipe, gamma, epsilon_0, reset_type, t_star, n_agents)
directory = './data_multiple_runs/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/%d_agents/4' % (visibility_pipe, gamma, reset_type, t_star, n_agents)
# directory = "./data_benchmark_multiple/visibility_0.50_gamma_0.9995_reset_line_t_star_24000/16_agents/1"
# directory = './data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_closer/%d_agents' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents)
# directory = './data_benchmark_neigh/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents)
# directory = './data_benchmark_swarm/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_correct/%d_agents' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents)
# directory = './data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_noise_%.2f/%d_agents' % (visibility_pipe, gamma, epsilon_0, reset_type, std_dev_measure_pipe, n_agents)
# directory = './data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f_reset_%s/%d_agents' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0, reset_type, n_agents)
# directory = './data_baseline_new_reward/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/%d_agents' % (visibility_pipe, gamma, epsilon_0, reset_type, n_agents)

print(directory)

print(episode_to_be_plotted)

data_for_gif = load("%s/episode_%d.npz" % (directory, episode_to_be_plotted))

flag_single_agent = True

plot_whole_trajectories(n_agents, data_for_gif["x_traj"], data_for_gif["y_traj"], directory, episode_to_be_plotted, flag_single_agent)

generate_gif_initial("episode_%d" % episode_to_be_plotted, data_for_gif, directory, flag_single_agent)
#
# generate_gif_final("episode_%d" % episode_to_be_plotted, data_for_gif, directory, flag_single_agent)

