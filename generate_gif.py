from numpy import load
from plot_functions import generate_gif_initial, plot_whole_trajectories, generate_gif_final
from math import pi

n_agents = 1
# epsilon_0 = 0.2
episode_to_be_plotted = 1599
# pipe_recognition_probability = 1/2.

prob_no_switch_state = 0.9

# reward_follow_smart_agent = 1

std_dev_measure_pipe = pi/16.

prob_end_surge = 1/15.

forgetting_factor = 0.99

weight_smart_agent = 0.8

visibility_pipe = 1

# directory = './data_swarming_behavior/reward_following_%.1f/%d_agents' % (reward_follow_smart_agent, n_agents)
directory = "./data_swarming_behavior_6_states/slower_lr_exp_func_weight_%.2f_noise_%.2f_visibility_%.2f/%d_agents" % (weight_smart_agent, std_dev_measure_pipe,visibility_pipe, n_agents)
# directory = "./data/benchmark_probability_recognition_double_correlated_noise_on_pipe_real_smaller_noise" #% pipe_recognition_probability
data_for_gif = load("%s/episode_%d.npz" % (directory, episode_to_be_plotted))

flag_single_agent = True

plot_whole_trajectories(n_agents, data_for_gif["x_traj"], data_for_gif["y_traj"], directory, episode_to_be_plotted, flag_single_agent)

generate_gif_initial("episode_%d" % episode_to_be_plotted, data_for_gif, directory, flag_single_agent)

generate_gif_final("episode_%d" % episode_to_be_plotted, data_for_gif, directory, flag_single_agent)

