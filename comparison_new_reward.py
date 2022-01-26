import numpy as np
from auxiliary_functions import moving_average
import matplotlib.pyplot as plt
from math import pi

weight_smart_agent = 0.8

std_dev_measure_pipe = pi/16.

visibility_pipe = 0.75

t_star_lr = 6000

gamma = 0.998

pipe_recognition_probability = 1.

epsilon_0 = 0.3

reset_type = "line"

# directory = './data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0)
directory = './data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f_reset_%s/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0, reset_type)

# data_swarm_6_states_1 = np.load(directory + '%d_agents/data_for_plots.npz' % 1)
# visited_sections_1 = data_swarm_6_states_1["fraction_of_seen_sections_of_pipe"]
# average_visited_sections_1 = data_swarm_6_states_1["average_fraction_pipe"]
# moving_average_sections_1 = moving_average(visited_sections_1, 500)
# moving_average_average_sections_1 = moving_average(average_visited_sections_1, 500)

data_swarm_6_states_2 = np.load(directory + '%d_agents/data_for_plots.npz' % 2)
visited_sections_2 = data_swarm_6_states_2["fraction_of_seen_sections_of_pipe"]
average_visited_sections_2 = data_swarm_6_states_2["average_fraction_pipe"]
moving_average_sections_2 = moving_average(visited_sections_2, 500)
moving_average_average_sections_2 = moving_average(average_visited_sections_2, 500)

data_swarm_6_states_4 = np.load(directory + '%d_agents/data_for_plots.npz' % 4)
visited_sections_4 = data_swarm_6_states_4["fraction_of_seen_sections_of_pipe"]
average_visited_sections_4 = data_swarm_6_states_4["average_fraction_pipe"]
moving_average_sections_4 = moving_average(visited_sections_4, 500)
moving_average_average_sections_4 = moving_average(average_visited_sections_4, 500)

# data_swarm_6_states_8 = np.load(directory + '%d_agents/data_for_plots.npz' % 8)
# visited_sections_8 = data_swarm_6_states_8["fraction_of_seen_sections_of_pipe"]
# average_visited_sections_8 = data_swarm_6_states_8["average_fraction_pipe"]
# moving_average_sections_8 = moving_average(visited_sections_8, 500)
# moving_average_average_sections_8 = moving_average(average_visited_sections_8, 500)

directory = './data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0)


# directory = './data_baseline_new_reward/noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f/' % (std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0)
#
nn_data_swarm_6_states_2 = np.load(directory + '%d_agents/data_for_plots.npz' % 2)
nn_visited_sections_2 = nn_data_swarm_6_states_2["fraction_of_seen_sections_of_pipe"]
nn_average_visited_sections_2 = nn_data_swarm_6_states_2["average_fraction_pipe"]
nn_moving_average_sections_2 = moving_average(nn_visited_sections_2, 500)
nn_moving_average_average_sections_2 = moving_average(nn_average_visited_sections_2, 500)

nn_data_swarm_6_states_4 = np.load(directory + '%d_agents/data_for_plots.npz' % 4)
nn_visited_sections_4 = nn_data_swarm_6_states_4["fraction_of_seen_sections_of_pipe"]
nn_average_visited_sections_4 = nn_data_swarm_6_states_4["average_fraction_pipe"]
nn_moving_average_sections_4 = moving_average(nn_visited_sections_4, 500)
nn_moving_average_average_sections_4 = moving_average(nn_average_visited_sections_4, 500)

fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)
# ax1.set_title('Fraction of visited pipe sections (as a swarm)', fontsize=20)
ax1.set_title('Average fraction of visited pipe sections', fontsize=20)

ax1.tick_params(axis="x", labelsize=20)
ax1.tick_params(axis="y", labelsize=20)


# ax1.plot(range(len(visited_sections_1))[-moving_average_sections_1.size:], moving_average_sections_1, label="1")
# ax1.plot(range(len(visited_sections_2))[-moving_average_sections_2.size:], moving_average_sections_2, label="2")
# ax1.plot(range(len(visited_sections_4))[-moving_average_sections_4.size:], moving_average_sections_4, label="4")
# # ax1.plot(range(len(visited_sections_8))[-moving_average_sections_8.size:], moving_average_sections_8, label="8")
# #
# # # ax1.plot(range(len(nn_visited_sections_1))[-nn_moving_average_sections_1.size:], nn_moving_average_sections_1, label="1", color = "C0", linestyle = "--")
# ax1.plot(range(len(nn_visited_sections_2))[-nn_moving_average_sections_2.size:], nn_moving_average_sections_2, label="2", color = "C0", linestyle = ":")
# ax1.plot(range(len(nn_visited_sections_4))[-nn_moving_average_sections_4.size:], nn_moving_average_sections_4, label="4", color = "C1", linestyle = ":")

# ax1.plot(range(len(average_visited_sections_1))[-moving_average_average_sections_1.size:], moving_average_average_sections_1, label="1", color = "C0")
ax1.plot(range(len(average_visited_sections_2))[-moving_average_average_sections_2.size:], moving_average_average_sections_2, label="2", color = "C1")
ax1.plot(range(len(average_visited_sections_4))[-moving_average_average_sections_4.size:], moving_average_average_sections_4, label="4", color = "C2")
# ax1.plot(range(len(average_visited_sections_8))[-moving_average_average_sections_8.size:], moving_average_average_sections_8, label="8", color = "C3")

# ax1.plot(range(len(nn_average_visited_sections_1))[-nn_moving_average_average_sections_1.size:], nn_moving_average_average_sections_1, label="1", color = "C0", linestyle = ":")
ax1.plot(range(len(nn_average_visited_sections_2))[-nn_moving_average_average_sections_2.size:], nn_moving_average_average_sections_2, label="2", color = "C1", linestyle = ":")
ax1.plot(range(len(nn_average_visited_sections_4))[-nn_moving_average_average_sections_4.size:], nn_moving_average_average_sections_4, label="4", color = "C2", linestyle = ":")

ax1.legend(fontsize=20, loc='center left', title='n_agents:\n dotted = baseline case', bbox_to_anchor=(1, 0.5))
plt.ylim(0, 1)
plt.grid()

plt.show()