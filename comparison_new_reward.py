import numpy as np
from auxiliary_functions import moving_average
import matplotlib.pyplot as plt
from math import pi

weight_smart_agent = 0.8

std_dev_measure_pipe = pi/16.

visibility_pipe = 0.75

t_star_lr = 6000

gamma = 0.9995

pipe_recognition_probability = 1.

epsilon_0 = 0.3

# directory = './data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0)
directory = './data_baseline_new_reward/noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f/' % (std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0)

data_swarm_6_states_1 = np.load(directory + '%d_agents/data_for_plots.npz' % 1)
visited_sections_1 = data_swarm_6_states_1["fraction_of_seen_sections_of_pipe"]
average_visited_sections_1 = data_swarm_6_states_1["average_fraction_pipe"]
moving_average_sections_1 = moving_average(visited_sections_1, 250)
moving_average_average_sections_1 = moving_average(average_visited_sections_1, 250)

data_swarm_6_states_2 = np.load(directory + '%d_agents/data_for_plots.npz' % 2)
visited_sections_2 = data_swarm_6_states_2["fraction_of_seen_sections_of_pipe"]
average_visited_sections_2 = data_swarm_6_states_2["average_fraction_pipe"]
moving_average_sections_2 = moving_average(visited_sections_2, 250)
moving_average_average_sections_2 = moving_average(average_visited_sections_2, 250)

# data_swarm_6_states_4 = np.load(directory + '%d_agents/data_for_plots.npz' % 4)
# visited_sections_4 = data_swarm_6_states_4["fraction_of_seen_sections_of_pipe"]
# average_visited_sections_4 = data_swarm_6_states_4["average_fraction_pipe"]
# moving_average_sections_4 = moving_average(visited_sections_4, 250)
# moving_average_average_sections_4 = moving_average(average_visited_sections_4, 250)

# std_dev_measure_pipe = pi/64.


# directory = './data_swarming_behavior_6_states/slower_lr_exp_func_weight_%.2f_noise_%.2f_visibility_%.2f/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe)
# directory = './data_swarming_behavior_6_states/in_line_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr)

# directory = './data_swarming_behavior_6_states/average_neigh_in_line_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr)

# directory = './data_swarming_behavior_6_states/dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe)

# w_data_swarm_6_states_1 = np.load(directory + '%d_agents/data_for_plots.npz' % 1)
# w_visited_sections_1 = w_data_swarm_6_states_1["fraction_of_seen_sections_of_pipe"]
# w_average_visited_sections_1 = w_data_swarm_6_states_1["average_fraction_pipe"]
# w_moving_average_sections_1 = moving_average(w_visited_sections_1, 50)
# w_moving_average_average_sections_1 = moving_average(w_average_visited_sections_1, 50)
#
# # directory = './data_swarming_behavior_6_states/average_neigh_in_line_dep_lr_weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr)
#
# w_data_swarm_6_states_2 = np.load(directory + '%d_agents/data_for_plots.npz' % 2)
# w_visited_sections_2 = w_data_swarm_6_states_2["fraction_of_seen_sections_of_pipe"]
# w_average_visited_sections_2 = w_data_swarm_6_states_2["average_fraction_pipe"]
# w_moving_average_sections_2 = moving_average(w_visited_sections_2, 50)
# w_moving_average_average_sections_2 = moving_average(w_average_visited_sections_2, 50)
#
# w_data_swarm_6_states_4 = np.load(directory + '%d_agents/data_for_plots.npz' % 4)
# w_visited_sections_4 = w_data_swarm_6_states_4["fraction_of_seen_sections_of_pipe"]
# w_average_visited_sections_4 = w_data_swarm_6_states_4["average_fraction_pipe"]
# w_moving_average_sections_4 = moving_average(w_visited_sections_4, 50)
# w_moving_average_average_sections_4 = moving_average(w_average_visited_sections_4, 50)

#
# w_data_swarm_6_states_6 = np.load(directory + '%d_agents/data_for_plots.npz' % 6)
# w_visited_sections_6 = w_data_swarm_6_states_6["fraction_of_seen_sections_of_pipe"]
# w_moving_average_sections_6 = moving_average(w_visited_sections_6, 50)
#
# w_data_swarm_6_states_8 = np.load(directory + '%d_agents/data_for_plots.npz' % 8)
# w_visited_sections_8 = w_data_swarm_6_states_8["fraction_of_seen_sections_of_pipe"]
# w_moving_average_sections_8 = moving_average(w_visited_sections_8, 50)
#
# w_data_swarm_6_states_16 = np.load(directory + '%d_agents/data_for_plots.npz' % 16)
# w_visited_sections_16 = w_data_swarm_6_states_16["fraction_of_seen_sections_of_pipe"]
# w_moving_average_sections_16 = moving_average(w_visited_sections_16, 50)

fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)
ax1.set_title('Moving average of the fraction of visited pipe sections', fontsize=20)

ax1.tick_params(axis="x", labelsize=20)
ax1.tick_params(axis="y", labelsize=20)


ax1.plot(range(len(visited_sections_1))[-moving_average_sections_1.size:], moving_average_sections_1, label="1")
ax1.plot(range(len(visited_sections_2))[-moving_average_sections_2.size:], moving_average_sections_2, label="2")
# ax1.plot(range(len(visited_sections_4))[-moving_average_sections_4.size:], moving_average_sections_4, label="4")
# # ax1.plot(range(len(visited_sections_6))[-moving_average_sections_6.size:], moving_average_sections_6, label="6")
# # ax1.plot(range(len(visited_sections_8))[-moving_average_sections_8.size:], moving_average_sections_8, label="8")
# # ax1.plot(range(len(visited_sections_16))[-moving_average_sections_16.size:], moving_average_sections_16, label="16")
#
ax1.plot(range(len(average_visited_sections_1))[-moving_average_average_sections_1.size:], moving_average_average_sections_1, label="1", color = "C0", linestyle = "--")
ax1.plot(range(len(average_visited_sections_2))[-moving_average_average_sections_2.size:], moving_average_average_sections_2, label="2", color = "C1", linestyle = "--")
# ax1.plot(range(len(average_visited_sections_4))[-moving_average_average_sections_4.size:], moving_average_average_sections_4, label="4", color = "C2", linestyle = "--")
# # ax1.plot(range(len(average_visited_sections_6))[-moving_average_average_sections_6.size:], moving_average_average_sections_6, label="6", color = "C3", linestyle = "--")
# ax1.plot(range(len(visited_sections_8))[-moving_average_sections_8.size:], moving_average_sections_8, label="8")
# ax1.plot(range(len(visited_sections_16))[-moving_average_sections_16.size:], moving_average_sections_16, label="16")

# ax1.plot(range(len(w_visited_sections_1))[-w_moving_average_sections_1.size:], w_moving_average_sections_1, label="1", color = "C0", linestyle = "--")
# ax1.plot(range(len(w_visited_sections_2))[-w_moving_average_sections_2.size:], w_moving_average_sections_2, label="2", color = "C1", linestyle = "--")
# ax1.plot(range(len(w_visited_sections_4))[-w_moving_average_sections_4.size:], w_moving_average_sections_4, label="4", color = "C2", linestyle = "--")
# ax1.plot(range(len(w_visited_sections_6))[-w_moving_average_sections_6.size:], w_moving_average_sections_6, label="6", color = "C3", linestyle = "--")
# ax1.plot(range(len(w_visited_sections_8))[-w_moving_average_sections_8.size:], w_moving_average_sections_8, label="8", color = "C4", linestyle = "--")
# ax1.plot(range(len(w_visited_sections_16))[-w_moving_average_sections_16.size:], w_moving_average_sections_16, label="16", color = "C5", linestyle = "--")

# ax1.plot(range(len(w_average_visited_sections_1))[-w_moving_average_average_sections_1.size:], w_moving_average_average_sections_1, label="1", color = "C0", linestyle = "--")
# ax1.plot(range(len(w_average_visited_sections_2))[-w_moving_average_average_sections_2.size:], w_moving_average_average_sections_2, label="2", color = "C1", linestyle = "--")
# ax1.plot(range(len(w_average_visited_sections_4))[-w_moving_average_average_sections_4.size:], w_moving_average_average_sections_4, label="4", color = "C2", linestyle = "--")
# ax1.plot(range(len(visited_sections_6))[-moving_average_sections_6.size:], moving_average_sections_6, label="6")
# ax1.plot(range(len(visited_sections_8))[-moving_average_sections_8.size:], moving_average_sections_8, label="8")
# ax1.plot(range(len(visited_sections_16))[-moving_average_sections_16.size:], moving_average_sections_16, label="16")


ax1.legend(fontsize=20, loc='center left', title='n_agents:', bbox_to_anchor=(1, 0.5))
plt.ylim(0, 1)
plt.grid()

plt.show()