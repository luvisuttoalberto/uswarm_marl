import numpy as np
from auxiliary_functions import moving_average
import matplotlib.pyplot as plt

visibility_pipe = 0.6

gamma = 0.9995

epsilon_0 = 0.3

reset_type = "area"

t_star_lr = 24000



directory = './data_benchmark_multiple_swarm/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/' % (visibility_pipe, gamma, reset_type, t_star_lr)
# directory = './data_benchmark_average_policy/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d_fixed_length_shorter/' % (visibility_pipe, gamma, reset_type, t_star_lr)

# directory = './data_benchmark_swarm/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_extended/' % (visibility_pipe, gamma, epsilon_0, reset_type)

data_swarm_6_states_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 1)
visited_sections_1 = data_swarm_6_states_1["fraction_of_seen_sections_of_pipe"]
average_visited_sections_1 = data_swarm_6_states_1["average_fraction_pipe"]
moving_average_sections_1 = moving_average(visited_sections_1, 1)
moving_average_average_sections_1 = moving_average(average_visited_sections_1, 1)
# #
# data_swarm_6_states_2 = np.load(directory + '%d_agents/data_for_plots.npz' % 2)
# visited_sections_2 = data_swarm_6_states_2["fraction_of_seen_sections_of_pipe"]
# average_visited_sections_2 = data_swarm_6_states_2["average_fraction_pipe"]
# moving_average_sections_2 = moving_average(visited_sections_2, 250)
# moving_average_average_sections_2 = moving_average(average_visited_sections_2, 250)

# data_swarm_6_states_4 = np.load(directory + '%d_agents/data_for_plots.npz' % 4)
# visited_sections_4 = data_swarm_6_states_4["fraction_of_seen_sections_of_pipe"]
# average_visited_sections_4 = data_swarm_6_states_4["average_fraction_pipe"]
# moving_average_sections_4 = moving_average(visited_sections_4, 250)
# moving_average_average_sections_4 = moving_average(average_visited_sections_4, 250)

# data_swarm_6_states_8 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 8)
# visited_sections_8 = data_swarm_6_states_8["fraction_of_seen_sections_of_pipe"]
# average_visited_sections_8 = data_swarm_6_states_8["average_fraction_pipe"]
# moving_average_sections_8 = moving_average(visited_sections_8, 1)
# moving_average_average_sections_8 = moving_average(average_visited_sections_8, 1)

# data_swarm_6_states_16 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 16)
# visited_sections_16 = data_swarm_6_states_16["fraction_of_seen_sections_of_pipe"]
# average_visited_sections_16 = data_swarm_6_states_16["average_fraction_pipe"]
# moving_average_sections_16 = moving_average(visited_sections_16, 1)
# moving_average_average_sections_16 = moving_average(average_visited_sections_16, 1)

# directory = './data_swarming_behavior_new_reward/weight_%.2f_noise_%.2f_visibility_%.2f_t_star_%d_gamma_%.4f_recognition_%.2f_eps_%.1f/' % (weight_smart_agent, std_dev_measure_pipe, visibility_pipe, t_star_lr, gamma, pipe_recognition_probability, epsilon_0)

# directory = './data_benchmark_neigh/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_extended/' % (visibility_pipe, gamma, epsilon_0, reset_type)

reset_type = "area"

directory = './data_benchmark_average_policy/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d_fixed_length/' % (visibility_pipe, gamma, reset_type, t_star_lr)


# directory = './data_benchmark_multiple_neigh/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/' % (visibility_pipe, gamma, reset_type, t_star_lr)

nn_data_swarm_6_states_1 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 1)
nn_visited_sections_1 = nn_data_swarm_6_states_1["fraction_of_seen_sections_of_pipe"]
nn_average_visited_sections_1 = nn_data_swarm_6_states_1["average_fraction_pipe"]
nn_moving_average_sections_1 = moving_average(nn_visited_sections_1, 1)
nn_moving_average_average_sections_1 = moving_average(nn_average_visited_sections_1, 1)

# nn_data_swarm_6_states_2 = np.load(directory + '%d_agents/data_for_plots.npz' % 2)
# nn_visited_sections_2 = nn_data_swarm_6_states_2["fraction_of_seen_sections_of_pipe"]
# nn_average_visited_sections_2 = nn_data_swarm_6_states_2["average_fraction_pipe"]
# nn_moving_average_sections_2 = moving_average(nn_visited_sections_2, 250)
# nn_moving_average_average_sections_2 = moving_average(nn_average_visited_sections_2, 250)

# nn_data_swarm_6_states_4 = np.load(directory + '%d_agents/data_for_plots.npz' % 4)
# nn_visited_sections_4 = nn_data_swarm_6_states_4["fraction_of_seen_sections_of_pipe"]
# nn_average_visited_sections_4 = nn_data_swarm_6_states_4["average_fraction_pipe"]
# nn_moving_average_sections_4 = moving_average(nn_visited_sections_4, 250)
# nn_moving_average_average_sections_4 = moving_average(nn_average_visited_sections_4, 250)

# nn_data_swarm_6_states_8 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 8)
# nn_visited_sections_8 = nn_data_swarm_6_states_8["fraction_of_seen_sections_of_pipe"]
# nn_average_visited_sections_8 = nn_data_swarm_6_states_8["average_fraction_pipe"]
# nn_moving_average_sections_8 = moving_average(nn_visited_sections_8, 1)
# nn_moving_average_average_sections_8 = moving_average(nn_average_visited_sections_8, 1)

# nn_data_swarm_6_states_16 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 16)
# nn_visited_sections_16 = nn_data_swarm_6_states_16["fraction_of_seen_sections_of_pipe"]
# nn_average_visited_sections_16 = nn_data_swarm_6_states_16["average_fraction_pipe"]
# nn_moving_average_sections_16 = moving_average(nn_visited_sections_16, 1)
# nn_moving_average_average_sections_16 = moving_average(nn_average_visited_sections_16, 1)

fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)

ax1.tick_params(axis="x", labelsize=20)
ax1.tick_params(axis="y", labelsize=20)


ax1.set_title('Fraction of visited pipe sections (as a swarm)', fontsize=20)
ax1.plot(range(len(visited_sections_1))[-moving_average_sections_1.size:], moving_average_sections_1, label="1")
# ax1.plot(range(len(visited_sections_2))[-moving_average_sections_2.size:], moving_average_sections_2, label="2")
# ax1.plot(range(len(visited_sections_4))[-moving_average_sections_4.size:], moving_average_sections_4, label="without neighbors", color = "C0")
# ax1.plot(range(len(visited_sections_8))[-moving_average_sections_8.size:], moving_average_sections_8, label="without neighbors", color = "C1")
# ax1.plot(range(len(visited_sections_16))[-moving_average_sections_16.size:], moving_average_sections_16, label="line", color = "C2")
ax1.plot(range(len(nn_visited_sections_1))[-nn_moving_average_sections_1.size:], nn_moving_average_sections_1, label="1", color = "C0", linestyle = ":")
# ax1.plot(range(len(nn_visited_sections_2))[-nn_moving_average_sections_2.size:], nn_moving_average_sections_2, label="2", color = "C1", linestyle = ":")
# ax1.plot(range(len(nn_visited_sections_4))[-nn_moving_average_sections_4.size:], nn_moving_average_sections_4, label="with neighbors", color = "C0", linestyle = "--")
# ax1.plot(range(len(nn_visited_sections_8))[-nn_moving_average_sections_8.size:], nn_moving_average_sections_8, label="with neighbors", color = "C1", linestyle = "--")
# ax1.plot(range(len(nn_visited_sections_16))[-nn_moving_average_sections_16.size:], nn_moving_average_sections_16, label="area", color = "C2", linestyle = "--")

# ax1.set_title('Average fraction of visited pipe sections', fontsize=20)
# ax1.plot(range(len(average_visited_sections_1))[-moving_average_average_sections_1.size:], moving_average_average_sections_1, label="1", color = "C0")
# ax1.plot(range(len(average_visited_sections_2))[-moving_average_average_sections_2.size:], moving_average_average_sections_2, label="2", color = "C1")
# ax1.plot(range(len(average_visited_sections_4))[-moving_average_average_sections_4.size:], moving_average_average_sections_4, label="without neighbors", color = "C0")
# ax1.plot(range(len(average_visited_sections_8))[-moving_average_average_sections_8.size:], moving_average_average_sections_8, label="without neighbors", color = "C1")
# ax1.plot(range(len(average_visited_sections_16))[-moving_average_average_sections_16.size:], moving_average_average_sections_16, label="line", color = "C2")
# ax1.plot(range(len(nn_average_visited_sections_2))[-nn_moving_average_average_sections_2.size:], nn_moving_average_average_sections_2, label="2", color = "C1", linestyle = ":")
# ax1.plot(range(len(nn_average_visited_sections_4))[-nn_moving_average_average_sections_4.size:], nn_moving_average_average_sections_4, label="with neighbors", color = "C0", linestyle = "--")
# ax1.plot(range(len(nn_average_visited_sections_8))[-nn_moving_average_average_sections_8.size:], nn_moving_average_average_sections_8, label="with neighbors", color = "C1", linestyle = "--")
# ax1.plot(range(len(nn_average_visited_sections_16))[-nn_moving_average_average_sections_16.size:], nn_moving_average_average_sections_16, label="area", color = "C2", linestyle = "--")

ax1.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim(0, 1)
plt.grid()

# plt.savefig("%s/%.2f_.png" % (directory, episode), bbox_inches='tight')

plt.show()