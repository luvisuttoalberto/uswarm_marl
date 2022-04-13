import numpy as np
from auxiliary_functions import moving_average
import matplotlib.pyplot as plt

pipe_recognition_probability = 0.25

data_neigh_1_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 1)
visited_sections_1_nn = data_neigh_1_nn["fraction_of_seen_sections_of_pipe"]
moving_average_sections_1_nn = moving_average(visited_sections_1_nn, 30)

data_neigh_2_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 2)
visited_sections_2_nn = data_neigh_2_nn["fraction_of_seen_sections_of_pipe"]
moving_average_sections_2_nn = moving_average(visited_sections_2_nn, 30)

# data_neigh_3_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 3)
# visited_sections_3_nn = data_neigh_3_nn["fraction_of_seen_sections_of_pipe"]
# moving_average_sections_3_nn = moving_average(visited_sections_3_nn, 30)

data_neigh_4_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 4)
visited_sections_4_nn = data_neigh_4_nn["fraction_of_seen_sections_of_pipe"]
moving_average_sections_4_nn = moving_average(visited_sections_4_nn, 30)

# data_neigh_5_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 5)
# visited_sections_5_nn = data_neigh_5_nn["fraction_of_seen_sections_of_pipe"]
# moving_average_sections_5_nn = moving_average(visited_sections_5_nn, 30)
#
# data_neigh_6_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 6)
# visited_sections_6_nn = data_neigh_6_nn["fraction_of_seen_sections_of_pipe"]
# moving_average_sections_6_nn = moving_average(visited_sections_6_nn, 30)
#
# data_neigh_7_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 7)
# visited_sections_7_nn = data_neigh_7_nn["fraction_of_seen_sections_of_pipe"]
# moving_average_sections_7_nn = moving_average(visited_sections_7_nn, 30)

data_neigh_8_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 8)
visited_sections_8_nn = data_neigh_8_nn["fraction_of_seen_sections_of_pipe"]
moving_average_sections_8_nn = moving_average(visited_sections_8_nn, 30)

data_neigh_1 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 1)
visited_sections_1 = data_neigh_1["fraction_of_seen_sections_of_pipe"]
moving_average_sections_1 = moving_average(visited_sections_1, 30)

data_neigh_2 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 2)
visited_sections_2 = data_neigh_2["fraction_of_seen_sections_of_pipe"]
moving_average_sections_2 = moving_average(visited_sections_2, 30)

# data_neigh_3 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 3)
# visited_sections_3 = data_neigh_3["fraction_of_seen_sections_of_pipe"]
# moving_average_sections_3 = moving_average(visited_sections_3, 30)

data_neigh_4 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 4)
visited_sections_4 = data_neigh_4["fraction_of_seen_sections_of_pipe"]
moving_average_sections_4 = moving_average(visited_sections_4, 30)

# data_neigh_5 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 5)
# visited_sections_5 = data_neigh_5["fraction_of_seen_sections_of_pipe"]
# moving_average_sections_5 = moving_average(visited_sections_5, 30)
#
# data_neigh_6 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 6)
# visited_sections_6 = data_neigh_6["fraction_of_seen_sections_of_pipe"]
# moving_average_sections_6 = moving_average(visited_sections_6, 30)
#
# data_neigh_7 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 7)
# visited_sections_7 = data_neigh_7["fraction_of_seen_sections_of_pipe"]
# moving_average_sections_7 = moving_average(visited_sections_7, 30)

data_neigh_8 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 8)
visited_sections_8 = data_neigh_8["fraction_of_seen_sections_of_pipe"]
moving_average_sections_8 = moving_average(visited_sections_8, 30)

fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)
ax1.set_title('Moving average of the fraction of visited pipe sections', fontsize=20)

ax1.tick_params(axis="x", labelsize=20)
ax1.tick_params(axis="y", labelsize=20)

ax1.plot(range(len(visited_sections_1_nn))[-moving_average_sections_1_nn.size:], moving_average_sections_1_nn, label="1", linestyle = "--")
ax1.plot(range(len(visited_sections_2_nn))[-moving_average_sections_2_nn.size:], moving_average_sections_2_nn, label="2", linestyle = "--")
# ax1.plot(range(len(visited_sections_3_nn))[-moving_average_sections_3_nn.size:], moving_average_sections_3_nn, label="3", linestyle = "--)
ax1.plot(range(len(visited_sections_4_nn))[-moving_average_sections_4_nn.size:], moving_average_sections_4_nn, label="4", linestyle = "--")
# ax1.plot(range(len(visited_sections_5_nn))[-moving_average_sections_5_nn.size:], moving_average_sections_5_nn, label="5", linestyle = "--)
# ax1.plot(range(len(visited_sections_6_nn))[-moving_average_sections_6_nn.size:], moving_average_sections_6_nn, label="6", linestyle = "--")
# ax1.plot(range(len(visited_sections_7_nn))[-moving_average_sections_7_nn.size:], moving_average_sections_7_nn, label="7", linestyle = "--)
ax1.plot(range(len(visited_sections_8_nn))[-moving_average_sections_8_nn.size:], moving_average_sections_8_nn, label="8", linestyle = "--")

ax1.plot(range(len(visited_sections_1))[-moving_average_sections_1.size:], moving_average_sections_1, label="1", color = "C0")
ax1.plot(range(len(visited_sections_2))[-moving_average_sections_2.size:], moving_average_sections_2, label="2", color = "C1")
# ax1.plot(range(len(visited_sections_3))[-moving_average_sections_3.size:], moving_average_sections_3, label="3",, color = "C0 linestyle = "--)
ax1.plot(range(len(visited_sections_4))[-moving_average_sections_4.size:], moving_average_sections_4, label="4", color = "C2")
# ax1.plot(range(len(visited_sections_5))[-moving_average_sections_5.size:], moving_average_sections_5, label="5",, color = "C0 linestyle = "--)
# ax1.plot(range(len(visited_sections_6))[-moving_average_sections_6.size:], moving_average_sections_6, label="6"), color = "C0
# ax1.plot(range(len(visited_sections_7))[-moving_average_sections_7.size:], moving_average_sections_7, label="7",, color = "C0 linestyle = "--)
ax1.plot(range(len(visited_sections_8))[-moving_average_sections_8.size:], moving_average_sections_8, label="8", color = "C3")


ax1.legend(fontsize=20, loc='center left', title='Agents:', bbox_to_anchor=(1, 0.5))

plt.ylim(0, 1)
plt.grid()

plt.show()

data_neigh_1_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 1)
visited_sections_1_nn = data_neigh_1_nn["average_highest_reward"]
moving_average_sections_1_nn = moving_average(visited_sections_1_nn, 30)

data_neigh_2_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 2)
visited_sections_2_nn = data_neigh_2_nn["average_highest_reward"]
moving_average_sections_2_nn = moving_average(visited_sections_2_nn, 30)

data_neigh_4_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 4)
visited_sections_4_nn = data_neigh_4_nn["average_highest_reward"]
moving_average_sections_4_nn = moving_average(visited_sections_4_nn, 30)

data_neigh_8_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 8)
visited_sections_8_nn = data_neigh_8_nn["average_highest_reward"]
moving_average_sections_8_nn = moving_average(visited_sections_8_nn, 30)

data_neigh_1 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 1)
visited_sections_1 = data_neigh_1["average_highest_reward"]
moving_average_sections_1 = moving_average(visited_sections_1, 30)

data_neigh_2 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 2)
visited_sections_2 = data_neigh_2["average_highest_reward"]
moving_average_sections_2 = moving_average(visited_sections_2, 30)

data_neigh_4 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 4)
visited_sections_4 = data_neigh_4["average_highest_reward"]
moving_average_sections_4 = moving_average(visited_sections_4, 30)

data_neigh_8 = np.load("./data/benchmark_probability_recognition_noise_on_pipe/%d_agents/data_for_plots.npz" % 8)
visited_sections_8 = data_neigh_8["average_highest_reward"]
moving_average_sections_8 = moving_average(visited_sections_8, 30)

fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)
ax1.set_title('Moving average of the average highest reward', fontsize=20)

ax1.tick_params(axis="x", labelsize=20)
ax1.tick_params(axis="y", labelsize=20)

ax1.plot(range(len(visited_sections_1_nn))[-moving_average_sections_1_nn.size:], moving_average_sections_1_nn, label="1", linestyle = "--")
ax1.plot(range(len(visited_sections_2_nn))[-moving_average_sections_2_nn.size:], moving_average_sections_2_nn, label="2", linestyle = "--")
ax1.plot(range(len(visited_sections_4_nn))[-moving_average_sections_4_nn.size:], moving_average_sections_4_nn, label="4", linestyle = "--")
ax1.plot(range(len(visited_sections_8_nn))[-moving_average_sections_8_nn.size:], moving_average_sections_8_nn, label="8", linestyle = "--")

ax1.plot(range(len(visited_sections_1))[-moving_average_sections_1.size:], moving_average_sections_1, label="1", color = "C0")
ax1.plot(range(len(visited_sections_2))[-moving_average_sections_2.size:], moving_average_sections_2, label="2", color = "C1")
ax1.plot(range(len(visited_sections_4))[-moving_average_sections_4.size:], moving_average_sections_4, label="4", color = "C2")
ax1.plot(range(len(visited_sections_8))[-moving_average_sections_8.size:], moving_average_sections_8, label="8", color = "C3")


ax1.legend(fontsize=20, loc='center left', title='Agents:', bbox_to_anchor=(1, 0.5))

plt.ylim(0, 1)
plt.grid()

plt.show()



data_neigh_1_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 1)
visited_sections_1_nn = data_neigh_1_nn["maximum_distance_towards_objective"]
moving_average_sections_1_nn = moving_average(visited_sections_1_nn, 30)

data_neigh_2_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 2)
visited_sections_2_nn = data_neigh_2_nn["maximum_distance_towards_objective"]
moving_average_sections_2_nn = moving_average(visited_sections_2_nn, 30)

# data_neigh_3_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 3)
# visited_sections_3_nn = data_neigh_3_nn["maximum_distance_towards_objective"]
# moving_average_sections_3_nn = moving_average(visited_sections_3_nn, 30)

data_neigh_4_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 4)
visited_sections_4_nn = data_neigh_4_nn["maximum_distance_towards_objective"]
moving_average_sections_4_nn = moving_average(visited_sections_4_nn, 30)

# data_neigh_5_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 5)
# visited_sections_5_nn = data_neigh_5_nn["maximum_distance_towards_objective"]
# moving_average_sections_5_nn = moving_average(visited_sections_5_nn, 30)
#
# data_neigh_6_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 6)
# visited_sections_6_nn = data_neigh_6_nn["maximum_distance_towards_objective"]
# moving_average_sections_6_nn = moving_average(visited_sections_6_nn, 30)
#
# data_neigh_7_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 7)
# visited_sections_7_nn = data_neigh_7_nn["maximum_distance_towards_objective"]
# moving_average_sections_7_nn = moving_average(visited_sections_7_nn, 30)

data_neigh_8_nn = np.load("./data/benchmark_probability_recognition_noise_on_pipe_nn/%d_agents/data_for_plots.npz" % 8)
visited_sections_8_nn = data_neigh_8_nn["maximum_distance_towards_objective"]
moving_average_sections_8_nn = moving_average(visited_sections_8_nn, 30)


fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)
ax1.set_title('Moving average of the maximum distance towards the objective', fontsize=20)

ax1.tick_params(axis="x", labelsize=20)
ax1.tick_params(axis="y", labelsize=20)

ax1.plot(range(len(visited_sections_1_nn))[-moving_average_sections_1_nn.size:], moving_average_sections_1_nn, label="1", linestyle = "--")
ax1.plot(range(len(visited_sections_2_nn))[-moving_average_sections_2_nn.size:], moving_average_sections_2_nn, label="2", linestyle = "--")
# ax1.plot(range(len(visited_sections_3_nn))[-moving_average_sections_3_nn.size:], moving_average_sections_3_nn, label="3", linestyle = "--)
ax1.plot(range(len(visited_sections_4_nn))[-moving_average_sections_4_nn.size:], moving_average_sections_4_nn, label="4", linestyle = "--")
# ax1.plot(range(len(visited_sections_5_nn))[-moving_average_sections_5_nn.size:], moving_average_sections_5_nn, label="5", linestyle = "--)
# ax1.plot(range(len(visited_sections_6_nn))[-moving_average_sections_6_nn.size:], moving_average_sections_6_nn, label="6", linestyle = "--")
# ax1.plot(range(len(visited_sections_7_nn))[-moving_average_sections_7_nn.size:], moving_average_sections_7_nn, label="7", linestyle = "--)
ax1.plot(range(len(visited_sections_8_nn))[-moving_average_sections_8_nn.size:], moving_average_sections_8_nn, label="8", linestyle = "--")

ax1.legend(fontsize=20, loc='center left', title='Agents:', bbox_to_anchor=(1, 0.5))

plt.ylim(0, 1)
plt.grid()

plt.show()

