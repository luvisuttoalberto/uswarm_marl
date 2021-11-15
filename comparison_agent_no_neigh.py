import numpy as np
from auxiliary_functions import moving_average
import matplotlib.pyplot as plt

n_agents = 2

data_neigh_2_1_5 = np.load("./data/pipe_neigh_rand_sampled/%d_agents/data_for_plots.npz" % n_agents)

data_no_neigh_2_1_5 = np.load("./data/pipe_no_neigh_rand_sampled/%d_agents/data_for_plots.npz" % n_agents)

data_neigh_2_4 = np.load("./data/pipe_neigh_rand_sampled_R_4/%d_agents/data_for_plots.npz" % n_agents)

data_no_neigh_2_4 = np.load("./data/pipe_no_neigh_rand_sampled_R_4/%d_agents/data_for_plots.npz" % n_agents)

n_agents = 8

data_neigh_8_1_5 = np.load("./data/pipe_neigh_rand_sampled/%d_agents/data_for_plots.npz" % n_agents)

data_no_neigh_8_1_5 = np.load("./data/pipe_no_neigh_rand_sampled/%d_agents/data_for_plots.npz" % n_agents)

data_neigh_8_4 = np.load("./data/pipe_neigh_rand_sampled_R_4/%d_agents/data_for_plots.npz" % n_agents)

data_no_neigh_8_4 = np.load("./data/pipe_no_neigh_rand_sampled_R_4/%d_agents/data_for_plots.npz" % n_agents)

n_agents = 4

data_neigh_4_4_eps = np.load("./data/pipe_neigh_rand_sampled_eps_3/%d_agents/data_for_plots.npz" % n_agents)

data_neigh_4_1_5 = np.load("./data/pipe_neigh_rand_sampled/%d_agents/data_for_plots.npz" % n_agents)

data_no_neigh_4_1_5 = np.load("./data/pipe_no_neigh_rand_sampled/%d_agents/data_for_plots.npz" % n_agents)

data_no_neigh_4_4 = np.load("./data/pipe_no_neigh_rand_sampled_R_4/%d_agents/data_for_plots.npz" % n_agents)

data_neigh_4_4 = np.load("./data/pipe_neigh_rand_sampled_R_4/%d_agents/data_for_plots.npz" % n_agents)

visited_section_neigh_2_1_5 = data_neigh_2_1_5["average_highest_reward"]
visited_section_neigh_2_4 = data_neigh_2_4["average_highest_reward"]
visited_section_neigh_4_1_5 = data_neigh_4_1_5["average_highest_reward"]
visited_section_neigh_4_4 = data_neigh_4_4["average_highest_reward"]
visited_section_neigh_8_1_5 = data_neigh_8_1_5["average_highest_reward"]
visited_section_neigh_8_4 = data_neigh_8_4["average_highest_reward"]
visited_section_no_neigh_2_1_5 = data_no_neigh_2_1_5["average_highest_reward"]
visited_section_no_neigh_2_4 = data_no_neigh_2_4["average_highest_reward"]
visited_section_no_neigh_4_1_5 = data_no_neigh_4_1_5["average_highest_reward"]
visited_section_no_neigh_4_4 = data_no_neigh_4_4["average_highest_reward"]
visited_section_no_neigh_8_1_5 = data_no_neigh_8_1_5["average_highest_reward"]
visited_section_no_neigh_8_4 = data_no_neigh_8_4["average_highest_reward"]
visited_section_neigh_4_4_eps = data_neigh_4_4_eps["average_highest_reward"]

moving_average_neigh_2_1_5 = moving_average(visited_section_neigh_2_1_5, 30)
moving_average_neigh_2_4 = moving_average(visited_section_neigh_2_4, 30)
moving_average_neigh_4_1_5 = moving_average(visited_section_neigh_4_1_5, 30)
moving_average_neigh_4_4 = moving_average(visited_section_neigh_4_4, 30)
moving_average_neigh_8_1_5 = moving_average(visited_section_neigh_8_1_5, 30)
moving_average_neigh_8_4 = moving_average(visited_section_neigh_8_4, 30)
moving_average_no_neigh_2_1_5 = moving_average(visited_section_no_neigh_2_1_5, 30)
moving_average_no_neigh_2_4 = moving_average(visited_section_no_neigh_2_4, 30)
moving_average_no_neigh_4_1_5 = moving_average(visited_section_no_neigh_4_1_5, 30)
moving_average_no_neigh_4_4 = moving_average(visited_section_no_neigh_4_4, 30)
moving_average_no_neigh_8_1_5 = moving_average(visited_section_no_neigh_8_1_5, 30)
moving_average_no_neigh_8_4 = moving_average(visited_section_no_neigh_8_4, 30)
moving_average_neigh_4_4_eps = moving_average(visited_section_neigh_4_4_eps, 30)

fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)
ax1.set_title('Moving average of the maximum reward', fontsize=20)

ax1.plot(range(len(visited_section_neigh_2_1_5))[-moving_average_neigh_2_1_5.size:], moving_average_neigh_2_1_5, 'C0-', label="R=1.5 \n n=2")
ax1.plot(range(len(visited_section_neigh_2_4))[-moving_average_neigh_2_4.size:], moving_average_neigh_2_4, 'C0:', label="R=4 \n n=2")
ax1.plot(range(len(visited_section_neigh_4_1_5))[-moving_average_neigh_4_1_5.size:], moving_average_neigh_4_1_5, 'C1-', label="R=1.5 \n n=4")
ax1.plot(range(len(visited_section_neigh_4_4))[-moving_average_neigh_4_4.size:], moving_average_neigh_4_4, 'C1:', label="R=4 \n n=4")
ax1.plot(range(len(visited_section_neigh_8_1_5))[-moving_average_neigh_8_1_5.size:], moving_average_neigh_8_1_5, 'C2-', label="R=1.5 \n n=8")
ax1.plot(range(len(visited_section_neigh_8_4))[-moving_average_neigh_8_4.size:], moving_average_neigh_8_4, 'C2:', label="R=4 \n n=8")
ax1.plot(range(len(visited_section_no_neigh_2_1_5))[-moving_average_no_neigh_2_1_5.size:], moving_average_no_neigh_2_1_5, 'C0--', label="R=1.5 \n n=2 \n no neigh")
ax1.plot(range(len(visited_section_no_neigh_2_4))[-moving_average_no_neigh_2_4.size:], moving_average_no_neigh_2_4, 'C0-.', label="R=4 \n n=2 \n no neigh")
ax1.plot(range(len(visited_section_no_neigh_4_1_5))[-moving_average_no_neigh_4_1_5.size:], moving_average_no_neigh_4_1_5, 'C1--', label="R=1.5 \n n=4 \n no neigh")
ax1.plot(range(len(visited_section_no_neigh_4_4))[-moving_average_no_neigh_4_4.size:], moving_average_no_neigh_4_4, 'C1-.', label="R=4 \n n=4 \n no neigh")
ax1.plot(range(len(visited_section_no_neigh_8_1_5))[-moving_average_no_neigh_8_1_5.size:], moving_average_no_neigh_8_1_5, 'C2--', label="R=1.5 \n n=8 \n no neigh")
ax1.plot(range(len(visited_section_no_neigh_8_4))[-moving_average_no_neigh_8_4.size:], moving_average_no_neigh_8_4, 'C2-.', label="R=4 \n n=8 \n no neigh")

# ax1.plot(range(len(visited_section_neigh_4_4_eps))[-moving_average_neigh_4_4_eps.size:], moving_average_neigh_4_4_eps, 'C4-.', label="eps=0.3, R=4 \n n=4")


ax1.legend(fontsize=14, loc='center left', title='Agents:', bbox_to_anchor=(1, 0.5))

plt.ylim(0, 1)
plt.grid()

plt.show()

fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)
ax1.set_title('Moving average of the maximum reward, R = 4, n_agents = 4, different exploration rates', fontsize=20)

ax1.plot(range(len(visited_section_neigh_4_4))[-moving_average_neigh_4_4.size:], moving_average_neigh_4_4, label="0.5")
ax1.plot(range(len(visited_section_neigh_4_4_eps))[-moving_average_neigh_4_4_eps.size:], moving_average_neigh_4_4_eps, label="0.3")

ax1.legend(fontsize=14, loc='center left', title='epsilon_0:', bbox_to_anchor=(1, 0.5))

plt.ylim(0, 1)
plt.grid()

plt.show()
