import numpy as np
from auxiliary_functions import moving_average
import matplotlib.pyplot as plt

n_agents = 8

data_neigh_8 = np.load("./trajectories_try_punishment_for_losing_pipe/%d_agents/data_for_plots.npz" % n_agents)

data_no_neigh_8 = np.load("./trajectories_try_no_neigh/%d_agents/data_for_plots.npz" % n_agents)

n_agents = 4

data_neigh_4 = np.load("./trajectories_try_punishment_for_losing_pipe/%d_agents/data_for_plots.npz" % n_agents)

data_no_neigh_4 = np.load("./trajectories_try_no_neigh/%d_agents/data_for_plots.npz" % n_agents)

visited_section_neigh_4 = data_neigh_4["fraction_of_seen_sections_of_pipe"]
visited_section_no_neigh_4 = data_no_neigh_4["fraction_of_seen_sections_of_pipe"]

visited_section_neigh_8 = data_neigh_8["fraction_of_seen_sections_of_pipe"]
visited_section_no_neigh_8 = data_no_neigh_8["fraction_of_seen_sections_of_pipe"]

moving_average_neigh_4 = moving_average(visited_section_neigh_4, 30)
moving_average_no_neigh_4 = moving_average(visited_section_no_neigh_4, 30)

moving_average_neigh_8 = moving_average(visited_section_neigh_8, 30)
moving_average_no_neigh_8 = moving_average(visited_section_no_neigh_8, 30)

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode')
ax1.set_title('Moving average on 30 steps of the fraction of visited visible sections of the pipe')

ax1.plot(range(len(visited_section_neigh_4))[-moving_average_neigh_4.size:], moving_average_neigh_4,'C0', label="4, neigh")
ax1.plot(range(len(visited_section_no_neigh_4))[-moving_average_no_neigh_4.size:], moving_average_no_neigh_4, 'C0--',
         label="4, no neigh")

ax1.plot(range(len(visited_section_neigh_8))[-moving_average_neigh_8.size:], moving_average_neigh_8, 'C1', label="8, neigh")
ax1.plot(range(len(visited_section_no_neigh_8))[-moving_average_no_neigh_8.size:], moving_average_no_neigh_8, 'C1--',
         label="8, no neigh")

ax1.legend(fontsize=14, loc='center left', title='Agents:', bbox_to_anchor=(1, 0.5))

plt.ylim(0, 1)
plt.grid()

plt.show()
