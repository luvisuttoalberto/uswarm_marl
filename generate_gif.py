from numpy import load
from plot_functions import generate_gifs, plot_whole_trajectories

n_agents = 4
episode_to_be_plotted = 100
directory = "./data_pipe_neigh_rand_sampled"
data_for_gif = load("%s/%d_agents/episode_%d.npz" % (directory, n_agents, episode_to_be_plotted))

plot_whole_trajectories(n_agents, data_for_gif["x_traj"], data_for_gif["y_traj"], directory, episode_to_be_plotted)

generate_gifs("episode_%d" % episode_to_be_plotted, data_for_gif, directory)
