import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from time import time
from auxiliary_functions import moving_average


def plot_whole_trajectories(n_agents, x_trajectory, y_trajectory, directory, episode, flag_save_in_directory):
    """
    Plots the complete trajectories of agents across one episode.
    """
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title("Whole trajectories")

    for i in range(n_agents):
        ax1.plot(x_trajectory[i], y_trajectory[i], label="%d" % (i + 1), linewidth=0.5)

    lim_x_r = np.max(x_trajectory)
    lim_x_l = np.min(x_trajectory)
    lim_y_r = np.max(y_trajectory)
    lim_y_l = np.min(y_trajectory)
    plt.xlim(lim_x_l, lim_x_r)
    plt.ylim(lim_y_l, lim_y_r)

    ax1.legend(title='Agent: ', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()
    if flag_save_in_directory:
        plt.savefig("%s/whole_trajectories_%d.png" % (directory, episode), bbox_inches='tight')
    else:
        pathlib.Path("./results_multiple_trajectory").mkdir(parents=True, exist_ok=True)
        plt.savefig("./results_multiple_trajectory/whole_trajectories.png", bbox_inches='tight')
    plt.close(fig)


def plot_maximum_distance(maximum_distance_towards_objective):
    """
        Plots the maximum distance reached towards the objective.
    """

    moving_average_distance = moving_average(maximum_distance_towards_objective, 30)

    fig = plt.figure(figsize=(40, 15))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('episode')
    ax1.set_title('Maximum distance reached on the preferred direction of movement')

    ax1.plot(range(maximum_distance_towards_objective.size), maximum_distance_towards_objective)
    ax1.plot(range(len(maximum_distance_towards_objective))[-moving_average_distance.size:], moving_average_distance)
    plt.grid()
    plt.ylim(0, 1)

    plt.show()


def plot_average_highest_reward(average_highest_reward):

    moving_average_reward = moving_average(average_highest_reward, 30)

    fig = plt.figure(figsize=(40, 15))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('episode')
    ax1.set_title('Average highest reward')

    ax1.plot(range(average_highest_reward.size), average_highest_reward)
    ax1.plot(range(len(average_highest_reward))[-moving_average_reward.size:], moving_average_reward)

    plt.grid()
    plt.ylim(-1, 1)

    plt.show()


def plot_fraction_visited_pipes(fraction_of_seen_sections_of_pipe):
    """
        Plots the maximum distance reached towards the objective.
    """

    moving_average_frac = moving_average(fraction_of_seen_sections_of_pipe, 30)

    fig = plt.figure(figsize=(40, 15))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('episode')
    ax1.set_title('Visited visible sections of the pipe')

    ax1.plot(range(fraction_of_seen_sections_of_pipe.size), fraction_of_seen_sections_of_pipe)
    ax1.plot(range(len(fraction_of_seen_sections_of_pipe))[-moving_average_frac.size:], moving_average_frac)

    plt.grid()
    plt.ylim(0, 1)

    plt.show()

def plot_average_fraction_visited_pipes(average_fraction_pipe):
    """
        Plots the maximum distance reached towards the objective.
    """

    moving_average_frac = moving_average(average_fraction_pipe, 30)

    fig = plt.figure(figsize=(40, 15))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('episode')
    ax1.set_title('Averaged visited visible sections of the pipe')

    ax1.plot(range(average_fraction_pipe.size), average_fraction_pipe)
    ax1.plot(range(len(average_fraction_pipe))[-moving_average_frac.size:], moving_average_frac)

    plt.grid()
    plt.ylim(0, 1)

    plt.show()

def plot_Q_matrix_no_neigh_version(state_reward_index, n_agents, Q, k_s):
    titles = ['Sees the pipe', "Close right", "Close left", 'Just lost', "Lost", "Long Lost"]
    fig, axes = plt.subplots(1, n_agents, sharey=True, figsize=(n_agents * 4, 10))
    action_ticks = np.arange(7)
    action_labels = ["%.1f" % (-3 * math.pi / 16), "%.1f" % (-2 * math.pi / 16), "%.1f" % (-math.pi / 16), 0, "%.1f" % (math.pi / 16), "%.1f" % (2 * math.pi / 16), "%.1f" % (3 * math.pi / 16)]
    state_ticks = np.arange(k_s)
    fig.suptitle("Relative position state: %s" % titles[state_reward_index])
    if n_agents <= 1:
        axes.set_xlabel("Actions (rotation applied")
        axes.set_ylabel("Pipe states")
        image = axes.imshow(Q[0, :, state_reward_index])
        maximums = np.argmax(Q[0, :, state_reward_index], axis=1)
        for j in range(k_s):
            axes.text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
        fig.colorbar(image, ax=axes)
        axes.set_xticks(action_ticks)
        axes.set_yticks(state_ticks[:])
        axes.set_xticklabels(action_labels)
        axes.tick_params(labelleft=True)
        axes.tick_params(axis='x', labelsize=9)
    else:
        for i in range(n_agents):
            axes[i].set_title("Agent {}".format(i))
            axes[i].set_xlabel("Actions (rotation applied)")
            axes[i].set_ylabel("Pipe states")
            image = axes[i].imshow(Q[i, :, state_reward_index])
            maximums = np.argmax(Q[i, :, state_reward_index], axis=1)
            for j in range(k_s):
                axes[i].text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
            fig.colorbar(image, ax=axes[i])
            axes[i].set_xticks(action_ticks, labelsize=1)
            axes[i].set_yticks(state_ticks[:])
            axes[i].set_xticklabels(action_labels)
            axes[i].tick_params(labelleft=True)
            axes[i].tick_params(axis='x', labelsize=9)
    plt.show()


def plot_Q_matrix_pipe(state_neighbours_index, state_reward_index, n_agents, Q, k_s):
    """
        Plots the Q matrices of each agent, given a "pipe" state index.
        In this way we can better visualize the values of the state-action matrix in and out of the reward region.
        """
    titles = ['Sees the pipe', "Close right", "Close left", 'Just lost', "Lost", "Long Lost"]
    fig, axes = plt.subplots(1, n_agents, sharey=True, figsize=(n_agents * 4, 10))
    action_ticks = np.arange(7)
    action_labels = ["%.1f" % (-3*math.pi/16), "%.1f" % (-2*math.pi/16), "%.1f" % (-math.pi/16), 0, "%.1f" % (math.pi/16), "%.1f" % (2*math.pi/16), "%.1f" % (3*math.pi/16)]
    state_ticks = np.arange(k_s)
    if state_neighbours_index == 32:
        fig.suptitle("Relative position state: %s \n Neighbors state: no neighbors" % titles[state_reward_index])
    else:
        fig.suptitle("Relative position state: %s \n Neighbors state: %d" % (titles[state_reward_index], state_neighbours_index))

    if n_agents > 1:
        for i in range(n_agents):
            axes[i].set_title("Agent {}".format(i))
            axes[i].set_xlabel("Actions (rotation applied)")
            axes[i].set_ylabel("Pipe states")
            image = axes[i].imshow(Q[i, state_neighbours_index, :-1, state_reward_index])
            maximums = np.argmax(Q[i, state_neighbours_index, :-1, state_reward_index], axis=1)
            for j in range(k_s-1):
                axes[i].text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
            fig.colorbar(image, ax=axes[i])
            axes[i].set_xticks(action_ticks, labelsize = 1)
            axes[i].set_yticks(state_ticks[:-1])
            axes[i].set_xticklabels(action_labels)
            axes[i].tick_params(labelleft = True)
            axes[i].tick_params(axis='x', labelsize = 9)
    else:
        axes.set_title("Agent {}".format(0))
        axes.set_xlabel("Actions (rotation applied)")
        axes.set_ylabel("Pipe states")
        image = axes.imshow(Q[0, state_neighbours_index, :-1, state_reward_index])
        maximums = np.argmax(Q[0, state_neighbours_index, :-1, state_reward_index], axis=1)
        for j in range(k_s-1):
            axes.text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
        fig.colorbar(image, ax=axes)
        axes.set_xticks(action_ticks)
        axes.set_yticks(state_ticks[:-1])
        axes.set_xticklabels(action_labels)
        axes.tick_params(labelleft = True)
        axes.tick_params(axis='x', labelsize = 9)

    plt.show()


def plot_Q_matrix_neigh(state_pipe_index, state_reward_index, n_agents, Q, k_s):
    """
        Plots the Q matrices of each agent, given a "pipe" state index.
        In this way we can better visualize the values of the state-action matrix in and out of the reward region.
        """
    action_ticks = np.arange(7)
    state_ticks = np.arange(k_s)
    titles = ['Sees the pipe', "Close right", "Close left", 'Just lost', "Lost", "Long Lost"]
    action_labels = ["%.1f" % (-3 * math.pi / 16), "%.1f" % (-2 * math.pi / 16), "%.1f" % (-math.pi / 16), 0,
                     "%.1f" % (math.pi / 16), "%.1f" % (2 * math.pi / 16), "%.1f" % (3 * math.pi / 16)]
    fig, axes = plt.subplots(1, n_agents, sharey=True, figsize=(n_agents * 4, 10))
    fig.suptitle("Relative position state: %s \n Pipe state: %d" % (titles[state_reward_index], state_pipe_index))
    if n_agents > 1:
        for i in range(n_agents):
            axes[i].set_title("Agent {}".format(i))
            axes[i].set_xlabel("Actions (rotation applied)")
            axes[i].set_ylabel("Neighbors states")
            image = axes[i].imshow(Q[i, :, state_pipe_index, state_reward_index])
            maximums = np.argmax(Q[i, :, state_pipe_index, state_reward_index], axis=1)
            for j in range(k_s):
                axes[i].text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
            fig.colorbar(image, ax=axes[i])
            axes[i].set_xticks(action_ticks)
            axes[i].set_yticks(state_ticks)
            axes[i].set_xticklabels(action_labels)
            axes[i].tick_params(labelleft=True)
            axes[i].tick_params(axis='x', labelsize=9)
    else:
        axes.set_title("Agent {}".format(0))
        axes.set_xlabel("Actions (rotation applied)")
        axes.set_ylabel("Neighbors states")
        image = axes.imshow(Q[0, :, state_pipe_index, state_reward_index])
        maximums = np.argmax(Q[0, :, state_pipe_index, state_reward_index], axis=1)
        for j in range(k_s):
            axes.text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
        fig.colorbar(image, ax=axes)
        axes.set_xticks(action_ticks)
        axes.set_yticks(state_ticks)
        axes.set_xticklabels(action_labels)
        axes.tick_params(labelleft=True)
        axes.tick_params(axis='x', labelsize=9)

    plt.show()


def plot_policy(k_s, k_s_pipe, arrows_action, Q, Q_visits, agent_index):
    X = np.arange(k_s-1)
    Y = np.arange(k_s)
    titles = ['Sees the pipe', "Close right", "Close left", 'Just lost', "Lost", "Long Lost"]

    tick_labels = ["$-\pi$", " ", " "," "," "," "," "," ","$-\\frac{\pi}{2}$"," "," "," "," "," "," "," ","0"," "," "," "," "," "," "," ","$\\frac{\pi}{2}$"," "," "," "," "," "," ","$\pi-\\frac{\pi}{16}$ ",]
    tick_labels_neigh = ["$-\pi$", " ", " "," "," "," "," "," ","$-\\frac{\pi}{2}$"," "," "," "," "," "," "," ","0"," "," "," "," "," "," "," ","$\\frac{\pi}{2}$"," "," "," "," "," "," ","$\pi-\\frac{\pi}{16}$ ", "no-neigh"]
    for j in range(k_s_pipe):
    # for j in [3]:
        fig = plt.figure(figsize=(25, 25))
        axes = fig.add_subplot(1, 1, 1)

        # axes.set_title("Agent %d, Relative position wrt pipe: %s" % (agent_index, titles[j]), fontsize=20)
        axes.set_xlabel("Pipe state", fontsize = 30)
        axes.set_ylabel("Neighbors state", fontsize = 30)
        # axes.set_xlabel("Neighbors state", fontsize = 20)
        # axes.set_ylabel("Pipe state", fontsize = 20)
        axes.set_xticks(X)
        axes.set_yticks(Y)
        axes.set_xticklabels(tick_labels)
        axes.set_yticklabels(tick_labels_neigh)
        axes.tick_params(axis="x", labelsize=30)
        axes.tick_params(axis="y", labelsize=30)

        greedy_policy = np.argmax(Q[:, :, j, :], axis=2)
        optimal_policy_arrows = np.zeros((k_s, k_s, 2))
        optimal_policy_arrows[:, :] = arrows_action[greedy_policy]
        U, V = optimal_policy_arrows[:, :, 0], optimal_policy_arrows[:, :, 1]
        for i in range(k_s-1):
            for k in range(k_s):
                # if Q_visits[k, i, j] != 0:
                axes.quiver(X[i], Y[k], U[k, i], V[k, i], color='%f' % (1 - Q_visits[k, i, j]), pivot='middle')
                # axes.quiver(X[i], Y[k], U[i, k], V[i, k], color='%f' % (1 - Q_visits[i, k, j]), pivot='middle')

        # plt.grid()
        plt.show()

def plot_policy_no_neigh(k_s, k_s_pipe, arrows_action, Q, Q_visits, agent_index):
    X = np.arange(k_s)
    Y = np.arange(k_s_pipe)
    titles = ['Sees the pipe', "Close right", "Close left", 'Just lost', "Lost", "Long Lost"]
    fig = plt.figure(figsize=(25, 25))
    axes = fig.add_subplot(1, 1, 1)

    axes.set_title("Agent %d" % agent_index, fontsize = 20)
    axes.set_xlabel("Pipe state", fontsize = 20)
    axes.set_ylabel("Relative position wrt pipe", fontsize = 20)
    axes.set_xticks(X)
    axes.set_yticks(Y)
    axes.tick_params(axis="x", labelsize=20)
    axes.tick_params(axis="y", labelsize=20)
    axes.set_yticklabels(titles)

    greedy_policy = np.argmax(Q[:, :, :], axis=2)
    optimal_policy_arrows = np.zeros((k_s, k_s_pipe, 2))
    optimal_policy_arrows[:, :] = arrows_action[greedy_policy]
    U, V = optimal_policy_arrows[:, :, 0], optimal_policy_arrows[:, :, 1]
    for j in range(k_s_pipe):
        for i in range(k_s):
            # if Q_visits[i, j] != 0:
            axes.quiver(X[i], Y[j], U[i, j], V[i, j], color='%f' % (1 - Q_visits[i, j]), pivot='middle')

    plt.show()

def plot_visit_matrices_pipe(state_neighbours_index, state_reward_index, n_agents, V, k_s):
    titles = ['Sees the pipe', "Close right", "Close left", 'Just lost', "Lost", "Long Lost"]
    fig, axes = plt.subplots(1, n_agents, sharey=True, figsize=(n_agents * 4, 10))
    action_ticks = np.arange(7)
    action_labels = ["%.1f" % (-3 * math.pi / 16), "%.1f" % (-2 * math.pi / 16), "%.1f" % (-math.pi / 16), 0,
                     "%.1f" % (math.pi / 16), "%.1f" % (2 * math.pi / 16), "%.1f" % (3 * math.pi / 16)]
    state_ticks = np.arange(k_s)
    if state_neighbours_index == 32:
        fig.suptitle("Relative position state: %s \n Neighbors state: no neighbors" % titles[state_reward_index])
    else:
        fig.suptitle("Relative position state: %s \n Neighbors state: %d" % (titles[state_reward_index], state_neighbours_index))

    if n_agents > 1:
        for i in range(n_agents):
            axes[i].set_title("Agent {}".format(i))
            axes[i].set_xlabel("Actions (rotation applied")
            axes[i].set_ylabel("Pipe states")
            image = axes[i].imshow(V[i, state_neighbours_index, :-1, state_reward_index])
            maximums = np.argmax(V[i, state_neighbours_index, :-1, state_reward_index], axis=1)
            for j in range(k_s-1):
                axes[i].text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
            fig.colorbar(image, ax=axes[i])
            axes[i].set_xticks(action_ticks, labelsize=1)
            axes[i].set_yticks(state_ticks[:-1])
            axes[i].set_xticklabels(action_labels)
            axes[i].tick_params(labelleft=True)
            axes[i].tick_params(axis='x', labelsize=9)
    else:
        axes.set_title("Agent {}".format(0))
        axes.set_xlabel("Actions (rotation applied)")
        axes.set_ylabel("Pipe states")
        image = axes.imshow(V[0, state_neighbours_index, :, state_reward_index])
        maximums = np.argmax(V[0, state_neighbours_index, :, state_reward_index], axis=1)
        for j in range(k_s):
            axes.text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
        fig.colorbar(image, ax=axes)
        axes.set_xticks(action_ticks)
        axes.set_yticks(state_ticks[:-1])
        axes.set_xticklabels(action_labels)
        axes.tick_params(labelleft=True)
        axes.tick_params(axis='x', labelsize=9)

    plt.show()

def plot_visit_matrices_neigh(state_pipe_index, state_reward_index, n_agents, V, k_s):
    action_ticks = np.arange(7)
    state_ticks = np.arange(k_s)
    titles = ['Sees the pipe', "Close right", "Close left", 'Just lost', "Lost", "Long Lost"]
    action_labels = ["%.1f" % (-3 * math.pi / 16), "%.1f" % (-2 * math.pi / 16), "%.1f" % (-math.pi / 16), 0,
                     "%.1f" % (math.pi / 16), "%.1f" % (2 * math.pi / 16), "%.1f" % (3 * math.pi / 16)]
    fig, axes = plt.subplots(1, n_agents, sharey=True, figsize=(n_agents * 4, 10))
    fig.suptitle("Relative position state: %s \n Pipe state: %d" % (titles[state_reward_index], state_pipe_index))
    if n_agents > 1:
        for i in range(n_agents):
            axes[i].set_title("Agent {}".format(i))
            axes[i].set_xlabel("Actions (rotation applied)")
            axes[i].set_ylabel("Neighbors states")
            image = axes[i].imshow(V[i, :-1, state_pipe_index, state_reward_index])
            maximums = np.argmax(V[i, :-1, state_pipe_index, state_reward_index], axis=1)
            for j in range(k_s-1):
                axes[i].text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
            fig.colorbar(image, ax=axes[i])
            axes[i].set_xticks(action_ticks)
            axes[i].set_yticks(state_ticks)
            axes[i].set_xticklabels(action_labels)
            axes[i].tick_params(labelleft=True)
            axes[i].tick_params(axis='x', labelsize=9)
    else:
        axes.set_title("Agent {}".format(0))
        axes.set_xlabel("Actions (rotation applied)")
        axes.set_ylabel("Neighbors states")
        image = axes.imshow(V[0, :, state_pipe_index, state_reward_index])
        maximums = np.argmax(V[0, :, state_pipe_index, state_reward_index], axis=1)
        for j in range(k_s):
            axes.text(maximums[j], j, 'x', ha="center", va="center", color="white", fontsize='small')
        fig.colorbar(image, ax=axes)
        axes.set_xticks(action_ticks)
        axes.set_yticks(state_ticks)
        axes.set_xticklabels(action_labels)
        axes.tick_params(labelleft=True)
        axes.tick_params(axis='x', labelsize=9)

    plt.show()

def plot_Q_matrices(k_s_pipe, n_agents, Q, k_s):
    """
    Auxiliary method to plot the Q matrices in a more compact way.
    """
    for i in range(k_s_pipe):
        for j in [15, 16, 32]:
            plot_Q_matrix_pipe(j, i, n_agents, Q, k_s)
        for j in [15, 16, 17]:
            plot_Q_matrix_neigh(j, i, n_agents, Q, k_s)

def plot_visit_matrices(k_s_pipe, n_agents, V, k_s):
    """
    Auxiliary method to plot the Q matrices in a more compact way.
    """
    for i in range(k_s_pipe):
        for j in [16, 32]:
            plot_visit_matrices_pipe(j, i, n_agents, V, k_s)
        for j in [16, 17]:
            plot_visit_matrices_neigh(j, i, n_agents, V, k_s)

def generate_gif_initial(title, data, directory, flag_single_agent):
    x_traj = data["x_traj"]
    y_traj = data["y_traj"]
    T = x_traj.shape[1]
    n_agents = x_traj.shape[0]
    boolean_array_visibility = data["boolean_array_visibility"]
    vector_fov_starts = data["vector_fov_starts"]
    vector_fov_ends = data["vector_fov_ends"]
    orientation = data["orientation"]
    visibility_of_pipe = data["visibility_of_pipe"]
    colors_agent = data["colors_agent"]
    color_agent = np.empty((n_agents, T), dtype=str)

    for i in range(n_agents):
        for j in range(T):
            if colors_agent[i][j] == 0:
                color_agent[i][j] = 'b'
            elif colors_agent[i][j] == 0.2:
                color_agent[i][j] = 'g'
            elif colors_agent[i][j] == 0.4:
                color_agent[i][j] = 'c'
            else:
                color_agent[i][j] = 'r'

    gif_time = time()
    print("Generating initial trajectory gif.....")
    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
               'b', 'g', 'r', 'c', 'm', 'y', 'orange', 'grey',
               'brown', 'salmon', 'sienna', 'aquamarine', 'darkviolet', 'steelblue', 'darkgoldenrod', 'darkgreen',
               'purple', 'coral', 'yellow', 'yellowgreen', 'teal', 'darkslategray', 'slateblue', 'midnightblue',
               'deepskyblue', 'dodgerblue', 'olive', 'saddlebrown', 'indigo', 'mediumblue', 'pink', 'deeppink']
    fig = plt.figure(figsize=(30, 30))
    camera = Camera(fig)
    # end_point_initial = min(500, T)
    end_point_initial = min(100, T)
    xlim_r = np.max(x_traj[:, 0:end_point_initial])

    plt.xlim([np.min(x_traj[:, 0:end_point_initial]), xlim_r])
    plt.ylim([min(-2, np.min(y_traj[:, 0:end_point_initial])), max(2, np.max(y_traj[:, 0:end_point_initial]))])
    plt.gca().set_aspect('equal', adjustable='box')

    for i in range(0, end_point_initial):
        for k in [x for x in range(0, len(boolean_array_visibility)) if x * 5 < xlim_r]:
            if boolean_array_visibility[k]:
                if not boolean_array_visibility[k-1] and k > 0:
                    plt.plot([k * 5 + 3.5, (k + 1) * 5 + 3.5], [0, 0], 'k')
                # elif k < len(boolean_array_visibility) and not boolean_array_visibility[k+1]:
                #     plt.plot([k * 5, (k + 1) * 5], [0, 0], 'k')
                else:
                    plt.plot([k * 5, (k + 1) * 5 + 3.5], [0, 0], 'k')
        # plt.plot([np.min(x_traj[:, 0:end_point_initial]), xlim_r], [0, 0], 'k')
        for j in range(n_agents):
            plt.plot(x_traj[j][max(0, i - 50):i + 1], y_traj[j][max(0, i - 50):i + 1], colours[j],
                     label="Fish n: %d" % (j + 1))
            # plt.plot(x_traj[j][i], y_traj[j][i], colours[j], marker=(3, 0, orientation[j][i]), markersize=15)
            plt.plot(x_traj[j][i], y_traj[j][i], color_agent[j][i], marker=(3, 0, orientation[j][i]), markersize=15)
            if visibility_of_pipe[j][i]:
                color = 'k'
            else:
                color = 'r'
            plt.arrow(x_traj[j][i], y_traj[j][i], *vector_fov_starts[j][i], color=color, head_width=0, head_length=0)
            plt.arrow(x_traj[j][i], y_traj[j][i], *vector_fov_ends[j][i], color=color, head_width=0, head_length=0)
            plt.plot([x_traj[j][i] + vector_fov_starts[j][i][0], x_traj[j][i] + vector_fov_ends[j][i][0]],
                     [y_traj[j][i] + vector_fov_starts[j][i][1], y_traj[j][i] + vector_fov_ends[j][i][1]], color=color)

        camera.snap()
    animation = camera.animate(blit=True)
    animation.save("%s/%s_initial.gif" % (directory, title), writer="pillow")

    print("Initial trajectory gif ready. Time used: ", time() - gif_time)

def generate_gif_final(title, data, directory, flag_single_agent):
    x_traj = data["x_traj"]
    y_traj = data["y_traj"]
    T = x_traj.shape[1]
    n_agents = x_traj.shape[0]
    boolean_array_visibility = data["boolean_array_visibility"]
    vector_fov_starts = data["vector_fov_starts"]
    vector_fov_ends = data["vector_fov_ends"]
    orientation = data["orientation"]
    visibility_of_pipe = data["visibility_of_pipe"]
    colors_agent = data["colors_agent"]
    color_agent = np.empty((n_agents, T), dtype=str)

    for i in range(n_agents):
        for j in range(T):
            if colors_agent[i][j] == 0:
                color_agent[i][j] = 'b'
            elif colors_agent[i][j] == 0.2:
                color_agent[i][j] = 'g'
            elif colors_agent[i][j] == 0.4:
                color_agent[i][j] = 'c'
            else:
                color_agent[i][j] = 'r'

    colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
               'b', 'g', 'r', 'c', 'm', 'y', 'orange', 'grey',
               'brown', 'salmon', 'sienna', 'aquamarine', 'darkviolet', 'steelblue', 'darkgoldenrod', 'darkgreen',
               'purple', 'coral', 'yellow', 'yellowgreen', 'teal', 'darkslategray', 'slateblue', 'midnightblue',
               'deepskyblue', 'dodgerblue', 'olive', 'saddlebrown', 'indigo', 'mediumblue', 'pink', 'deeppink']

    gif_time = time()
    print("Generating final trajectory gif.....")

    fig = plt.figure(figsize=(30, 30))
    camera = Camera(fig)
    start_point_final = max(T - 500, 0)
    xlim_l = np.min(x_traj[:, start_point_final:T])
    xlim_r = np.max(x_traj[:, start_point_final:T])
    plt.xlim([xlim_l, xlim_r])
    plt.ylim([min(-2, np.min(y_traj[:, start_point_final:T])), max(2, np.max(y_traj[:, start_point_final:T]))])
    plt.gca().set_aspect('equal', adjustable='box')

    for i in range(start_point_final, T):
        for k in [x for x in range(0, len(boolean_array_visibility)) if x * 5 < xlim_r and (x + 1) * 5 > xlim_l]:
            if boolean_array_visibility[k]:
                if not boolean_array_visibility[k - 1] and k > 0:
                    plt.plot([k * 5 + 3.5, (k + 1) * 5 + 3.5], [0, 0], 'k')
                elif k < len(boolean_array_visibility) and not boolean_array_visibility[k + 1]:
                    plt.plot([k * 5, (k + 1) * 5], [0, 0], 'k')
                else:
                    plt.plot([k * 5, (k + 1) * 5 + 3.5], [0, 0], 'k')
        for j in range(n_agents):
            plt.plot(x_traj[j][max(start_point_final, i - 50):i + 1], y_traj[j][max(start_point_final, i - 50):i + 1],
                     colours[j], label="Fish n: %d" % (j + 1))
            # plt.plot(x_traj[j][i], y_traj[j][i], colours[j], marker=(3, 0, orientation[j][i]), markersize=15)
            plt.plot(x_traj[j][i], y_traj[j][i], color_agent[j][i], marker=(3, 0, orientation[j][i]), markersize=15)
            if visibility_of_pipe[j][i]:
                color = 'k'
            else:
                color = 'r'
            plt.arrow(x_traj[j][i], y_traj[j][i], *vector_fov_starts[j][i],color = color, head_width=0, head_length=0)
            plt.arrow(x_traj[j][i], y_traj[j][i], *vector_fov_ends[j][i],color = color, head_width=0, head_length=0)
            plt.plot([x_traj[j][i] + vector_fov_starts[j][i][0], x_traj[j][i] + vector_fov_ends[j][i][0]],
                     [y_traj[j][i] + vector_fov_starts[j][i][1], y_traj[j][i] + vector_fov_ends[j][i][1]], color = color)

        camera.snap()
    animation = camera.animate(blit=True)
    animation.save("%s/%s_final.gif" % (directory, title), writer="pillow")
    print("Final trajectory gif ready. Time used: ", time() - gif_time)

# def generate_gif_final(title, data, directory, flag_single_agent):
#
#     gif_time = time()
#     print("Generating final trajectory gif.....")
#
#     fig = plt.figure(figsize=(30, 30))
#     camera = Camera(fig)
#     start_point_final = max(T - 500, 0)
#     xlim_l = np.min(x_traj[:, start_point_final:T])
#     xlim_r = np.max(x_traj[:, start_point_final:T])
#     plt.xlim([xlim_l, xlim_r])
#     plt.ylim([np.min(y_traj[:, start_point_final:T]), np.max(y_traj[:, start_point_final:T])])
#     plt.gca().set_aspect('equal', adjustable='box')
#
#     for i in range(start_point_final, T):
#         for k in [x for x in range(0, len(boolean_array_visibility)) if x * 5 < xlim_r and (x + 1) * 5 > xlim_l]:
#             if boolean_array_visibility[k]:
#                 plt.plot([k * 5, (k + 1) * 5], [0, 0], 'k')
#         for j in range(n_agents):
#             plt.plot(x_traj[j][max(start_point_final, i - 50):i + 1], y_traj[j][max(start_point_final, i - 50):i + 1],
#                      colours[j], label="Fish n: %d" % (j + 1))
#             plt.plot(x_traj[j][i], y_traj[j][i], colours[j], marker=(3, 0, orientation[j][i]), markersize=15)
#             plt.arrow(x_traj[j][i], y_traj[j][i], *vector_fov_starts[j][i], head_width=0, head_length=0)
#             plt.arrow(x_traj[j][i], y_traj[j][i], *vector_fov_ends[j][i], head_width=0, head_length=0)
#             plt.plot([x_traj[j][i] + vector_fov_starts[j][i][0], x_traj[j][i] + vector_fov_ends[j][i][0]],
#                      [y_traj[j][i] + vector_fov_starts[j][i][1], y_traj[j][i] + vector_fov_ends[j][i][1]], 'k')
#
#         camera.snap()
#     animation = camera.animate(blit=True)
#     animation.save("./trajectories_try_1/%d_agents/%s_final.gif" % (n_agents, title), writer='pillow')
#     print("Final trajectory gif ready. Time used: ", time() - gif_time)

#   Methods for plotting
# def plot_polar_order_parameter(polar_order_parameter, number_of_steps_per_episode):
#     """
#     Plots the polar order parameter.
#     """
#     fig = plt.figure(figsize=(9, 6))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_xlabel('episode')
#     ax1.set_title('Polar order parameter')
#
#     ax1.plot(range(polar_order_parameter.size),
#              np.divide(polar_order_parameter, number_of_steps_per_episode))
#     plt.grid()
#     plt.ylim(0, 1)
#
#     plt.show()

# def plot_frequency_states(current_episode, frequency_state_neighbours, number_of_steps_per_episode, k_s,
#                           frequency_state_pipe, frequency_state_reward_region):
#     """
#     Plots the frequency of occupation of the states by the agents, with matrix plots for clearer visualization.
#     """
#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_title("Frequency of the 'neighbours' state occupation")
#     ax1.set_xlabel("States")
#     ax1.set_ylabel("Agents")
#     plt.imshow(np.divide(frequency_state_neighbours, number_of_steps_per_episode[current_episode]))
#     plt.colorbar()
#     plt.show()
#
#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_title("Frequency of the 'neighbours' state occupation, \n excluding the 'no_neighbours' state")
#     ax1.set_xlabel("States")
#     ax1.set_ylabel("Agents")
#     selector_to_avoid_no_neighbour_state = [x for x in range(frequency_state_neighbours.shape[1]) if
#                                             x != k_s - 1]
#     plt.imshow(np.divide(frequency_state_neighbours[:, selector_to_avoid_no_neighbour_state],
#                          number_of_steps_per_episode[current_episode]))
#     plt.colorbar()
#     plt.show()
#
#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_title("Frequency of the 'pipe' state occupation")
#     ax1.set_xlabel("States")
#     ax1.set_ylabel("Agents")
#     plt.imshow(np.divide(frequency_state_pipe, number_of_steps_per_episode[current_episode]))
#     plt.colorbar()
#     plt.show()
#
#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_title(
#         "Frequency of the 'region' state occupation, \n (0 = below, 1 = in, 2 = over, 3 = no information)")
#     ax1.set_xlabel("States")
#     ax1.set_ylabel("Agents")
#     plt.imshow(np.divide(frequency_state_reward_region, number_of_steps_per_episode[current_episode]))
#     plt.colorbar()
#     plt.show()


# def plot_whole_trajectories_squared(n_agents, x_trajectory, y_trajectory):
#     """
#         Plots the complete trajectories of agents across one episode,
#         changing the scale of the axes in order to have a squared graph.
#         """
#     fig = plt.figure(figsize=(20, 20))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_title("Whole trajectories, with the correct scale to avoid confusion")
#
#     for i in range(n_agents):
#         ax1.plot(x_trajectory[i], y_trajectory[i], label="%d" % (i + 1), linewidth=0.5)
#
#     lim_x_r = np.max(x_trajectory)
#     lim_x_l = np.min(x_trajectory)
#     lim_y_r = np.max(y_trajectory)
#     lim_y_l = np.min(y_trajectory)
#     lim_l = np.min(np.array([lim_x_l, lim_y_l]))
#     lim_r = np.max(np.array([lim_x_r, lim_y_r]))
#     plt.xlim(lim_l, lim_r)
#     plt.ylim(lim_l, lim_r)
#
#     ax1.legend(title='Agent: ', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
#
#     plt.show()


# def plot_whole_trajectories_reward_region(n_agents, x_trajectory, y_trajectory, radius):
#     """
#         Plots the trajectories of agents across one episode, focusing on only the "pipe" region.
#         """
#     fig = plt.figure(figsize=(20, 6))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_title("Whole trajectories, rescaled")
#
#     for i in range(n_agents):
#         ax1.plot(x_trajectory[i], y_trajectory[i], label="%d" % (i + 1), linewidth=0.5)
#
#     lim_x_r = np.max(x_trajectory)
#     lim_x_l = np.min(x_trajectory)
#     plt.xlim(lim_x_l, lim_x_r)
#     plt.ylim(-radius, radius)
#
#     ax1.legend(title='Agent: ', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
#
#     plt.show()

# def plot_average_reward(current_episode, rewards, n_agents):
#     """
#         Plots the reward for episode current_episode, averaged across agents, and its moving average
#         """
#     fig = plt.figure(figsize=(9, 6))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_xlabel('t')
#     ax1.set_ylabel('reward')
#     ax1.set_title('Average reward in episode %d' % current_episode)
#
#     average_reward = np.sum(rewards, axis=0) / n_agents
#     if average_reward.size < 50:
#         n = 1
#     else:
#         n = 50
#     moving_average_reward = moving_average(average_reward, n)
#
#     ax1.plot(range(average_reward.size), average_reward, label="Average reward")
#     ax1.plot(range(average_reward.size)[-moving_average_reward.size:], moving_average_reward,
#              label="Moving average reward")
#     ax1.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
#
#     plt.ylim(-1, 1)
#     plt.grid()
#
#     plt.show()


# def plot_single_moving_average_rewards(current_episode, number_of_steps_per_episode, n_agents, rewards):
#     """
#         Plots the moving average rewards of each agent for episode current_episode.
#         """
#     t_space = np.array(np.linspace(0, number_of_steps_per_episode[current_episode] - 1,
#                                    number_of_steps_per_episode[current_episode]))
#
#     fig = plt.figure(figsize=(9, 6))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_xlabel('t')
#     ax1.set_title('Moving average reward for the single agents in episode %d' % current_episode)
#
#     for i in range(n_agents):
#         if rewards[i].size < 50:
#             n = 1
#         else:
#             n = 50
#         moving_average_reward_single = moving_average(rewards[i], n)
#         ax1.plot(t_space[-moving_average_reward_single.size:], moving_average_reward_single, label="%d" % (i + 1))
#
#     ax1.legend(title='Agent: ', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.ylim(-1, 1)
#
#     plt.grid()
#
#     plt.show()


# def plot_cumulative_sum_rewards(current_episode, number_of_steps_per_episode, rewards, gamma):
#     t_space = np.array(np.linspace(0, number_of_steps_per_episode[current_episode] - 1,
#                                    number_of_steps_per_episode[current_episode]))
#
#     fig = plt.figure(figsize=(9, 6))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_xlabel('t')
#     ax1.set_title('Cumulative sum of the rewards in episode %d' % current_episode)
#
#     sum_of_rewards = np.sum(rewards, axis=0)
#     cum_sum_rewards = np.cumsum(sum_of_rewards)
#     ax1.plot(range(cum_sum_rewards.size), cum_sum_rewards, label="Cumulative sum")
#
#     for i in range(cum_sum_rewards.size):
#         sum_of_rewards[i] *= gamma ** t_space[i]
#
#     cum_sum_rewards = np.cumsum(sum_of_rewards)
#     ax1.plot(range(cum_sum_rewards.size), cum_sum_rewards, label="Discounted cumulative sum")
#
#     ax1.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
#
#     plt.grid()
#
#     plt.show()


# def plot_average_reward_episodes(epochs_rewards, n_agents, number_of_steps_per_episode):
#     """
#         Plots the average reward for all episodes.
#         The value of a single episode is the reward averaged across all agents and all timesteps of the episode.
#         """
#     fig = plt.figure(figsize=(9, 6))
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_xlabel('t')
#     ax1.set_ylabel('reward')
#     ax1.set_title('Average reward on episodes done so far')
#
#     average_reward_epochs = np.divide(np.sum(epochs_rewards, axis=0) / n_agents, number_of_steps_per_episode)
#     epochs_range = range(average_reward_epochs.size)
#     if average_reward_epochs.size < 50:
#         n = 1
#     else:
#         n = 50
#     moving_average_reward_epochs = moving_average(average_reward_epochs, n)
#
#     ax1.plot(range(average_reward_epochs.size), average_reward_epochs, label="Average reward")
#     ax1.plot(epochs_range[-moving_average_reward_epochs.size:], moving_average_reward_epochs,
#              label="Moving average reward")
#
#     ax1.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.ylim(-1, 1)
#
#     plt.grid()
#
#     plt.show()


# def plot_initial_trajectories(timesteps, n_agents, x_trajectory, y_trajectory):
#     """
#     Plots the trajectories of agents for the first timesteps of an episode.
#     """
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('y')
#     ax1.set_title("Trajectories of the first %d timesteps" % timesteps)
#
#     for i in range(n_agents):
#         ax1.plot(x_trajectory[i][0:timesteps], y_trajectory[i][0:timesteps], label="%d" % (i + 1))
#
#     lim_x_r = np.max(x_trajectory[:, 0:timesteps])
#     lim_x_l = np.min(x_trajectory[:, 0:timesteps])
#     lim_y_r = np.max(y_trajectory[:, 0:timesteps])
#     lim_y_l = np.min(y_trajectory[:, 0:timesteps])
#     lim_l = np.min(np.array([lim_x_l, lim_y_l]))
#     lim_r = np.max(np.array([lim_x_r, lim_y_r]))
#     plt.xlim(lim_l, lim_r)
#     plt.ylim(lim_l, lim_r)
#
#     ax1.legend(title='Agent: ', fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))
#
#     plt.show()


# def plot_final_trajectories(timesteps, n_agents, x_trajectory, y_trajectory):
#     """
#         Plots the trajectories of agents for the last timesteps of an episode, separately
#         """
#     fig, axes = plt.subplots(1, n_agents, figsize=(20, 5))
#     fig.suptitle("Trajectories of the last %d timesteps for different agents" % timesteps)
#     for i in range(n_agents):
#         axes[i].set_title("Agent {}".format(i))
#         axes[i].set_xlabel("x")
#         axes[i].set_ylabel("y")
#         for j in range(n_agents):
#             axes[i].plot(x_trajectory[j][-timesteps:], y_trajectory[j][-timesteps:], label="%d" % (j + 1))
#         lim_x_r = np.max(x_trajectory[i, -timesteps:])
#         lim_x_l = np.min(x_trajectory[i, -timesteps:])
#         lim_y_r = np.max(y_trajectory[i, -timesteps:])
#         lim_y_l = np.min(y_trajectory[i, -timesteps:])
#         axes[i].set_xlim(lim_x_l, lim_x_r)
#         axes[i].set_ylim(lim_y_l, lim_y_r)
#
#     plt.show()
