import numpy as np
from auxiliary_functions import \
    compute_rotation_matrix, learning_rate_adaptive, \
    exploration_rate_adaptive, is_scalar_in_visible_interval
from math import sqrt, degrees, pi, floor
from agent import Agent
from numpy.linalg import norm as euclidean_norm
import random
from time import time


class HiddenPipeEnvironment:
    """
    Class that defines the whole environment.
    """

    def __init__(self,
                 theta_max,
                 v0,
                 radius,
                 k_s,
                 k_s_pipe,
                 k_a,
                 alpha_0,
                 phi,
                 n_episodes,
                 t_star_epsilon,
                 t_star_lr,
                 epsilon_0,
                 slope_pipe,
                 offset_pipe,
                 mean_velocity_noise,
                 mean_position_noise,
                 std_dev_velocity_noise,
                 std_dev_position_noise,
                 reset_type,
                 gamma):
        """
        Constructor of the class.
        """

        # The number of agents is initialized to 0; then agents are added to the environment (and to the agents list)
        # one at a time
        self.n_agents = 0
        self.agents_list = []

        self.theta_max = theta_max
        self.v0 = v0
        self.R = radius

        # States
        self.K_s = k_s
        self.K_s_pipe = k_s_pipe

        # Initialization of the vector of possible states (Discretization of the states)
        self.possible_states = [x for x in np.linspace(-pi, pi, k_s)]
        # Giving a value to the state in which there are no neighbours
        self.no_neighbours_state = 100.
        # Appending the "no neighbours" state to the previously initialized vector
        self.possible_states[k_s - 1] = self.no_neighbours_state

        # This auxiliary vector of states is used in order to assign states close to -pi and pi to the same state
        self.extended_states = [x for x in np.linspace(-pi, pi, k_s)]
        self.extended_states[k_s - 1] = self.no_neighbours_state
        # Extending the vector of possible states by appending pi
        self.extended_states.append(pi)

        # Actions
        self.K_a = k_a

        # Initialization of the vector of possible actions (Discretization of the actions)
        self.possible_actions = [x for x in np.linspace(-self.theta_max, self.theta_max, self.K_a)]

        # Initialization of previously explained parameters
        self.alpha_0 = alpha_0
        self.phi = phi
        # self.T = T
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.t_star_epsilon = t_star_epsilon
        self.t_star_lr = t_star_lr
        self.epsilon_0 = epsilon_0

        # Initialization of parameters connected to the pipe position and orientation
        self.slope_pipe = slope_pipe
        self.offset_pipe = offset_pipe
        self.angle_pipe = np.arctan(self.slope_pipe)
        self.vector_pipe = np.dot(compute_rotation_matrix(self.angle_pipe), np.array([1, 0]))
        self.perpendicular_angle_pipe = self.angle_pipe + pi / 2
        self.vector_perpendicular_to_pipe = np.dot(compute_rotation_matrix(self.perpendicular_angle_pipe),
                                                   np.array([1, 0]))

        # Initialization of parameters connected to the gaussian noise on position and velocity
        self.mean_velocity_noise = mean_velocity_noise
        self.mean_position_noise = mean_position_noise
        self.std_dev_velocity_noise = std_dev_velocity_noise
        self.std_dev_position_noise = std_dev_position_noise

        # self.distance_from_pipe = distance_from_pipe

        self.reset_type = reset_type

        # Pre-computing the values of the learning and exploration rate for each timestep
        self.learning_rate_vector = np.empty(self.n_episodes)
        self.exploration_rate_vector = np.empty(self.n_episodes)
        for i in range(self.n_episodes):
            self.learning_rate_vector[i] = learning_rate_adaptive(i, self.alpha_0, self.t_star_lr)
            self.exploration_rate_vector[i] = exploration_rate_adaptive(i, self.epsilon_0, self.t_star_epsilon)

        # self.maximum_reward = 1/(1-self.gamma)
        self.maximum_reward = 8000

        # Auxiliary value to avoid multiple computation of the denominator while computing the distance from the pipe
        # (always the same)
        self.auxiliary_den_dist_line = sqrt(self.slope_pipe ** 2 + 1)

        # Plot-related vectors
        # Spatial trajectories
        self.x_trajectory = np.zeros((self.n_agents,))
        self.y_trajectory = np.zeros((self.n_agents,))
        self.orientation = np.zeros((self.n_agents,))
        self.vector_fov_starts = np.zeros((self.n_agents,))
        self.vector_fov_ends = np.zeros((self.n_agents,))

        # Reward vector storing rewards of a single episode, for each agent
        # self.rewards = np.zeros((self.n_agents,))

        # Reward vector storing the cumulative reward of each episode, summed across all timesteps and all agents
        # self.epochs_rewards = np.zeros(self.n_episodes)

        # Vectors used to count the frequency of occupancy of each state
        # self.frequency_state_reward_region = np.zeros((self.n_agents, self.K_s_pipe))
        # self.frequency_state_neighbours = np.zeros((self.n_agents, self.K_s))
        # self.frequency_state_pipe = np.zeros((self.n_agents, self.K_s))

        # Vector used to store the maximum distance towards the objective reached in each episode.
        self.maximum_distance_towards_objective = np.zeros(self.n_episodes)

        self.fraction_of_seen_sections_of_pipe = np.zeros(self.n_episodes)

        # Vector used to store the polar order parameter (alignment of agents across each episode)
        # self.polar_order_param = np.zeros(self.n_episodes)
        self.number_of_steps_per_episode = np.zeros(self.n_episodes, dtype=int)

        self.done = False

        # self.boolean_array_visibility = np.empty(int(1 / (1 - self.gamma)))

        # self.boolean_array_visited_pipes = np.empty(int(1 / 1 - self.gamma))

        self.output_directory = '.'

    def add_agent(self, x, y, v):
        """
        Adds a new agent to the environment, given its position and velocity; updates related parameters and data
        structures.
        """
        new_agent = Agent(x,
                          y,
                          v,
                          self.v0,
                          self.phi,
                          self.K_a,
                          self.possible_states,
                          self.K_s_pipe,
                          self.R,
                          self.gamma)
        new_agent.oriented_distance_from_pipe = self.compute_oriented_distance_from_pipe(new_agent.p)
        self.agents_list.append(new_agent)
        self.n_agents += 1

        self.x_trajectory = np.zeros((self.n_agents,))
        self.y_trajectory = np.zeros((self.n_agents,))
        self.orientation = np.zeros((self.n_agents,))
        self.vector_fov_ends = np.zeros((self.n_agents,))
        self.vector_fov_starts = np.zeros((self.n_agents,))
        # self.rewards = np.zeros((self.n_agents,))

    def compute_distance(self, i, j):
        """
        Computes the Euclidean distance between agent i and j.
        """
        return euclidean_norm(self.agents_list[i].p - self.agents_list[j].p)

    def compute_average_velocity(self, index):
        """
        Computes the average velocity across the neighbours of agent "index"
        """
        cnt = 0
        cum_sum = np.zeros(self.agents_list[index].v.shape)
        for i in [x for x in range(self.n_agents) if
                  x != index and self.compute_distance(index, x) < self.R and self.agents_list[
                      index].is_point_in_field_of_view(self.agents_list[x].p)]:
            cnt += 1
            cum_sum += self.agents_list[i].v
        if cnt == 0:  # there are no neighbours
            return cum_sum  # will be a vector of zeros
        else:
            return cum_sum / cnt

    def discretize_state(self, state):
        """
        Discretizes the state of an agent.
        """
        index_neighbours = (np.abs(self.extended_states - state[0])).argmin()

        # Takes into account the previously introduced problem with -pi and pi associated to two different states
        if index_neighbours == self.K_s:
            index_neighbours = 0

        index_pipe = (np.abs(self.extended_states - state[1])).argmin()

        # Takes into account the previously introduced problem with -pi and pi associated to two different states
        if index_pipe == self.K_s:
            index_pipe = 0

        return [self.possible_states[index_neighbours], self.possible_states[index_pipe], state[2]]

    def compute_oriented_distance_from_pipe(self, position):
        # computed as the distance of a point from a line (with sign)
        return (position[1] - self.slope_pipe * position[0] - self.offset_pipe) / self.auxiliary_den_dist_line

    # def is_agent_seeing_the_pipe(self, index):
    #     agent = self.agents_list[index]
    #     if -self.R < agent.oriented_distance_from_pipe < self.R \
    #             and is_scalar_in_visible_interval(agent.p[0], self.boolean_array_visibility, 5):
    #         return agent.oriented_distance_from_pipe * self.compute_oriented_distance_from_pipe(
    #             agent.p + agent.vector_start_fov) <= 0 \
    #                or agent.oriented_distance_from_pipe * self.compute_oriented_distance_from_pipe(
    #             agent.p + agent.vector_end_fov) <= 0
    #     else:
    #         return False
    def is_agent_seeing_the_pipe(self, index):
        agent = self.agents_list[index]
        if -self.R < agent.oriented_distance_from_pipe < self.R and np.random.binomial(size=1, p=.5, n=1):
            return agent.oriented_distance_from_pipe * self.compute_oriented_distance_from_pipe(
                agent.p + agent.vector_start_fov) <= 0 \
                   or agent.oriented_distance_from_pipe * self.compute_oriented_distance_from_pipe(
                agent.p + agent.vector_end_fov) <= 0
        else:
            return False

    def obtain_agent_state(self, index):
        """
        Obtains the state of agent "index".
        """
        # Pipe state computation
        if self.agents_list[index].flag_is_agent_seeing_the_pipe:  # agent is seeing the pipe
            state_reward_region = 1
        elif self.agents_list[index].flag_agent_knows_info_on_position_of_pipe:
            if self.agents_list[index].oriented_distance_from_pipe > 0:
                state_reward_region = 0
            else:
                state_reward_region = 2
        else:  # agent knows no info on the pipe
            self.agents_list[index].flag_agent_knows_info_on_position_of_pipe = False
            state_reward_region = 3

        # Neighbours state computation: compute the angular difference between the orientation of the agent and
        # the normalized average velocity of its neighbours
        average_velocity = self.compute_average_velocity(index)
        normalized_average_velocity = euclidean_norm(average_velocity)

        # Compute a rotated v of pi/2; needed for computation of the state
        rotated_v = np.dot(compute_rotation_matrix(pi / 2), self.agents_list[index].v)

        if normalized_average_velocity == 0:  # there are no neighbours
            state_neighbours = self.no_neighbours_state
        else:  # there are neighbours
            if np.dot(average_velocity, rotated_v) > 0:
                state_neighbours = np.arccos(
                    np.dot(average_velocity, self.agents_list[index].v) / normalized_average_velocity)
            else:
                state_neighbours = - np.arccos(
                    np.dot(average_velocity, self.agents_list[index].v) / normalized_average_velocity)

        if np.dot(self.vector_pipe, rotated_v) > 0:
            state_pipe = np.arccos(
                np.dot(self.vector_pipe, self.agents_list[index].v) / euclidean_norm(self.vector_pipe))
        else:
            state_pipe = -np.arccos(
                np.dot(self.vector_pipe, self.agents_list[index].v) / euclidean_norm(self.vector_pipe))

        state = np.array([state_neighbours, state_pipe, state_reward_region])
        return self.discretize_state(state)

    def obtain_reward_of_agent(self, index):
        """
        Obtains the reward of agent "index"
        """
        if self.agents_list[index].flag_is_agent_seeing_the_pipe:
            return np.cos(self.agents_list[index].Beta - self.angle_pipe)
        else:  # agent is not in the reward region
            return 0

    def save_episode_trajectories(self, t):
        """
        Auxiliary method that saves trajectories, needed only for detailed intermediate plots of the single episode.
        """
        for i in range(self.n_agents):
            # self.rewards[i][t] = self.agents_list[i].r
            self.x_trajectory[i][t] = self.agents_list[i].p[0]
            self.y_trajectory[i][t] = self.agents_list[i].p[1]
            self.orientation[i][t] = degrees(self.agents_list[i].Beta) - 90
            self.vector_fov_starts[i][t] = self.agents_list[i].vector_start_fov
            self.vector_fov_ends[i][t] = self.agents_list[i].vector_end_fov
            # agent_state_indexes = self.agents_list[i].obtain_state_indexes(self.agents_list[i].s)
            # self.frequency_state_reward_region[i][agent_state_indexes[2]] += 1
            # self.frequency_state_neighbours[i, agent_state_indexes[0]] += 1
            # self.frequency_state_pipe[i, agent_state_indexes[1]] += 1

    # def update_polar_order_parameter(self, current_episode):
    #     """
    #     Auxiliary method that updates the polar order parameter vector in the correspondent episode.
    #     """
    #     velocities_sum = np.zeros(self.agents_list[0].v.shape)
    #     for i in range(self.n_agents):
    #         velocities_sum += self.agents_list[i].v
    #     self.polar_order_param[current_episode] += euclidean_norm(velocities_sum) / self.n_agents

    def simulation_step(self, t, current_episode):
        """
        Simulates a single timestep t of episode current_episode.
        """
        # Update of the polar order parameter
        # self.update_polar_order_parameter(current_episode)

        # Action update
        for i in range(self.n_agents):
            self.agents_list[i].update_action(self.exploration_rate_vector[current_episode])

        # Velocity and position update
        for i in range(self.n_agents):
            self.agents_list[i].update_velocity_noisy(self.possible_actions[self.agents_list[i].a],
                                                      self.mean_velocity_noise,
                                                      self.std_dev_velocity_noise)
            self.agents_list[i].update_position_noisy(self.mean_position_noise, self.std_dev_position_noise)
            self.agents_list[i].oriented_distance_from_pipe = self.compute_oriented_distance_from_pipe(
                self.agents_list[i].p)
            self.agents_list[i].update_info_position_of_pipe(self.is_agent_seeing_the_pipe(i))
            # self.agents_list[i].flag_is_agent_seeing_the_pipe = self.is_agent_seeing_the_pipe(i)
            # if self.agents_list[i].flag_is_agent_seeing_the_pipe:
            #     self.agents_list[i].timeout_info_pipe = 0
            #     self.agents_list[i].flag_agent_knows_info_on_position_of_pipe = True
            #     # index_section = floor(self.agents_list[i].p[0] / 5)
            #     # if not self.boolean_array_visited_pipes[index_section]:
            #     #     self.boolean_array_visited_pipes[index_section] = 1
            # else:
            #     self.agents_list[i].timeout_info_pipe += 1
            #     if self.agents_list[i].timeout_info_pipe > 10:
            #         self.agents_list[i].flag_agent_knows_info_on_position_of_pipe = False


        # State update
        for i in range(self.n_agents):
            self.agents_list[i].update_state(self.obtain_agent_state(i))

        # Reward computation and Q matrix update
        for i in range(self.n_agents):
            reward = self.obtain_reward_of_agent(i)
            self.agents_list[i].update_Q_matrix_exp_sarsa(self.learning_rate_vector[current_episode],
                                                          reward,
                                                          self.exploration_rate_vector[current_episode],
                                                          t == self.number_of_steps_per_episode[current_episode] - 1)
            # self.epochs_rewards[i][current_episode] += reward

    def reset_position_and_velocities_in_area(self):
        """
        Resets the position and velocities of all agents so that:
        - their position falls inside a specific area
        - they all face towards the right (means v[0] > 0)
        - the leftmost agent sees them all (has all the other agents as neighbours)
        """
        for i in range(self.n_agents):
            random_velocity = np.empty(2)
            random_velocity[0] = random.uniform(0, 1)
            random_velocity[1] = random.uniform(-1, 1)
            normalized_velocity = random_velocity / euclidean_norm(random_velocity)
            self.agents_list[i].set_position(random.uniform(0.5, 1),
                                             random.uniform(-0.5, 0.5))
            self.agents_list[i].set_velocity(normalized_velocity)

        looking_agent_index = random.choice(range(self.n_agents))
        self.agents_list[looking_agent_index].set_position(0, 0)
        self.agents_list[looking_agent_index].set_velocity(np.array([1, 0]))

    def reset_position_and_velocities_in_line(self):
        """
        Reset the position and velocities of all agents positioning them:
        - on a line
        - facing the same direction
        - in random positions
        """
        order = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            order[i] = i

        random.shuffle(order)
        for i in range(self.n_agents):
            self.agents_list[i].set_position(order[i] * 0.5, 0)
            self.agents_list[i].set_velocity(np.array([1, 0]))

    def simulate_episode(self, current_episode, save_trajectory):
        """
        Simulates a whole episode.
        """
        self.number_of_steps_per_episode[current_episode] = int(np.random.geometric(1 - self.gamma))
        # number_of_intervals = floor(self.number_of_steps_per_episode[current_episode] * self.v0 / 5) + 2
        # self.boolean_array_visibility = np.random.binomial(size=max(number_of_intervals, 1), n=1, p=.5)
        # self.boolean_array_visibility[0] = 1
        # self.boolean_array_visited_pipes = np.zeros(max(number_of_intervals, 1))

        for i in range(self.n_agents):
            self.agents_list[i].update_state(self.obtain_agent_state(i))

        if save_trajectory:
            self.x_trajectory = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.y_trajectory = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.orientation = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.vector_fov_starts = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode], 2))
            self.vector_fov_ends = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode], 2))
            # self.rewards = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            # self.frequency_state_reward_region = np.zeros((self.n_agents, self.K_s_pipe))
            # self.frequency_state_neighbours = np.zeros((self.n_agents, self.K_s))
            # self.frequency_state_pipe = np.zeros((self.n_agents, self.K_s))
            for t in range(self.number_of_steps_per_episode[current_episode]):
                self.simulation_step(t, current_episode)
                self.save_episode_trajectories(t)
        else:
            for t in range(self.number_of_steps_per_episode[current_episode]):
                self.simulation_step(t, current_episode)

        # Computes the maximum distance reached towards the objective (for plots)
        distance_from_objective = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            distance_from_objective[i] = euclidean_norm(
                self.vector_pipe * self.v0 * self.number_of_steps_per_episode[current_episode] - self.agents_list[i].p)
        self.maximum_distance_towards_objective[current_episode] = 1 - np.min(distance_from_objective) / (
                self.v0 * self.number_of_steps_per_episode[current_episode])

        # self.fraction_of_seen_sections_of_pipe[current_episode] = np.sum(self.boolean_array_visited_pipes) / np.sum(
        #     self.boolean_array_visibility)

        if save_trajectory:
            np.savez("%s/episode_%d.npz" % (self.output_directory, current_episode),
                     x_traj=self.x_trajectory,
                     y_traj=self.y_trajectory,
                     orientation=self.orientation,
                     vector_fov_starts=self.vector_fov_starts,
                     vector_fov_ends=self.vector_fov_ends
                     # boolean_array_visibility=self.boolean_array_visibility
                     )

        # Reset positions and velocities of the agents accordingly
        if self.reset_type == "area":
            self.reset_position_and_velocities_in_area()
        elif self.reset_type == "line":
            self.reset_position_and_velocities_in_line()

    def complete_simulation(self, interval_print_data, output_directory):
        """
        Performs the complete simulation, plotting single episode related data every "interval_print_data" steps.
        """
        # self.epochs_rewards = np.zeros((self.n_agents, self.n_episodes))

        self.output_directory = output_directory

        global_start_time = time()
        start_time = time()
        for j in range(self.n_episodes):

            save_trajectory = j % interval_print_data == 0 or j == self.n_episodes - 1
            self.simulate_episode(j, save_trajectory)
            if j % (self.n_episodes / 10) == 0:
                print(100*(j/self.n_episodes), " %, elapsed time: ", time() - start_time)
                start_time = time()

        matrices_to_be_saved = np.zeros([self.n_agents, self.K_s, self.K_s, self.K_s_pipe, self.K_a])
        for i in range(self.n_agents):
            matrices_to_be_saved[i, :] = self.agents_list[i].Q

        frequencies_for_policy_plots = np.zeros([self.n_agents, self.K_s, self.K_s, self.K_s_pipe])
        for i in range(self.n_agents):
            frequencies_for_policy_plots[i] = self.agents_list[i].Q_visits

        frequencies_for_policy_plots = frequencies_for_policy_plots
        np.savez("%s/data_for_plots.npz" % self.output_directory,
                 K_s=self.K_s,
                 K_s_pipe=self.K_s_pipe,
                 K_a=self.K_a,
                 theta_max=self.theta_max,
                 maximum_distance_towards_objective=self.maximum_distance_towards_objective,
                 fraction_of_seen_sections_of_pipe=self.fraction_of_seen_sections_of_pipe,
                 Q_matrices=matrices_to_be_saved,
                 Q_visits=frequencies_for_policy_plots,
                 number_of_steps_per_episode=self.number_of_steps_per_episode
                 )

        print("Global time: ", time() - global_start_time)
