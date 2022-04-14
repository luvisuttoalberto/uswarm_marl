import numpy as np

from agent_baseline import AgentBaseline
from auxiliary_functions import compute_rotation_matrix, learning_rate_adaptive, exploration_rate_adaptive, is_scalar_in_visible_interval
from math import sqrt, degrees, pi, floor
from numpy.linalg import norm as euclidean_norm
import random
from time import time


class HiddenPipeEnvironmentSingleAgent:
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
                 t_stop,
                 epsilon_0,
                 slope_pipe,
                 offset_pipe,
                 mean_velocity_noise,
                 mean_position_noise,
                 std_dev_velocity_noise,
                 std_dev_position_noise,
                 gamma,
                 prob_no_switch_state,
                 flag_spatially_uncorrelated_case,
                 std_dev_measure_pipe,
                 prob_end_surge,
                 forgetting_factor):
        """
        Constructor of the class.
        """

        # The number of agents is initialized to 0; then agents are added to the environment (and to the agents list)
        # one at a time
        self.n_agents = 1
        self.gamma = gamma
        self.v0 = v0
        self.phi = phi
        self.K_a = k_a
        self.K_s_pipe = k_s_pipe
        self.R = radius
# ----------------------------------------------------------------------------------------------------------------
# CLASSIC STATES
#         Initialization of the vector of possible states (Discretization of the states)
        self.possible_states = [x for x in np.linspace(-pi, pi - 2*pi/k_s, k_s)]
        # This auxiliary vector of states is used in order to assign states close to -pi and pi to the same state
        self.extended_states = [x for x in np.linspace(-pi, pi, k_s + 1)]
# -----------------------------------------------------------------------------------------------------------------
# ADDITIONAL STATES FOR NO INFO ON ORIENTATION OF THE PIPE
#         self.possible_states = [x for x in np.linspace(-pi, pi, k_s)]
#         # Giving a value to the state in which there are no neighbours
#         self.no_neighbours_state = 100.
#         # Appending the "no neighbours" state to the previously initialized vector
#         self.possible_states[k_s - 1] = self.no_neighbours_state
#
#         # This auxiliary vector of states is used in order to assign states close to -pi and pi to the same state
#         self.extended_states = [x for x in np.linspace(-pi, pi, k_s)]
#         self.extended_states[k_s - 1] = self.no_neighbours_state
#         # Extending the vector of possible states by appending pi
#         self.extended_states.append(pi)
# ------------------------------------------------------------------------------------------------------------------

        self.x_trajectory = np.zeros((self.n_agents,))
        self.y_trajectory = np.zeros((self.n_agents,))
        self.orientation = np.zeros((self.n_agents,))
        self.vector_fov_ends = np.zeros((self.n_agents,))
        self.vector_fov_starts = np.zeros((self.n_agents,))
        self.visibility_of_pipe = np.zeros((self.n_agents,), dtype=bool)
        self.boolean_array_visibility = np.empty((self.n_agents, int(1 / (1 - self.gamma))))

        self.slope_pipe = slope_pipe

        self.theta_max = theta_max

        # States
        self.K_s = k_s


        self.timeout_side = 0

        # self.extended_states[k_s - 1] = self.no_neighbours_state
        # Extending the vector of possible states by appending pi
        # self.extended_states.append(pi)

        # Actions

        # Initialization of the vector of possible actions (Discretization of the actions)
        self.possible_actions = [x for x in np.linspace(-self.theta_max, self.theta_max, self.K_a)]

        # Initialization of previously explained parameters
        self.alpha_0 = alpha_0
        # self.T = T
        self.n_episodes = n_episodes
        self.t_star_epsilon = t_star_epsilon
        self.t_star_lr = t_star_lr
        self.t_stop = t_stop
        self.epsilon_0 = epsilon_0

        # Initialization of parameters connected to the pipe position and orientation
        self.offset_pipe = offset_pipe
        self.angle_pipe = np.arctan(self.slope_pipe)
        self.vector_pipe = np.dot(compute_rotation_matrix(self.angle_pipe), np.array([1, 0]))
        # self.perpendicular_angle_pipe = self.angle_pipe + pi / 2
        # self.vector_perpendicular_to_pipe = np.dot(compute_rotation_matrix(self.perpendicular_angle_pipe),
        #                                            np.array([1, 0]))

        # Initialization of parameters connected to the gaussian noise on position and velocity
        self.mean_velocity_noise = mean_velocity_noise
        self.mean_position_noise = mean_position_noise
        self.std_dev_velocity_noise = std_dev_velocity_noise
        self.std_dev_position_noise = std_dev_position_noise

        # self.distance_from_pipe = distance_from_pipe

        # Pre-computing the values of the learning and exploration rate for each timestep
        self.learning_rate_vector = np.empty(self.n_episodes)
        self.exploration_rate_vector = np.empty(self.n_episodes)
        for i in range(self.n_episodes):
            self.learning_rate_vector[i] = learning_rate_adaptive(i, self.alpha_0, self.t_star_lr)
            self.exploration_rate_vector[i] = exploration_rate_adaptive(i, self.epsilon_0, self.t_star_epsilon, self.t_stop)

        # Auxiliary value to avoid multiple computation of the denominator while computing the distance from the pipe
        # (always the same)
        self.auxiliary_den_dist_line = sqrt(self.slope_pipe ** 2 + 1)

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

        self.number_of_steps_per_episode = np.zeros(self.n_episodes, dtype=int)

        self.boolean_array_visited_pipes = np.empty(int(1 / 1 - self.gamma))
        self.average_highest_reward = np.zeros(self.n_episodes)

        self.output_directory = '.'

        self.flag_spatially_uncorrelated_case = flag_spatially_uncorrelated_case

        self.prob_no_switch_state = prob_no_switch_state

        self.std_dev_measure_pipe = std_dev_measure_pipe

        self.prob_end_surge = prob_end_surge

        self.forgetting_factor = forgetting_factor

        self.agent = AgentBaseline(0, 0, np.array([1, 0]), self.v0, self.phi, self.K_a, self.possible_states, self.K_s_pipe, self.R, self.std_dev_measure_pipe, self.forgetting_factor, self.alpha_0, self.t_star_lr)
        self.agent.oriented_distance_from_pipe = self.compute_oriented_distance_from_pipe(self.agent.p)

    def discretize_state(self, state):
        """
            Discretizes the state of an agent.
            """

        index_pipe_state = (np.abs(self.extended_states - state[0])).argmin()

        # Takes into account the previously introduced problem with -pi and pi associated to two different states
        if index_pipe_state == self.K_s:
            index_pipe_state = 0

        return [self.possible_states[index_pipe_state], state[1]]

    def compute_oriented_distance_from_pipe(self, position):
        # computed as the distance of a point from a line (with sign)
        return (position[1] - self.slope_pipe * position[0] - self.offset_pipe) / self.auxiliary_den_dist_line

    def is_agent_seeing_the_pipe(self):
        if -self.R < self.agent.oriented_distance_from_pipe < self.R and is_scalar_in_visible_interval(self.agent.p[0], self.boolean_array_visibility[0], 5):
            return self.agent.oriented_distance_from_pipe * self.compute_oriented_distance_from_pipe(
                self.agent.p + self.agent.vector_start_fov) <= 0 \
                   or self.agent.oriented_distance_from_pipe * self.compute_oriented_distance_from_pipe(
                self.agent.p + self.agent.vector_end_fov) <= 0
        else:
            return False

    def obtain_agent_state(self):
        """
        Obtains the state of agent "index".
        """
        # Pipe state computation
        if self.agent.flag_is_agent_seeing_the_pipe:  # agent is seeing the pipe
            if -1 < self.agent.oriented_distance_from_pipe < 1:
                state_relative_position = 1
            elif self.agent.oriented_distance_from_pipe < -1:
                state_relative_position = 4
            else:
                state_relative_position = 5
        elif self.agent.s[1] == 1:
# -------------------------------------------------------------------------------------------------------------------
# VERSION WITHOUT INTERMEDIATE STATE
#             if self.agent.oriented_distance_from_pipe > 0:
#                 state_relative_position = 0
#             else:
#                 state_relative_position = 2

# VERSION WITH INTERMEDIATE STATE
            state_relative_position = 3
        elif self.agent.s[1] == 3:
            # if self.agent.timeout_info_pipe > 20:
            if np.random.binomial(1, self.prob_end_surge):
                self.timeout_side = 0
                # CHANGE THIS WITH LAST ESTIMATION OF RELATIVE POSITION WRT PIPE: TO BE KEPT IN MEMORY BY AGENT
                if self.agent.oriented_distance_from_pipe > 0:
                    state_relative_position = 0
                else:
                    state_relative_position = 2
            else:
                state_relative_position = self.agent.s[1]
# ------------------------------------------------------------------------------------------------------------------
        else:
            # if self.timeout_side < 5:
            if np.random.binomial(1, self.prob_no_switch_state):
                state_relative_position = self.agent.s[1]
                # self.timeout_side += 1
            else:
                # self.timeout_side = 0
                if self.agent.s[1] == 0:
                    state_relative_position = 2
                else:
                    state_relative_position = 0

        if state_relative_position == -1:
            print("ERRORE: previous s = ", self.agent.s[1])
        # Compute a rotated v of pi/2; needed for computation of the state
        rotated_v = np.dot(compute_rotation_matrix(pi / 2), self.agent.v)
# ------------------------------------------------------------------------------------------------------------------
# VERSION WITH NO EXTRA STATE FOR NO INFORMATION
        if np.dot(self.agent.vector_pipe, rotated_v) > 0:
            state_orientation = np.arccos(np.dot(self.agent.vector_pipe, self.agent.v) / euclidean_norm(self.agent.vector_pipe))
        else:
            state_orientation = -np.arccos(np.dot(self.agent.vector_pipe, self.agent.v) / euclidean_norm(self.agent.vector_pipe))
# VERSION WITH EXTRA STATE FOR NO INFORMATION
#         if state_relative_position == 0 or state_relative_position == 2:
#             state_orientation = self.no_neighbours_state
#         else:
#             if np.dot(self.agent.vector_pipe, rotated_v) > 0:
#                 state_orientation = np.arccos(
#                     np.dot(self.agent.vector_pipe, self.agent.v) / euclidean_norm(self.agent.vector_pipe))
#             else:
#                 state_orientation = -np.arccos(
#                     np.dot(self.agent.vector_pipe, self.agent.v) / euclidean_norm(self.agent.vector_pipe))
# ------------------------------------------------------------------------------------------------------------------
        state = np.array([state_orientation, state_relative_position])
        return self.discretize_state(state)

    def obtain_reward_of_agent(self):
        """
        Obtains the reward of agent "index"
        """
        if self.agent.flag_is_agent_seeing_the_pipe:
            return np.cos(self.agent.Beta - self.agent.angle_pipe)
        else:
            # if -5 < self.agent.oriented_distance_from_pipe < 5:
            return 0
            # else:
                # return -abs(self.agent.oriented_distance_from_pipe)

    def save_episode_trajectories(self, t):
        """
        Auxiliary method that saves trajectories, needed only for detailed intermediate plots of the single episode.
        """
        for i in range(self.n_agents):
            self.x_trajectory[i][t] = self.agent.p[0]
            self.y_trajectory[i][t] = self.agent.p[1]
            self.orientation[i][t] = degrees(self.agent.Beta) - 90
            self.vector_fov_starts[i][t] = self.agent.vector_start_fov
            self.vector_fov_ends[i][t] = self.agent.vector_end_fov
            self.visibility_of_pipe[i][t] = self.agent.flag_is_agent_seeing_the_pipe

    def simulation_step(self, t, current_episode):
        """
        Simulates a single timestep t of episode current_episode.
        """
        # Action update
        self.agent.update_action(self.exploration_rate_vector[current_episode])

        # Velocity and position update
        self.agent.update_velocity_noisy(self.possible_actions[self.agent.a], self.mean_velocity_noise, self.std_dev_velocity_noise)
        self.agent.update_position_noisy(self.mean_position_noise, self.std_dev_position_noise)
        self.agent.oriented_distance_from_pipe = self.compute_oriented_distance_from_pipe(self.agent.p)
        self.agent.update_info_on_pipe(self.is_agent_seeing_the_pipe(), t == 0)
        if self.agent.flag_is_agent_seeing_the_pipe:
            self.boolean_array_visited_pipes[floor(self.agent.p[0])] = 1

        # State update
        self.agent.update_state(self.obtain_agent_state())

        # Reward computation and Q matrix update
        reward = self.obtain_reward_of_agent()
        self.agent.update_Q_matrix_exp_sarsa(reward, self.exploration_rate_vector[current_episode], t == self.number_of_steps_per_episode[current_episode] - 1)
        self.average_highest_reward[current_episode] += self.agent.r / self.number_of_steps_per_episode[current_episode]

    def reset_position_and_velocity(self):
        """
        Resets the position and velocities of all agents so that:
        - their position falls inside a specific area
        - they all face towards the right (means v[0] > 0)
        - the leftmost agent sees them all (has all the other agents as neighbours)
        """
        random_velocity = np.empty(2)
        random_velocity[0] = random.uniform(0, 1)
        random_velocity[1] = random.uniform(-1, 1)
        normalized_velocity = random_velocity / euclidean_norm(random_velocity)
        self.agent.set_position(random.uniform(0.5, 1), random.uniform(-0.5, 0.5))
        self.agent.set_velocity(normalized_velocity)

    def simulate_episode(self, current_episode, save_trajectory):
        """
        Simulates a whole episode.
        """
        if current_episode >= self.t_stop:
            self.number_of_steps_per_episode[current_episode] = 1/(1-self.gamma)
        else:
            self.number_of_steps_per_episode[current_episode] = int(np.random.geometric(1 - self.gamma))
        number_of_intervals = floor(self.number_of_steps_per_episode[current_episode] * self.v0) + 2
        self.boolean_array_visibility = np.zeros((self.n_agents, max(floor(number_of_intervals / 5) + 1, 1)))
        self.boolean_array_visibility[0] = np.random.binomial(size=max(floor(number_of_intervals / 5) + 1, 1), n=1, p=.5)
        self.boolean_array_visibility[0][0] = 1
        self.boolean_array_visited_pipes = np.zeros(max(number_of_intervals, 1))

        self.agent.oriented_distance_from_pipe = self.compute_oriented_distance_from_pipe(self.agent.p)
        self.agent.update_info_on_pipe(self.is_agent_seeing_the_pipe(), True)
        if self.agent.flag_is_agent_seeing_the_pipe:
            self.boolean_array_visited_pipes[floor(self.agent.p[0])] = 1
        self.agent.update_state(self.obtain_agent_state())

        if save_trajectory:
            self.x_trajectory = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.y_trajectory = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.orientation = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.vector_fov_starts = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode], 2))
            self.vector_fov_ends = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode], 2))
            self.visibility_of_pipe = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]), dtype=bool)
            for t in range(self.number_of_steps_per_episode[current_episode]):
                self.simulation_step(t, current_episode)
                self.save_episode_trajectories(t)
        else:
            for t in range(self.number_of_steps_per_episode[current_episode]):
                self.simulation_step(t, current_episode)

        # Computes the maximum distance reached towards the objective (for plots)
        distance_from_objective = np.zeros(self.n_agents)
        distance_from_objective[0] = euclidean_norm(self.vector_pipe * self.v0 * self.number_of_steps_per_episode[current_episode] - self.agent.p)
        self.maximum_distance_towards_objective[current_episode] = 1 - np.min(distance_from_objective) / (self.v0 * self.number_of_steps_per_episode[current_episode])

        self.fraction_of_seen_sections_of_pipe[current_episode] = np.sum(self.boolean_array_visited_pipes) / (5*np.sum(self.boolean_array_visibility[0]))
        if save_trajectory:
            np.savez("%s/episode_%d.npz" % (self.output_directory, current_episode),
                     x_traj=self.x_trajectory,
                     y_traj=self.y_trajectory,
                     orientation=self.orientation,
                     vector_fov_starts=self.vector_fov_starts,
                     vector_fov_ends=self.vector_fov_ends,
                     visibility_of_pipe=self.visibility_of_pipe,
                     boolean_array_visibility=self.boolean_array_visibility[0]
                     )

        # Reset positions and velocities of the agents accordingly
        self.reset_position_and_velocity()
        # if self.reset_type == "area":
        #     self.reset_position_and_velocities_in_area()
        # elif self.reset_type == "line":
        #     self.reset_position_and_velocities_in_line()

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
                print(100 * (j / self.n_episodes), " %, elapsed time: ", time() - start_time)
                start_time = time()

        matrices_to_be_saved = np.zeros([self.n_agents, self.K_s, self.K_s_pipe, self.K_a])
        matrices_to_be_saved[0, :] = self.agent.Q

        frequencies_for_policy_plots = np.zeros([self.n_agents, self.K_s, self.K_s_pipe])
        frequencies_for_policy_plots[0] = self.agent.Q_visits

        np.savez("%s/data_for_plots.npz" % self.output_directory,
                 K_s=self.K_s,
                 K_s_pipe=self.K_s_pipe,
                 K_a=self.K_a,
                 theta_max=self.theta_max,
                 maximum_distance_towards_objective=self.maximum_distance_towards_objective,
                 fraction_of_seen_sections_of_pipe=self.fraction_of_seen_sections_of_pipe,
                 Q_matrices=matrices_to_be_saved,
                 Q_visits=frequencies_for_policy_plots,
                 number_of_steps_per_episode=self.number_of_steps_per_episode,
                 average_highest_reward=self.average_highest_reward
                 )

        print("Global time: ", time() - global_start_time)
