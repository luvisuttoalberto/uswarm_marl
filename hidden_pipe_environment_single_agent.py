import numpy as np
from auxiliary_functions import compute_rotation_matrix, exploration_rate_adaptive, is_scalar_in_visible_interval
from math import sqrt, degrees, pi, floor
from agent_baseline import AgentBaseline
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
                 std_dev_measure_pipe,
                 prob_end_surge,
                 forgetting_factor,
                 visibility_pipe,
                 pipe_recognition_probability):
        """
        Constructor of the class.
        """

        # Macros to represent the information states
        self.STATE_SEE = 0
        self.STATE_CLOSE_RIGHT = 1
        self.STATE_CLOSE_LEFT = 2
        self.STATE_JUST_LOST = 3
        self.STATE_LOST = 4

        self.n_agents = 1
        self.gamma = gamma

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
        self.n_episodes = n_episodes
        self.t_star_epsilon = t_star_epsilon
        self.t_star_lr = t_star_lr
        self.t_stop = t_stop
        self.epsilon_0 = epsilon_0

        # Initialization of parameters connected to the pipe position and orientation
        self.slope_pipe = slope_pipe
        self.offset_pipe = offset_pipe
        self.angle_pipe = np.arctan(self.slope_pipe)
        self.vector_pipe = np.dot(compute_rotation_matrix(self.angle_pipe), np.array([1, 0]))

        # Initialization of parameters connected to the gaussian noise on position and velocity
        self.mean_velocity_noise = mean_velocity_noise
        self.mean_position_noise = mean_position_noise
        self.std_dev_velocity_noise = std_dev_velocity_noise
        self.std_dev_position_noise = std_dev_position_noise

        # Pre-computing the values of the exploration rate for each time step
        self.exploration_rate_vector = np.empty(self.n_episodes)
        for i in range(self.n_episodes):
            self.exploration_rate_vector[i] = exploration_rate_adaptive(i, self.epsilon_0, self.t_star_epsilon, self.t_stop)

        # Auxiliary value to avoid multiple computation of the denominator while computing the distance from the pipe
        # (always the same)
        self.auxiliary_den_dist_line = sqrt(self.slope_pipe ** 2 + 1)

        # Plot-related vectors
        # Spatial trajectories
        self.x_trajectory = np.zeros((self.n_agents,))
        self.y_trajectory = np.zeros((self.n_agents,))
        self.orientation = np.zeros((self.n_agents,))
        self.vector_fov_ends = np.zeros((self.n_agents,))
        self.vector_fov_starts = np.zeros((self.n_agents,))
        self.visibility_of_pipe = np.zeros((self.n_agents,), dtype=bool)
        self.boolean_array_visibility = np.empty((self.n_agents, int(1 / (1 - self.gamma))))

        # Vector used to store the fraction of pipe seen in each episode
        self.fraction_of_seen_sections_of_pipe = np.zeros(self.n_episodes)

        # Vector used to store the average fraction of pipe seen in each episode
        self.average_fraction_pipe = np.zeros(self.n_episodes)

        # Vector that stores the number of time steps for each episode
        self.number_of_steps_per_episode = np.zeros(self.n_episodes, dtype=int)

        # Vector used to store if an agent has seen a certain section of the pipe or not
        self.boolean_array_visited_pipes = np.empty(int(1 / 1 - self.gamma))

        # Output directory in which to store data
        self.output_directory = '.'

        # Standard deviation for the error in the measure of the pipe orientation
        self.std_dev_measure_pipe = std_dev_measure_pipe

        # Probability of moving from state JustLost to state Lost
        self.prob_end_surge = prob_end_surge

        # Forgetting factor for the calculus of the running average over the pipe orientation
        self.forgetting_factor = forgetting_factor

        # Visibility of the pipe (value in [0,1])
        self.visibility_pipe = visibility_pipe

        # Probability of recognition of the pipe if it's in the agent's fov
        self.pipe_recognition_probability = pipe_recognition_probability

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

        return [self.possible_states[index_pipe_state]]

    def compute_oriented_distance_from_pipe(self, position):
        """
        Computes the distance of a point (the agent) from a line (the pipe)
        """
        return (position[1] - self.slope_pipe * position[0] - self.offset_pipe) / self.auxiliary_den_dist_line

    def is_agent_seeing_the_pipe(self):
        """
        Checks if the agent is seeing a visible section of the pipe
        """
        if -self.R < self.agent.oriented_distance_from_pipe < self.R and np.random.binomial(size=1, p=self.pipe_recognition_probability, n=1) and is_scalar_in_visible_interval(self.agent.p[0], self.boolean_array_visibility[0], 5):
            return self.agent.oriented_distance_from_pipe * self.compute_oriented_distance_from_pipe(self.agent.p + self.agent.vector_start_fov) <= 0 or self.agent.oriented_distance_from_pipe * self.compute_oriented_distance_from_pipe(self.agent.p + self.agent.vector_end_fov) <= 0
        else:
            return False

    def obtain_information_state(self, index):
        """
        Obtains the information state of agent "index"
        """
        if self.agent.flag_is_agent_seeing_the_pipe:  # agent is seeing the pipe
            if -0.5 < self.agent.oriented_distance_from_pipe < 0.5:
                state_information = self.STATE_SEE
            elif self.agent.oriented_distance_from_pipe < -0.5:
                state_information = self.STATE_CLOSE_RIGHT
            else:
                state_information = self.STATE_CLOSE_LEFT
        elif self.agent.s[1] in [self.STATE_SEE, self.STATE_CLOSE_RIGHT, self.STATE_CLOSE_LEFT]:
            state_information = self.STATE_JUST_LOST
        elif self.agent.s[1] == self.STATE_JUST_LOST:
            if np.random.binomial(1, self.prob_end_surge):
                state_information = self.STATE_LOST
            else:
                state_information = self.agent.s[1]
        else:
            state_information = self.agent.s[1]

        return state_information

    def obtain_orientations_states(self):
        """
        Obtains the pipe and the neighbors state of agent "index"
        """
        rotated_v = np.dot(compute_rotation_matrix(pi/2), self.agent.v)

        if np.dot(self.agent.vector_pipe, rotated_v) > 0:
            state_pipe = np.arccos(np.dot(self.agent.vector_pipe, self.agent.v) / euclidean_norm(self.agent.vector_pipe))
        else:
            state_pipe = -np.arccos(np.dot(self.agent.vector_pipe, self.agent.v) / euclidean_norm(self.agent.vector_pipe))

        state = np.asarray([state_pipe])
        return self.discretize_state(state)

    def obtain_reward_of_agent(self):
        """
        Obtains the reward of agent "index"
        """
        if self.agent.flag_is_agent_seeing_the_pipe:
            return np.cos(self.agent.Beta - self.agent.angle_pipe) - 1
        else: # agent is not seeing the pipe
            return -1

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
        Simulates a single time step t of episode current_episode.
        """
        # Action update
        self.agent.update_action(self.exploration_rate_vector[current_episode])

        # Velocity and position update
        self.agent.update_velocity_noisy(self.possible_actions[self.agent.a], self.mean_velocity_noise, self.std_dev_velocity_noise)
        self.agent.update_position_noisy(self.mean_position_noise, self.std_dev_position_noise)

        # Update infos on pipe for each agent
        self.agent.oriented_distance_from_pipe = self.compute_oriented_distance_from_pipe(self.agent.p)
        self.agent.update_info_on_pipe(self.is_agent_seeing_the_pipe(), t == 0)
        if self.agent.flag_is_agent_seeing_the_pipe:
            self.boolean_array_visited_pipes[floor(self.agent.p[0])] = 1

        # State update
        self.agent.update_information_state(self.obtain_information_state())

        self.agent.update_orientations_state(self.obtain_orientations_states())

        # Reward computation and Q matrix update
        reward = self.obtain_reward_of_agent()
        self.agent.update_Q_matrix_exp_sarsa(reward,
                                             self.exploration_rate_vector[current_episode],
                                             t == self.number_of_steps_per_episode[current_episode] - 1)

    def reset_position_and_velocity(self):
        """
        Resets the position and velocities
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
        # Compute episode length
        if current_episode >= self.t_stop:# If it's one of the benchmark episodes, length is fixed
            self.number_of_steps_per_episode[current_episode] = 1/(1-self.gamma)
        else: # Otherwise length is randomly sampled
            self.number_of_steps_per_episode[current_episode] = int(np.random.geometric(1 - self.gamma))

        # Divide the length of the pipe in intervals. Depending on agents' initial positions we have longer or shorter pipe.
        number_of_intervals = floor(self.number_of_steps_per_episode[current_episode] * self.v0) + 2

        # Sample the visibility of each section
        self.boolean_array_visibility = np.zeros((self.n_agents, max(floor(number_of_intervals / 5) + 1, 1)))
        self.boolean_array_visibility[0] = np.random.binomial(size=max(floor(number_of_intervals / 5) + 1, 1), n=1, p=self.visibility_pipe)
        self.boolean_array_visibility[0][0] = 1
        self.boolean_array_visited_pipes = np.zeros(max(number_of_intervals, 1))

        # Initialize info on pipe and states for each agent at first time step
        self.agent.oriented_distance_from_pipe = self.compute_oriented_distance_from_pipe(self.agent.p)
        self.agent.update_info_on_pipe(self.is_agent_seeing_the_pipe(), True)
        if self.agent.flag_is_agent_seeing_the_pipe:
            self.boolean_array_visited_pipes[floor(self.agent.p[0])] = 1

        self.agent.update_information_state(self.obtain_information_state())

        self.agent.update_orientations_state(self.obtain_orientations_states())

        if save_trajectory:
            # Initialize vectors to save data
            self.x_trajectory = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.y_trajectory = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.orientation = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]))
            self.vector_fov_starts = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode], 2))
            self.vector_fov_ends = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode], 2))
            self.visibility_of_pipe = np.zeros((self.n_agents, self.number_of_steps_per_episode[current_episode]), dtype=bool)

            # Simulate all time steps
            for t in range(self.number_of_steps_per_episode[current_episode]):
                self.simulation_step(t, current_episode)
                self.save_episode_trajectories(t)
        else:
            for t in range(self.number_of_steps_per_episode[current_episode]):
                self.simulation_step(t, current_episode)

        self.fraction_of_seen_sections_of_pipe[current_episode] = np.sum(self.boolean_array_visited_pipes) / (5*np.sum(self.boolean_array_visibility[0]))

        tmp_average_fraction_pipe = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            tmp_average_fraction_pipe[i] = np.sum(self.boolean_array_visited_pipes[i]) / (5 * np.sum(self.boolean_array_visibility[i]))

        self.average_fraction_pipe[current_episode] = np.mean(tmp_average_fraction_pipe)

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

    def complete_simulation(self, interval_print_data, output_directory):
        """
        Performs the complete simulation, plotting single episode related data every "interval_print_data" steps.
        """

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

        global_state_action_rate_visits = np.zeros([self.n_agents, len(self.possible_states), self.K_s_pipe, self.K_a])
        global_state_action_rate_visits[0] = self.agent.state_action_rate_visits

        np.savez("%s/data_for_plots.npz" % self.output_directory,
                 K_s=self.K_s,
                 K_s_pipe=self.K_s_pipe,
                 K_a=self.K_a,
                 theta_max=self.theta_max,
                 fraction_of_seen_sections_of_pipe=self.fraction_of_seen_sections_of_pipe,
                 Q_matrices=matrices_to_be_saved,
                 Q_visits=frequencies_for_policy_plots,
                 number_of_steps_per_episode=self.number_of_steps_per_episode,
                 average_fraction_pipe=self.average_fraction_pipe
                 )

        print("Global time: ", time() - global_start_time)
