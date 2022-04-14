import numpy as np
from math import pi
from auxiliary_functions import compute_rotation_matrix


class AgentBenchmark:
    """
    Class that defines a single agent.
    """

    def __init__(self, x, y, v, v0, phi, k_a, possible_states, k_s_pipe, radius, std_dev_measure_pipe, forgetting_factor, Q):
        """
        Constructor of the agent.
        x and y are the positional coordinates of the agent.
        v is the velocity.
        v0 is the value of the constant speed (scalar).
        phi is half of the agent's angle of view.
        k_a is the number of possible actions.
        possible_states is a vector containing all the possible neighbors states.
        k_s_pipe is the number of possible information states.
        """
        self.p = np.array([x, y])
        self.v = np.array(v)
        self.phi = phi
        self.v0 = v0
        self.K_a = k_a
        self.possible_states = possible_states
        self.R = radius
        self.Q = Q

        # Current angle of orientation of the agent
        self.Beta = np.arctan2(v[1], v[0])

        # Modification of Beta in order to have it in [0, 2*pi) instead of [-pi, pi)
        if self.Beta < 0:
            self.Beta += 2 * pi

        # Definition of auxiliary values that describe the beginning and end of the field of view of the agent
        self.start_angle_fov = self.Beta - self.phi
        if self.start_angle_fov < 0:
            self.start_angle_fov += 2 * pi
        self.end_angle_fov = (self.Beta + self.phi) % (2 * pi)

        self.vector_start_fov = np.dot(compute_rotation_matrix(self.start_angle_fov), np.array([1, 0])) * self.R
        self.vector_end_fov = np.dot(compute_rotation_matrix(self.end_angle_fov), np.array([1, 0])) * self.R

        #       Boolean auxiliary value stating if angle 0 is in the field of view;
        #       will be used to determine if neighbour is in the fov or not
        self.flag_0_in_interval = self.start_angle_fov > self.end_angle_fov

        #       Boolean auxiliary value stating if the pipe is in the field of view of the agent
        self.flag_is_agent_seeing_the_pipe = False
        # self.flag_agent_knows_info_on_position_of_pipe = False

        # Initialization of the Q matrix (Optimistic approach: initialized at the maximum possible value of the
        # reward) self.Q = np.ones([len(self.possible_states), K_s_pipe, self.K_a])*maximum_reward

        self.Q_visits = np.zeros([len(self.possible_states), len(self.possible_states), k_s_pipe])

        self.a = 0

        # Initialization of the agent's state to default values; will actually be updated before starting the
        # simulation
        self.s = [0.0, 0.0, 1]
        self.old_s = [0.0, 0.0, 0]

        #       Store the reward received by the agent. Needed only for plots
        self.r = 0

        # Auxiliary variable that stores the distance from the pipe (with sign). Added to avoid multiple computation
        # of values.
        self.oriented_distance_from_pipe = 0

        self.orientation_of_pipe = 0

        self.angle_pipe = 0

        self.vector_pipe = np.dot(compute_rotation_matrix(self.angle_pipe), np.array([1, 0]))

        self.std_dev_measure_pipe = std_dev_measure_pipe

        self.forgetting_factor = forgetting_factor

        self.weight_measure = 1.

        self.state_action_rate_visits = np.zeros([len(self.possible_states), len(self.possible_states), k_s_pipe, k_a])

    def update_fov_parameters(self):
        """
        Updates the field of view related parameters based on the value of Beta.
        """
        self.start_angle_fov = self.Beta - self.phi
        if self.start_angle_fov < 0:
            self.start_angle_fov += 2 * pi
        self.end_angle_fov = (self.Beta + self.phi) % (2 * pi)

        self.vector_start_fov = np.dot(compute_rotation_matrix(self.start_angle_fov), np.array([1, 0])) * self.R
        self.vector_end_fov = np.dot(compute_rotation_matrix(self.end_angle_fov), np.array([1, 0])) * self.R

        self.flag_0_in_interval = self.start_angle_fov > self.end_angle_fov

    def set_velocity(self, v):
        """
        Sets the agent's velocity to a given vector v (must be already normalized).
        Beta and the field of view related parameters are updated accordingly.
        """
        self.v = v

        self.Beta = np.arctan2(v[1], v[0])
        if self.Beta < 0:
            self.Beta += 2 * pi

        self.update_fov_parameters()

    def set_position(self, x, y):
        """
        Sets the agent's position to given 2-D coordinates.
        """
        self.p = np.array([x, y])

    def update_velocity(self, theta):
        """
        Updates the velocity orientation, performing a rotation of a given angle theta.
        Beta and the field of view related parameters are updated accordingly
        """
        self.v = np.dot(compute_rotation_matrix(theta), self.v)

        self.Beta = (self.Beta + theta) % (2 * pi)

        self.update_fov_parameters()

    def update_velocity_noisy(self, theta, mean, std_dev):
        """
        Noisy version of the update_velocity method. Updates the velocity orientation of the agent after addition of
        a gaussian noise sampled according to given parameters.
        """
        noisy_theta = theta + np.random.normal(mean, std_dev)
        self.update_velocity(noisy_theta)

    def update_position(self, delta_t=1):
        """
        Updates the position of the agent, according to its velocity and the provided timestep.
        """
        self.p = self.p + self.v0 * delta_t * self.v

    def update_position_noisy(self, mean, std_dev, delta_t=1):
        """
        Noisy version of the update_position method.
        Updates the position of the agent after addition of a gaussian noise sampled according to given parameters.
        """
        self.p = self.p + self.v0 * delta_t * self.v + np.random.normal(mean, std_dev, size=2) * delta_t

    def update_relative_position_state(self, state):
        self.old_s[2] = self.s[2]
        self.s[2] = state

    def update_orientations_state(self, state):
        self.old_s[0] = self.s[0]
        self.old_s[1] = self.s[1]
        self.s[0] = state[0]
        self.s[1] = state[1]

    def update_state(self, state):
        """
        Updates the agent state.
        """
        self.old_s = self.s
        self.s = state

    def update_info_on_pipe(self, is_agent_seeing_the_pipe, first_step):
        self.flag_is_agent_seeing_the_pipe = is_agent_seeing_the_pipe
        if self.flag_is_agent_seeing_the_pipe:
            measure_angle_pipe = 0 + np.random.normal(0, self.std_dev_measure_pipe)
            if not first_step:
                self.weight_measure = self.forgetting_factor * self.weight_measure + 1
            self.angle_pipe = (1-1/self.weight_measure)*self.angle_pipe + measure_angle_pipe/self.weight_measure
            self.vector_pipe = np.dot(compute_rotation_matrix(self.angle_pipe), np.array([1, 0]))

    def obtain_action_index_greedy_policy(self):
        """
        Selects the action to be taken by the agent and returns the correspondent index.
        Given the exploration rate, the action is selected according to a greedy policy.
        """
        state_indexes = self.obtain_state_indexes(self.s)
        best_value = np.max(self.Q[state_indexes[0], state_indexes[1], state_indexes[2]])
        best_actions = (self.Q[state_indexes[0], state_indexes[1], state_indexes[2]] == best_value)
        prob_actions = best_actions / np.sum(best_actions)

        return np.random.choice(range(self.K_a), p=np.reshape(prob_actions, -1))

    def update_action(self):
        """
        Updates the action taken by the agent, calling the previous method to select the action to be taken.
        """
        self.a = self.obtain_action_index_greedy_policy()

    def obtain_state_indexes(self, state):
        """
        Given the current state of the agent, computes and returns the correspondent indexes in the possible_states
        vector.
        """

        #       Neighbours state
        if state[0] == self.possible_states[-1]:  # no neighbours state
            state_index_neighbours = len(self.possible_states) - 1
        else:
            state_index_neighbours = np.where(self.possible_states == state[0])

        if state[1] == self.possible_states[-1]:  # no neighbours state
            state_index_pipe = len(self.possible_states) - 1
        else:
            state_index_pipe = np.where(self.possible_states == state[1])

        return [state_index_neighbours, state_index_pipe, int(state[2])]

    def update_Q_matrix_exp_sarsa(self, reward):
        """
        Updates the Q matrix of the agent, according to the given learning rate and reward.
        """
        self.r = reward
        old_state_indexes = self.obtain_state_indexes(self.old_s)
        self.Q_visits[old_state_indexes[0], old_state_indexes[1], old_state_indexes[2]] += 1
        self.state_action_rate_visits[old_state_indexes[0], old_state_indexes[1], old_state_indexes[2], self.a] += 1

    def obtain_policy_probabilities(self, state_indexes, exploration_rate):
        policy = np.ones(self.K_a) / self.K_a * exploration_rate
        best_value = np.max(self.Q[state_indexes[0], state_indexes[1], state_indexes[2]])
        best_actions = (self.Q[state_indexes[0], state_indexes[1], state_indexes[2]] == best_value)
        policy += np.reshape(best_actions / np.sum(best_actions) * (1 - exploration_rate), -1)
        return policy

    def is_point_in_field_of_view(self, position_candidate):
        """
        Boolean auxiliary method to find if another agent is in the field of view of the current one,
        given its position.
        """
        angle_candidate = np.arctan2(position_candidate[1] - self.p[1], position_candidate[0] - self.p[0])
        if angle_candidate < 0:
            angle_candidate += 2 * pi

        if self.flag_0_in_interval:
            return self.start_angle_fov <= angle_candidate <= 2 * pi or 0 <= angle_candidate <= self.end_angle_fov
        else:
            return self.start_angle_fov <= angle_candidate <= self.end_angle_fov
