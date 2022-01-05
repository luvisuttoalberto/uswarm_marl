import numpy as np
from math import floor


def compute_rotation_matrix(theta):
    """
    Computes a rotation matrix.
    theta is the angle of the rotation to be applied.
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def exploration_rate_adaptive(t, epsilon_0, t_star, t_stop):
    """
    Returns the exploration rate at a given timestep t.
    Function is constant up to timestep t_star, then decreases.
    """
    if t < t_star:
        return epsilon_0
    elif t < t_stop:
        # return epsilon_0 / ((t / t_star) ** 1.5)
        # return epsilon_0 / (1. + 1.5 * (t / t_star) ** 0.5) # longer_old_exp
        # return epsilon_0 / (1.5 * (t / t_star) ** 0.5) # longer_no_step_old_exp
        return epsilon_0 / (1.5*(t / t_star) ** 0.5) # no_step_no_scale_old_exp # REDO WITH t_star = 300
    else:
        return 0


def learning_rate_adaptive(t, alpha_0, t_star, t_stop):
    """
    Returns the learning rate at a given timestep t.
    Function is constant up to timestep t_star, then decreases.
    """
    if t < t_star:
        return alpha_0
    elif t < t_stop:
        return alpha_0 / (1.5*(t / t_star) ** 0.5)
    else:
        return 0


def moving_average(a, n):
    """
    Computes the moving average on n timesteps of some array a.
    Used in plots.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def is_scalar_in_visible_interval(x, array, length_intervals, flag_spatially_uncorrelated_case):
    if flag_spatially_uncorrelated_case:
        return True
    else:
        return x >= 0 and array[floor(x / length_intervals)]
