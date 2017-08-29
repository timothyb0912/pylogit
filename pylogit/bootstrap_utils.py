"""
@author:    Timothy Brathwaite
@name:      Bootstrap Utilities
@summary:   This module provides helpful functions for calculating the
            bootstrap confidence intervals.
"""
from numbers import Number
import numpy as np


def check_conf_percentage_validity(conf_percentage):
    condition_1 = isinstance(conf_percentage, Number)
    condition_2 = 0 < conf_percentage < 100
    if not (condition_1 and condition_2):
        msg = "conf_percentage MUST be a number between 0.0 and 100."
        raise ValueError(msg)
    return None


def ensure_samples_is_2d_ndarray(samples, name='bootstrap'):
    condition_1 = isinstance(samples, np.ndarray)
    condition_2 = samples.ndim == 2
    if not (condition_1 and condition_2):
        sample_name = name + "_samples"
        msg = "`{}` MUST be a 2D ndarray.".format(sample_name)
        raise ValueError(msg)
    return None


def get_alpha_from_conf_percentage(conf_percentage):
    return 100.0 - conf_percentage


def combine_conf_endpoints(lower_array, upper_array):
    return np.concatenate([lower_array[None, :], upper_array[None, :]], axis=0)
