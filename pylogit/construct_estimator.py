"""
@name:      Estimator Constructor
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for constructing the Estimation
            Objects used to provide convenience functions when estimating
            PyLogit's various choice models.
"""
from __future__ import absolute_import

from functools import partial
import warnings
import numpy as np
from scipy.sparse import diags

from . import choice_calcs as cc
from . import base_multinomial_cm_v2 as base_mcm
from .display_names import model_type_to_display_name as display_name_dict

from .mixed_logit import MixedEstimator
from .mixed_logit import split_param_vec as mixed_split_params

from .nested_logit import NestedEstimator
from .nested_logit import split_param_vec as nested_split_params

from .conditional_logit import MNLEstimator
from .conditional_logit import split_param_vec as mnl_split_params

from .clog_log import ClogEstimator
from .clog_log import split_param_vec as clog_split_params

from .asym_logit import AsymEstimator
from .asym_logit import split_param_vec as asym_split_params

from .scobit import ScobitEstimator
from .scobit import split_param_vec as scobit_split_params

from .uneven_logit import UnevenEstimator
from .uneven_logit import split_param_vec as uneven_split_params

try:
    # in Python 3 range returns an iterator instead of list
    # to maintain backwards compatibility use "old" version of range
    from past.builtins import range
except ImportError:
    pass

# Map the displayed model types to the internal model names.
display_name_to_model_type = {v : k for k, v in display_name_dict.items()}

# Map the internal model types to their appropriate estimator and split params
# functions
model_type_to_resources =\
    {"MNL": {'estimator': MNLEstimator, 'split_func': mnl_split_params},
     "Asym": {'estimator': AsymEstimator, 'split_func': asym_split_params},
     "Cloglog": {'estimator': ClogEstimator, 'split_func': asym_split_params},
     "Scobit": {'estimator': ScobitEstimator,
                'split_func': scobit_split_params},
     "Uneven": {'estimator': UnevenEstimator,
                'split_func': uneven_split_params},
     "Nested Logit": {'estimator': NestedEstimator,
                      'split_func': nested_split_params},
     "Mixed Logit": {'estimator': MixedEstimator,
                     'split_func': mixed_split_params}}


def create_estimation_obj(model_obj,
                          init_vals,
                          mappings=None,
                          ridge=None,
                          constrained_pos=None,
                          weights=None):
    """
    Should return a model estimation object corresponding to the model type of
    the `model_obj`.

    Parameters
    ----------
    model_obj : an instance or sublcass of the MNDC class.
    init_vals : 1D ndarray.
        The initial values to start the optimization process with. In the
        following order, there should be one value for each nest coefficient,
        shape parameter, outside intercept parameter, or index coefficient that
        is being estimated.
    mapping_dict : OrderedDict or None, optional.
        Keys will be `["rows_to_obs", "rows_to_alts", "chosen_row_to_obs",
        "rows_to_nests"]`. The value for `rows_to_obs` will map the rows of
        the `long_form` to the unique observations (on the columns) in
        their order of appearance. The value for `rows_to_alts` will map
        the rows of the `long_form` to the unique alternatives which are
        possible in the dataset (on the columns), in sorted order--not
        order of appearance. The value for `chosen_row_to_obs`, if not
        None, will map the rows of the `long_form` that contain the chosen
        alternatives to the specific observations those rows are associated
        with (denoted by the columns). The value of `rows_to_nests`, if not
        None, will map the rows of the `long_form` to the nest (denoted by
        the column) that contains the row's alternative. Default == None.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If a
        scalar is passed, then that scalar determines the ridge penalty for
        the optimization. The scalar should be greater than or equal to
        zero. Default `== None`.
    constrained_pos : list or None, optional.
        Denotes the positions of the array of estimated parameters that are
        not to change from their initial values. If a list is passed, the
        elements are to be integers where no such integer is greater than
        `init_vals.size.` Default == None.
    weights : 1D ndarray.
        Should contain the weights for each corresponding observation for each
        row of the long format data.
    """
    # Get the mapping matrices for each model
    mapping_matrices =\
        model_obj.get_mappings_for_fit() if mappings is None else mappings
    # Create the zero vector for each model.
    zero_vector = np.zeros(init_vals.shape[0])
    # Get the internal model name
    internal_model_name = display_name_to_model_type[model_obj.model_type]
    # Get the split parameter function and estimator class for this model.
    estimator_class, current_split_func =\
        (model_type_to_resources[internal_model_name]['estimator'],
         model_type_to_resources[internal_model_name]['split_func'])
    # Create the estimator instance that is desired.
    estimation_obj = estimator_class(model_obj,
                                     mapping_matrices,
                                     ridge,
                                     zero_vector,
                                     current_split_func,
                                     constrained_pos,
                                     weights=weights)
    # Return the created object
    return estimation_obj
