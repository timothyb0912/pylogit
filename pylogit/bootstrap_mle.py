"""
@author:    Timothy Brathwaite
@name:      Bootstrap Estimation Procedures
@summary:   This module provides functions that will perform the MLE for each
            of the bootstrap samples.
"""
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd

from . import pylogit as pl
from .display_names import model_type_to_display_name

try:
    # Python 3.x does not natively support xrange
    from past.builtins import xrange
except ImportError:
    pass


def extract_default_init_vals(orig_model_obj, mnl_point_series, num_params):
    """
    Get the default initial values for the desired model type, based on the
    point estimate of the MNL model that is 'closest' to the desired model.

    Parameters
    ----------
    orig_model_obj : an instance or sublcass of the MNDC class.
        Should correspond to the actual model that we want to bootstrap.
    mnl_point_series : pandas Series.
        Should denote the point estimate from the MNL model that is 'closest'
        to the desired model.
    num_params : int.
        Should denote the number of parameters being estimated (including any
        parameters that are being constrained during estimation).

    Returns
    -------
    init_vals : 1D ndarray of initial values for the MLE of the desired model.
    """
    # Initialize the initial values
    init_vals = np.zeros(num_params)
    # Figure out which values in mnl_point_series are the index coefficients
    no_outside_intercepts = orig_model_obj.intercept_names is None
    if no_outside_intercepts:
        init_index_coefs = mnl_point_series.values
        init_intercepts = None
    else:
        init_index_coefs =\
            mnl_point_series[orig_model_obj.ind_var_names].values
        init_intercepts =\
            mnl_point_series[orig_model_obj.intercept_names].values

    # Combine the initial interept values with the initial index coefficients
    if init_intercepts is not None:
        init_index_coefs =\
            np.concatenate([init_intercepts, init_index_coefs], axis=0)

    # Add any mixing variables to the index coefficients.
    if orig_model_obj.mixing_vars is not None:
        num_mixing_vars = len(orig_model_obj.mixing_vars)
        init_index_coefs = np.concatenate([init_index_coefs,
                                           np.zeros(num_mixing_vars)],
                                          axis=0)

    # Add index coefficients (and mixing variables) to the total initial array
    num_index = init_index_coefs.shape[0]
    init_vals[-1 * num_index:] = init_index_coefs

    # Note that the initial values for the transformed nest coefficients and
    # the shape parameters is zero so we don't have to change anything
    return init_vals


def get_model_abbrev(model_obj):
    """
    Extract the string used to specify the model type of this model object in
    `pylogit.create_chohice_model`.
    """
    # Get the 'display name' for our model.
    model_type = model_obj.model_type
    # Find the model abbreviation for this model's display name.
    for key in model_type_to_display_name:
        if model_type_to_display_name[key] == model_type:
            return key
    # If none of the strings in model_type_to_display_name matches our model
    # object, then raise an error.
    msg = "Model object has an unknown or incorrect model type."
    raise ValueError(msg)
    return None


def get_model_creation_kwargs(model_obj):
    """
    Get a dictionary of the keyword arguments needed to create the passed model
    object using `pylogit.create_choice_model`.
    """
    # Extract the model abbreviation for this model
    model_abbrev = get_model_abbrev(model_obj)

    # Create a dictionary to store the keyword arguments needed to Initialize
    # the new model object.d
    model_kwargs = {"model_type": model_abbrev,
                    "names": model_obj.name_spec,
                    "intercept_names": model_obj.intercept_names,
                    "intercept_ref_pos": model_obj.intercept_ref_position,
                    "shape_names": model_obj.shape_names,
                    "shape_ref_position": model_obj.shape_ref_position
                    "nest_spec": model_obj.nest_spec,
                    "mixing_vars": model_obj.mixing_vars,
                    "mixing_id_col": model_obj.mixing_id_col}

    return model_kwargs


def retrieve_point_est(orig_model_obj,
                       new_df,
                       num_params,
                       mnl_spec,
                       mnl_names,
                       mnl_init_vals,
                       mnl_fit_kwargs,
                       extract_init_vals=None,
                       **fit_kwargs):
    # Get the original specification and name dictionaries.
    advanced_spec = orig_model_obj.specification
    advanced_names = orig_model_obj.name_spec

    # Get specification and name dictionaries for the mnl model, for the case
    # where the model being bootstrapped is an MNL model. In this case, the
    # the mnl_spec and the mnl_names that are passed to the function are
    # expected to be None.
    if orig_model_obj.model_type == model_type_to_display_name["MNL"]:
        mnl_spec = advanced_spec
        mnl_names = advanced_names
        if mnl_init_vals is None:
            mnl_init_vals = np.zeros(num_params)
        if mnl_fit_kwargs is None:
            mnl_fit_kwargs = {}

    # Alter the mnl_fit_kwargs to ensure that we only perform point estimation
    mnl_fit_kwargs["just_point"] = True
    # Use BFGS by default to estimate the MNL since it works well for the MNL.
    if "method" not in mnl_fit_kwargs:
        mnl_fit_kwargs["method"] = "BFGS"

    # Initialize the mnl model object for the given bootstrap sample.
    mnl_obj = pl.create_choice_model(data=new_df,
                                     alt_id_col=orig_model_obj.alt_id_col,
                                     obs_id_col=orig_model_obj.obs_id_col
                                     choice_col=orig_model_obj.choice_col,
                                     specification=mnl_spec,
                                     model_type="MNL",
                                     names=mnl_names)

    # Get the MNL point estimate for the parameters of this bootstrap sample.
    mnl_point = mnl_obj.fit_mle(mnl_init_vals, **mnl_fit_kwargs)
    mnl_point_series = pd.Series(mnl_point["x"], index=mnl_obj.ind_var_names)

    # Denote the MNL point estimate as our final point estimate if the final
    # model we're interested in is an MNL.
    if orig_model_obj.model_type == model_type_to_display_name["MNL"]:
        final_point = mnl_point
    else:
        # Determine the function to be used when extracting the initial values
        # for the final model from the MNL MLE point estimate.
        if extract_init_vals is None:
            extraction_func = extract_default_init_vals
        else:
            extraction_func = extract_init_vals

        # Extract the initial values
        default_init_vals =\
            extraction_func(orig_model_obj, mnl_point_series, num_params)

        # Get the keyword arguments needed to initialize the new model object.
        model_kwargs = get_model_creation_kwargs(orig_model_obj)

        # Create a new model object
        new_obj =\
            pl.create_choice_model(data=new_df,
                                   alt_id_col=orig_model_obj.alt_id_col,
                                   obs_id_col=orig_model_obj.obs_id_col
                                   choice_col=orig_model_obj.choice_col,
                                   specification=orig_model_obj.specification,
                                   **model_kwargs)

        # Fit the model with new data, and return the point estimate dict.
        final_point = new_obj.fit_mle(default_init_vals, **fit_kwargs)

    return final_point
