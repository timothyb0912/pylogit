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
    init_vals = np.zeros(num_params, dtype=float)
    # Figure out which values in mnl_point_series are the index coefficients
    no_outside_intercepts = orig_model_obj.intercept_names is None
    if no_outside_intercepts:
        init_index_coefs = mnl_point_series.values
        init_intercepts = None
    else:
        init_index_coefs =\
            mnl_point_series.loc[orig_model_obj.ind_var_names].values
        init_intercepts =\
            mnl_point_series.loc[orig_model_obj.intercept_names].values

    # Add any mixing variables to the index coefficients.
    if orig_model_obj.mixing_vars is not None:
        num_mixing_vars = len(orig_model_obj.mixing_vars)
        init_index_coefs = np.concatenate([init_index_coefs,
                                           np.zeros(num_mixing_vars)],
                                          axis=0)

    # Account for the special transformation of the index coefficients that is
    # needed for the asymmetric logit model.
    if orig_model_obj.model_type == model_type_to_display_name["Asym"]:
        multiplier = np.log(len(np.unique(orig_model_obj.alt_IDs)))
        # Cast the initial index coefficients to a float dtype to ensure
        # successful broadcasting
        init_index_coefs = init_index_coefs.astype(float)
        # Adjust the scale of the index coefficients for the asymmetric logit.
        init_index_coefs /= multiplier

    # Combine the initial interept values with the initial index coefficients
    if init_intercepts is not None:
        init_index_coefs =\
            np.concatenate([init_intercepts, init_index_coefs], axis=0)

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

    Parameters
    ----------
    model_obj : An MNDC_Model instance.

    Returns
    -------
    str. The internal abbreviation used for the particular type of MNDC_Model.
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


def get_model_creation_kwargs(model_obj):
    """
    Get a dictionary of the keyword arguments needed to create the passed model
    object using `pylogit.create_choice_model`.

    Parameters
    ----------
    model_obj : An MNDC_Model instance.

    Returns
    -------
    model_kwargs : dict.
        Contains the keyword arguments and the required values that are needed
        to initialize a replica of `model_obj`.
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
                    "shape_ref_pos": model_obj.shape_ref_position,
                    "nest_spec": model_obj.nest_spec,
                    "mixing_vars": model_obj.mixing_vars,
                    "mixing_id_col": model_obj.mixing_id_col}

    return model_kwargs


def get_mnl_point_est(orig_model_obj,
                      new_df,
                      num_params,
                      mnl_spec,
                      mnl_names,
                      mnl_init_vals,
                      mnl_fit_kwargs):
    """
    Calculates the MLE for the desired MNL model.

    Parameters
    ----------
    orig_model_obj : An MNDC_Model instance.
        The object corresponding to the desired model being bootstrapped.
    new_df : pandas DataFrame.
        The pandas dataframe containing the data to be used to estimate the
        MLE of the MNL model for the current bootstrap sample.
    num_params : non-negative int.
        The number of parameters in the MLE of the `orig_model_obj`.
    mnl_spec : OrderedDict or None.
        If `orig_model_obj` is not a MNL model, then `mnl_spec` should be an
        OrderedDict that contains the specification dictionary used to estimate
        the MNL model that will provide starting values for the final estimated
        model. If `orig_model_obj` is a MNL model, then `mnl_spec` may be None.
    mnl_names : OrderedDict or None.
        If `orig_model_obj` is not a MNL model, then `mnl_names` should be an
        OrderedDict that contains the name dictionary used to initialize the
        MNL model that will provide starting values for the final estimated
        model. If `orig_model_obj` is a MNL, then `mnl_names` may be None.
    mnl_init_vals : 1D ndarray or None.
        If `orig_model_obj` is not a MNL model, then `mnl_init_vals` should be
        a 1D ndarray. `mnl_init_vals` should denote the initial values used to
        estimate the MNL model that provides starting values for the final
        desired model. If `orig_model_obj` is a MNL model, then `mnl_init_vals`
        may be None.
    mnl_fit_kwargs : dict or None.
        If `orig_model_obj` is not a MNL model, then `mnl_fit_kwargs` should be
        a dict. `mnl_fit_kwargs` should denote the keyword arguments used when
        calling the `fit_mle` function of the MNL model that will provide
        starting values to the desired choice model. If `orig_model_obj` is a
        MNL model, then `mnl_fit_kwargs` may be None.

    Returns
    -------
    mnl_point : dict.
        The dictionary returned by `scipy.optimize` after estimating the
        desired MNL model.
    mnl_obj : An MNL model instance.
        The model object used to estimate the desired MNL model.
    """
    # Get specification and name dictionaries for the mnl model, for the case
    # where the model being bootstrapped is an MNL model. In this case, the
    # the mnl_spec and the mnl_names that are passed to the function are
    # expected to be None.
    if orig_model_obj.model_type == model_type_to_display_name["MNL"]:
        mnl_spec = orig_model_obj.specification
        mnl_names = orig_model_obj.name_spec
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
                                     obs_id_col=orig_model_obj.obs_id_col,
                                     choice_col=orig_model_obj.choice_col,
                                     specification=mnl_spec,
                                     model_type="MNL",
                                     names=mnl_names)

    # Get the MNL point estimate for the parameters of this bootstrap sample.
    mnl_point = mnl_obj.fit_mle(mnl_init_vals, **mnl_fit_kwargs)
    return mnl_point, mnl_obj

def retrieve_point_est(orig_model_obj,
                       new_df,
                       num_params,
                       mnl_spec,
                       mnl_names,
                       mnl_init_vals,
                       mnl_fit_kwargs,
                       extract_init_vals=None,
                       **fit_kwargs):
    """
    Calculates the MLE for the desired MNL model.

    Parameters
    ----------
    orig_model_obj : An MNDC_Model instance.
        The object corresponding to the desired model being bootstrapped.
    new_df : pandas DataFrame.
        The pandas dataframe containing the data to be used to estimate the
        MLE of the MNL model for the current bootstrap sample.
    num_params : non-negative int.
        The number of parameters in the MLE of the `orig_model_obj`.
    mnl_spec : OrderedDict or None.
        If `orig_model_obj` is not a MNL model, then `mnl_spec` should be an
        OrderedDict that contains the specification dictionary used to estimate
        the MNL model that will provide starting values for the final estimated
        model. If `orig_model_obj` is a MNL model, then `mnl_spec` may be None.
    mnl_names : OrderedDict or None.
        If `orig_model_obj` is not a MNL model, then `mnl_names` should be an
        OrderedDict that contains the name dictionary used to initialize the
        MNL model that will provide starting values for the final estimated
        model. If `orig_model_obj` is a MNL, then `mnl_names` may be None.
    mnl_init_vals : 1D ndarray or None.
        If `orig_model_obj` is not a MNL model, then `mnl_init_vals` should be
        a 1D ndarray. `mnl_init_vals` should denote the initial values used to
        estimate the MNL model that provides starting values for the final
        desired model. If `orig_model_obj` is a MNL model, then `mnl_init_vals`
        may be None.
    mnl_fit_kwargs : dict or None.
        If `orig_model_obj` is not a MNL model, then `mnl_fit_kwargs` should be
        a dict. `mnl_fit_kwargs` should denote the keyword arguments used when
        calling the `fit_mle` function of the MNL model that will provide
        starting values to the desired choice model. If `orig_model_obj` is a
        MNL model, then `mnl_fit_kwargs` may be None.
    extract_init_vals : callable or None, optional.
        Should accept 3 arguments, in the following order. First, it should
        accept `orig_model_obj`. Second, it should accept a pandas Series of
        the estimated parameters from the MNL model. The index of the Series
        will be the names of the coefficients from `mnl_names`. Thirdly, it
        should accept an int denoting the number of parameters in the desired
        choice model. The callable should return a 1D ndarray of starting
        values for the desired choice model. Default == None.
    fit_kwargs : dict.
        Denotes the keyword arguments to be used when estimating the desired
        choice model using the current bootstrap sample (`new_df`). All such
        kwargs will be directly passed to the `fit_mle` method of the desired
        model object.

    Returns
    -------
    final_point : dict.
        The dictionary returned by `scipy.optimize` after estimating the
        desired choice model.
    """
    # Get the MNL point estimate for the parameters of this bootstrap sample.
    mnl_point, mnl_obj = get_mnl_point_est(orig_model_obj,
                                           new_df,
                                           num_params,
                                           mnl_spec,
                                           mnl_names,
                                           mnl_init_vals,
                                           mnl_fit_kwargs)
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
                                   obs_id_col=orig_model_obj.obs_id_col,
                                   choice_col=orig_model_obj.choice_col,
                                   specification=orig_model_obj.specification,
                                   **model_kwargs)

        # Be sure to add 'just_point' to perform pure point estimation.
        if 'just_point' not in fit_kwargs:
            fit_kwargs['just_point'] = True

        # Fit the model with new data, and return the point estimate dict.
        final_point = new_obj.fit_mle(default_init_vals, **fit_kwargs)

    return final_point
