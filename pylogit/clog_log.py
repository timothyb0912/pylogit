# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 15:45:08 2016

@name:      MultiNomial Clog-log
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating multinomial clog-log
            models (with the help of the "base_multinomial_cm.py" file).
            Differs from version 1 because it partitions the vector of
            parameter estimates, theta, into (shape, intercept, index)
            parameters.
"""

from functools import partial
import time
import sys
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags

import choice_calcs as cc
import base_multinomial_cm_v2 as base_mcm

# Define the boundary values which are not to be exceeded ducing computation
max_comp_value = 1e300
min_comp_value = 1e-300

max_exp = 700
min_exp = -700

# Create a variable that will be printed if there is a non-fatal error
# in the MNL class construction
_msg_1 = "The Multinomial Clog-Log Model has no shape parameters. "
_msg_2 = "shape_names and shape_ref_pos will be ignored if passed."
_shape_ignore_msg = _msg_1 + _msg_2

# Create a warning string that will be issued if ridge regression is performed.
_msg_3 = "NOTE: An L2-penalized regression is being performed. The "
_msg_4 = "reported standard errors and robust standard errors "
_msg_5 = "***WILL BE INCORRECT***."
_ridge_warning_msg = _msg_3 + _msg_4 + _msg_5

# Alias necessary functions from the base multinomial choice model module
general_log_likelihood = cc.calc_log_likelihood
general_gradient = cc.calc_gradient
general_calc_probabilities = cc.calc_probabilities
general_hessian = cc.calc_hessian


def split_param_vec(param_vec, rows_to_alts, design):
    """
    Parameters
    ----------
    param_vec : 1D ndarray.
        Elements should all be ints, floats, or longs. Should have as many
        elements as there are parameters being estimated.
    rows_to_alts : 2D scipy sparse matrix.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset. All
        elements should be zeros or ones.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.

    Returns
    -------
    `(None, intercepts, betas)` : tuple.
        The first element will be None since the clog-log model has no shape
        parameters. The second element will either be a 1D array of "outside"
        intercept parameters for this model or None, depending on whether
        outside intercepts are being estimated or not. The third element will
        be a 1D array of the index coefficients.
    """
    # Figure out how many parameters are in the index
    num_index_coefs = design.shape[1]

    # Isolate the initial shape parameters from the betas
    betas = param_vec[-1 * num_index_coefs:]

    # Get the remaining outside intercepts if there are any
    remaining_idx = param_vec.shape[0] - num_index_coefs
    if remaining_idx > 0:
        intercepts = param_vec[:remaining_idx]
    else:
        intercepts = None

    return None, intercepts, betas


def _cloglog_utility_transform(systematic_utilities,
                               alt_IDs,
                               rows_to_alts,
                               shape_params,
                               intercept_params,
                               intercept_ref_pos=None,
                               *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities : 1D ndarray.
        All elements should be ints, floats, or longs. Should contain the
        systematic utilities of each observation per available alternative.
        Note that this vector is formed by the dot product of the design matrix
        with the vector of utility coefficients.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_alts : 2D scipy sparse matrix.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset. All
        elements should be zeros or ones.
    shape_params : None or 1D ndarray.
        Should be None since the clog-log model has no shape parameters.
    intercept_params : None or 1D ndarray.
        If an array, each element should be an int, float, or long. If J is the
        total number of possible alternatives for the dataset being modeled,
        there should be J-1 elements in the array. Use None if no outside
        intercepts are being estimated.
    intercept_ref_pos : int, or None, optional.
        Specifies the index of the alternative, in the ordered array of unique
        alternatives, that is not having its intercept parameter estimated (in
        order to ensure identifiability). Should only be None if
        `intercept_params` is None.

    Returns
    -------
    transformations : 2D ndarray.
        Should have shape `(systematic_utilities.shape[0], 1)`. The returned
        array contains the transformed utility values for this model. All
        elements will be ints, longs, or floats.
    """
    # Calculate the data dependent part of the transformation
    # Also, along the way, guard against numeric underflow or overflow
    exp_v = np.exp(systematic_utilities)
    # exp_v[np.isposinf(exp_v)] = max_comp_value

    exp_exp_v = np.exp(exp_v)
    # exp_exp_v[np.isposinf(exp_exp_v)] = max_comp_value

    # Calculate the transformed systematic utilities
    transformations = np.log(exp_exp_v - 1)
    # Guard against underflow
    transformations[np.isneginf(transformations)] = -1 * max_comp_value
    # Guard against overflow when systematic utilities are moderately large
    too_big_idx = np.where(systematic_utilities >= 3.7)
    transformations[too_big_idx] = np.exp(systematic_utilities[too_big_idx])
    # Guard against overflow when systematic utilities are completely too big.
    inf_idx = np.isposinf(transformations)
    transformations[inf_idx] = max_comp_value


    # Account for the outside intercept parameters if there are any.
    if intercept_params is not None and intercept_ref_pos is not None:
        # Get a list of all the indices (or row indices) corresponding to the
        # alternatives whose intercept parameters are being estimated.
        needed_idxs = range(rows_to_alts.shape[1])
        needed_idxs.remove(intercept_ref_pos)

        if len(intercept_params.shape) > 1 and intercept_params.shape[1] > 1:
            # Get an array of zeros with shape
            # (num_possible_alternatives, num_parameter_samples)
            all_intercepts = np.zeros((rows_to_alts.shape[1],
                                       intercept_params.shape[1]))
            # For alternatives having their intercept estimated, replace the
            # zeros with the current value of the estimated intercepts
            all_intercepts[needed_idxs, :] = intercept_params
        else:
            # Get an array of zeros with shape (num_possible_alternatives,)
            all_intercepts = np.zeros(rows_to_alts.shape[1])
            # For alternatives having their intercept estimated, replace the
            # zeros with the current value of the estimated intercepts
            all_intercepts[needed_idxs] = intercept_params

        # Add the intercept values to f(x, beta, c)
        transformations += rows_to_alts.dot(all_intercepts)

    # Be sure to return a 2D array since other functions will be expecting that
    if len(transformations.shape) == 1:
        transformations = transformations[:, None]

    return transformations


def _cloglog_transform_deriv_v(systematic_utilities,
                               alt_IDs,
                               rows_to_alts,
                               shape_params,
                               output_array=None,
                               *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities : 1D ndarray.
        All elements should be ints, floats, or longs. Should contain the
        systematic utilities of each observation per available alternative.
        Note that this vector is formed by the dot product of the design matrix
        with the vector of utility coefficients.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_alts : 2D scipy sparse matrix.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset. All
        elements should be zeros or ones.
    shape_params : None or 1D ndarray.
        If an array, each element should be an int, float, or long. There
        should be one value per shape parameter of the model being used.
    output_array : 2D scipy sparse array.
        The array should be square and it should have
        `systematic_utilities.shape[0]` rows. It's data is to be replaced with
        the correct derivatives of the transformation vector with respect to
        the vector of systematic utilities. This argument is NOT optional.

    Returns
    -------
    output_array : 2D scipy sparse array.
        The shape of the returned array is `(systematic_utilities.shape[0],
        systematic_utilities.shape[0])`. The returned array specifies the
        derivative of the transformed utilities with respect to the systematic
        utilities. All elements are ints, floats, or longs.
    """
    exp_neg_v = np.exp(-1 * systematic_utilities)
    exp_v = np.exp(systematic_utilities)
    denom_part_1 = 1 - np.exp(-1 * exp_v)

    ##########
    # Guard against numeric overflow and underflow
    ##########
    exp_neg_v[np.isposinf(exp_neg_v)] = max_comp_value
    exp_neg_v[np.where(exp_neg_v == 0)] = min_comp_value
    # Note that we don't care about the limiting cases of exp_v.
    # This term can go to positive infinity or zero. If it goes to positive
    # infinity, then this is okay because denom_part_1 will just go to 1.
    # If exp_v goes to zero, then denom_part_1 will go to zero. We will simply
    # cater to that last outcome since we can't divide by zero. The next line
    # is retained to show what should NOT be done. We will use L'Hopital's rule
    # after calculating derivs, as should be done.
    # denom_part_1[np.where(denom_part_1 == 0)] = min_comp_value

    ##########
    # Calculate the required derivatives and guard against underflow
    ##########
    derivs = 1.0 / (denom_part_1 * exp_neg_v)
    # Note that the limiting value of the expression above, as the systematic
    # utility goes to negative infinity (i.e. as denom_part_1 goes to zero),
    # is one. This can be checked using L'Hopital's rule. We will define
    # infinity as being so negative that `denom_part_1 == 0`
    derivs[np.where(denom_part_1 == 0)] = 1
    derivs[np.isposinf(derivs)] = max_comp_value

    # Assign the calculated derivatives to the output array
    output_array.data = derivs

    # Return the matrix of dh_dv. Note the off-diagonal entries are zero
    # because each transformation only depends on its value of v and no others
    return output_array


def _cloglog_transform_deriv_c(*args, **kwargs):
    """
    Returns
    -------
    None. This is a place holder function since the Clog-log model has no shape
    parameters.
    """

    return None


def _cloglog_transform_deriv_alpha(systematic_utilities,
                                   alt_IDs,
                                   rows_to_alts,
                                   intercept_params,
                                   output_array=None,
                                   *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities : 1D ndarray.
        All elements should be ints, floats, or longs. Should contain the
        systematic utilities of each observation per available alternative.
        Note that this vector is formed by the dot product of the design matrix
        with the vector of utility coefficients.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_alts : 2D scipy sparse matrix.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset. All
        elements should be zeros or ones.
    intercept_params : 1D ndarray or None.
        If an array, each element should be an int, float, or long. For
        identifiability, there should be J- 1 elements where J is the total
        number of observed alternatives for this dataset.
    output_array: None or 2D scipy sparse array.
        If a sparse array is pased, it should contain the derivative of the
        vector of transformed utilities with respect to the intercept
        parameters outside of the index. This keyword argurment will be
        returned. If there are no intercept parameters outside of the index,
        then `output_array` should equal None. If there are intercept
        parameters outside of the index, then `output_array` should be
        `rows_to_alts` with the all of its columns except the column
        corresponding to the alternative whose intercept is not being estimated
        in order to ensure identifiability.

    Returns
    -------
    output_array.
    """
    return output_array


def _calc_neg_log_likelihood_and_neg_gradient(beta,
                                              design,
                                              alt_IDs,
                                              rows_to_obs,
                                              rows_to_alts,
                                              choice_vector,
                                              utility_transform,
                                              block_matrix_idxs,
                                              ridge,
                                              calc_dh_dv,
                                              calc_dh_dc,
                                              calc_dh_d_alpha,
                                              *args):
    """
    Parameters
    ----------
    beta : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features) and
        for each shape parameter being estimated.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_alts : 2D scipy sparse matrix.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset. All
        elements should be zeros or ones.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform : callable.
        Should accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 1D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated.
    block_matrix_idxs : list of arrays.
        There will be one array per column in `rows_to_obs`. The arrays will
        note which rows correspond to which observations.
    ridge : int, float, long, or None.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization.
    calc_dh_dv : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of systematic utilities. The dimensions of the returned vector should
        be `(design.shape[0], design.shape[0])`.
    calc_dh_dc : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of systematic utilities. The dimensions of the returned vector should
        be `(design.shape[0], rows_to_alts.shape[1])`.
    calc_dh_d_alpha : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, a 2D sparse scipy matrix mapping rows of the design
        matrix to the alternatives, and a 1D array of intercept parameters, as
        well as miscellaneous args and kwargs. If there are intercept
        parameters, the callable should return a 2D array whose elements
        contain the derivative of the tranformed utility vector with respect to
        the vector of intercept parameters. The dimensions of the returned
        vector should be `(design.shape[0], rows_to_alts.shape[1] - 1)`. If
        there are no 'outside' intercept parameters, the callable should return
        None.

    Returns
    -------
    `(neg_log_likelihood, neg_gradient_vec)` : tuple.
        The first element is a float. The second element is a 1D numpy array of
        shape `== beta.shape`. The first element is the negative log-likelihood
        of this model evaluated at the passed values of beta. The second
        element is the gradient of the negative log-likelihood with respect to
        the vector of shape parameters and utility coefficients.
    """
    # Separate the shape, intercept, and beta parameters
    shape_vec, intercept_vec, coefficient_vec = split_param_vec(beta,
                                                                rows_to_alts,
                                                                design)

    # Calculate the needed quantities
    neg_log_likelihood = -1 * general_log_likelihood(coefficient_vec,
                                                     design,
                                                     alt_IDs,
                                                     rows_to_obs,
                                                     rows_to_alts,
                                                     choice_vector,
                                                     utility_transform,
                                             intercept_params=intercept_vec,
                                                     shape_params=shape_vec,
                                                     ridge=ridge)

    neg_beta_gradient_vec = -1 * general_gradient(coefficient_vec,
                                                  design,
                                                  alt_IDs,
                                                  rows_to_obs,
                                                  rows_to_alts,
                                                  choice_vector,
                                                  utility_transform,
                                                  calc_dh_dc,
                                                  calc_dh_dv,
                                                  calc_dh_d_alpha,
                                                  intercept_vec,
                                                  shape_vec,
                                                  ridge)

    return neg_log_likelihood, neg_beta_gradient_vec


def _calc_neg_hessian(beta,
                      design,
                      alt_IDs,
                      rows_to_obs,
                      rows_to_alts,
                      choice_vector,
                      utility_transform,
                      block_matrix_idxs,
                      ridge,
                      calc_dh_dv,
                      calc_dh_dc,
                      calc_dh_d_alpha,
                      *args):
    """
    Parameters
    ----------
    beta : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features) and
        for each shape parameter being estimated.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_alts : 2D scipy sparse matrix.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset. All
        elements should be zeros or ones.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform : callable.
        Should accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 1D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated.
    block_matrix_idxs : list of arrays.
        There will be one array per column in `rows_to_obs`. The arrays will
        note which rows correspond to which observations.
    ridge : int, float, long, or None.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization.
    calc_dh_dv : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of systematic utilities. The dimensions of the returned vector should
        be `(design.shape[0], design.shape[0])`.
    calc_dh_dc : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of systematic utilities. The dimensions of the returned vector should
        be `(design.shape[0], rows_to_alts.shape[1])`.
    calc_dh_d_alpha : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, a 2D sparse scipy matrix mapping rows of the design
        matrix to the alternatives, and a 1D array of intercept parameters, as
        well as miscellaneous args and kwargs. If there are intercept
        parameters, the callable should return a 2D array whose elements
        contain the derivative of the tranformed utility vector with respect to
        the vector of intercept parameters. The dimensions of the returned
        vector should be `(design.shape[0], rows_to_alts.shape[1] - 1)`. If
        there are no 'outside' intercept parameters, the callable should return
        None.

    Returns
    -------
    neg_hessian : 2D ndarray.
        There will be as many rows (and columns) as there are transformed shape
        parameters, intercept parameters, and index coefficients, combined. All
        elements wil be ints, floats, or longs.
    """
    # Separate the shape, intercept, and beta parameters
    shape_vec, intercept_vec, coefficient_vec = split_param_vec(beta,
                                                                rows_to_alts,
                                                                design)

    # Calculate the hessian
    return -1 * general_hessian(coefficient_vec,
                                design,
                                alt_IDs,
                                rows_to_obs,
                                rows_to_alts,
                                utility_transform,
                                calc_dh_dc,
                                calc_dh_dv,
                                calc_dh_d_alpha,
                                block_matrix_idxs,
                                intercept_vec,
                                shape_vec,
                                ridge)


def _estimate(init_values, design_matrix, alt_id_vector,
              choice_vector, rows_to_obs, rows_to_alts,
              chosen_row_to_obs, intercept_ref_pos,
              print_results=True, method='newton-cg',
              loss_tol=1e-06, gradient_tol=1e-06,
              maxiter=1000, ridge=False,
              **kwargs):
    """
    Parameters
    ----------
    init_values : 1D ndarray.
        The initial values to start the optimizatin process with. There should
        be one value for each index coefficient, outside intercept parameter,
        and shape parameter being estimated.
    design_matrix : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    alt_id_vector : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_alts : 2D scipy sparse matrix.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset. All
        elements should be zeros or ones.
    chosen_row_to_obs :  2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix indicates, for each observation
        (on the columns), which rows of the design matrix were the realized
        outcome. Should have one and only one `1` in each column. No row should
        have more than one `1` though it is okay if a row is all zeros.
    intercept_ref_pos : int, or None.
        Should only be an int when the intercepts being estimated are not part
        of the index. Specifies the alternative in the ordered array of unique
        alternative ids whose intercept or alternative-specific constant is not
        estimated, to ensure model identifiability.
    print_res : bool, optional.
        Determines whether the timing and initial and final log likelihood
        results will be printed as they they are determined. Default `== True`.
    method : str, optional.
        Should be a valid string that can be passed to scipy.optimize.minimize.
        Determines the optimization algorithm which is used for this problem.
        Default `== 'bfgs'`.
    loss_tol : float, optional.
        Determines the tolerance on the difference in objective function values
        from one iteration to the next that is needed to determine convergence.
        Default `== 1e-06`.
    gradient_tol : float, optional.
        Determines the tolerance on the difference in gradient values from one
        iteration to the next which is needed to determine convergence.
        Default `== 1e-06`.
    maxiter : int, optional.
        Determines the maximum number of iterations used by the optimizer.
        Default `== 1000`.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If a scalar is
        passed, then that scalar determines the ridge penalty for the
        optimization. The scalar should be greater than or equal to zero.
        Default `== None`.
    kwargs : optional.
        Other keyword arguments given to this function will be passed directly
        to scipy.optimize.minimize().

    Returns
    -------
    results : dict.
        Result dictionary returned by `scipy.optimize.minimize`. In addition to
        the generic key-value pairs that are returned, `results` will have the
        folowing keys:
        - "final_log_likelihood"
        - "long_probs"
        - "residuals"
        - "ind_chi_squareds"
        - "simulated_sequence_probs"
        - "expanded_sequence_probs"
        - "utility_coefs"
        - "shape_params"
        - "intercept_params"
        - "nest_params"
        - "log_likelihood_null"
        - "rho_squared"
        - "rho_bar_squared"
        - "final_gradient"
        - "final_hessian"
        - "fisher_info"
        - "constrained_pos"
    """
    ##########
    # Make sure we have the correct dimensions for the initial parameter values
    ##########
    # Figure out how many shape parameters we should have and how many index
    # coefficients we should have
    num_alts = rows_to_alts.shape[1]
    num_index_coefs = design_matrix.shape[1]

    try:
        if intercept_ref_pos is not None:
            assumed_param_dimensions = num_index_coefs + num_alts - 1
        else:
            assumed_param_dimensions = num_index_coefs + num_alts
        assert init_values.shape[0] == assumed_param_dimensions
    except AssertionError as e:
        print("The initial values are of the wrong dimension")
        print("It should be of dimension {}".format(assumed_param_dimensions))
        print("But instead it has dimension {}".format(init_values.shape[0]))
        raise e

    ##########
    # Check other function arguments for 'correctness'
    ##########
    # Make sure the ridge regression parameter is None or a real scalar
    try:
        assert ridge is None or isinstance(ridge, (int, float, long))
    except AssertionError as e:
        print("ridge should be None or an int, float, or long.")
        print("The passed value of ridge had type: {}".format(type(ridge)))
        raise e

    ##########
    # Initialize needed matrices so we avoid the costly memory allocation
    # process within the rest of the calculations
    ##########
    pre_dh_dv = diags(np.ones(design_matrix.shape[0]), 0, format='csr')

    ##########
    # Create convenience functions needed to compute the necessary derivatives
    # in the estimation process and to compute the log-likelihood of the model
    ##########
    easy_calc_dh_dv = lambda *args: _cloglog_transform_deriv_v(*args,
                                              output_array=pre_dh_dv)

    easy_utility_transform = partial(_cloglog_utility_transform,
                                     intercept_ref_pos=intercept_ref_pos)

    # Pre-create the sparse matrix that will be used as the derivative of the
    # transformation vector with respect to the intercept parameters
    needed_idxs = range(rows_to_alts.shape[1])
    if intercept_ref_pos is not None:
        needed_idxs.remove(intercept_ref_pos)
        dh_d_alpha = rows_to_alts.copy().transpose()[needed_idxs,
                                                      :].transpose()
    else:
        dh_d_alpha = None
    calc_dh_d_alpha = lambda *args: _cloglog_transform_deriv_alpha(*args,
                                                       output_array=dh_d_alpha)

    ##########
    # Begin the model estimation process.
    ##########
    # Isolate the initial shape, intercept, and beta parameters.
    init_shapes, init_intercepts, init_betas = split_param_vec(init_values,
                                                               rows_to_alts,
                                                               design_matrix)

    # Get the log-likelihood at zero and the initial log likelihood
    # Note, we use intercept_params=None since this will cause the function
    # to think there are no intercepts being added to the transformation
    # vector, which is the same as adding zero to the transformation vector
    log_likelihood_at_zero = general_log_likelihood(
                                          np.zeros(design_matrix.shape[1]),
                                                             design_matrix,
                                                             alt_id_vector,
                                                                rows_to_obs,
                                                             rows_to_alts,
                                                             choice_vector,
                                                    easy_utility_transform,
                                                     intercept_params=None,
                                                               ridge=ridge)

    # Note the clog-log model has no shape parameters so there is no such
    # parameter included in the log-likelihood calculations below or above
    initial_log_likelihood = general_log_likelihood(init_betas,
                                                     design_matrix,
                                                     alt_id_vector,
                                                     rows_to_obs,
                                                     rows_to_alts,
                                                     choice_vector,
                                                   easy_utility_transform,
                                          intercept_params=init_intercepts,
                                                   ridge=ridge)
    ##########
    # Print initial model conditions
    # i.e., log-likelihoods at 'zero' and at initial values
    ##########
    if print_results:
        # Print the log-likelihood at zero
        print("Log-likelihood at zero: {:,.4f}".format(log_likelihood_at_zero))

        # Print the log-likelihood at the starting values
        print("Initial Log-likelihood: {:,.4f}".format(initial_log_likelihood))
        sys.stdout.flush()

    ##########
    # Perform the minimization to estimate the multinomial asymmetric logit
    ##########
    # Get the block matrix indices for the hessian matrix. Do it outside the
    # iterative minimization process in order to minimize unnecessary
    # computations
    block_matrix_indices = cc.create_matrix_block_indices(rows_to_obs)

    # Start timing the estimation process
    start_time = time.time()

    results = minimize(_calc_neg_log_likelihood_and_neg_gradient,
                       init_values,
                       args=(design_matrix,
                             alt_id_vector,
                             rows_to_obs,
                             rows_to_alts,
                             choice_vector,
                             easy_utility_transform,
                             block_matrix_indices,
                             ridge,
                             easy_calc_dh_dv,
                             _cloglog_transform_deriv_c,
                             calc_dh_d_alpha),
                       method=method,
                       jac=True,
                       hess=_calc_neg_hessian,
                       tol=loss_tol,
                       options={'gtol': gradient_tol,
                                "maxiter": maxiter},
                       **kwargs)

    #########
    # Store the raw and processed outputs of the estimation outputs
    #########
    # Calculate the final log-likelihood. Note the '-1' is because we minimized
    # the negative log-likelihood but we want the actual log-likelihood
    final_log_likelihood = -1 * results["fun"]

    # Stop timing the estimation process and report the timing results
    end_time = time.time()
    if print_results:
        elapsed_sec = (end_time - start_time)
        elapsed_min = elapsed_sec / 60.0
        if elapsed_min > 1.0:
            print("Estimation Time: {:.2f} minutes.".format(elapsed_min))
        else:
            print("Estimation Time: {:.2f} seconds.".format(elapsed_sec))
        print("Final log-likelihood: {:,.4f}".format(final_log_likelihood))
        sys.stdout.flush()

    # Isolate the final shape, intercept, and beta parameters
    split_res = split_param_vec(results.x, rows_to_alts, design_matrix)
    final_shape_params, final_intercept_params, final_utility_coefs = split_res

    # Store the separate values of the shape, intercept, and beta parameters
    # in the estimation results dict
    results["utility_coefs"] = final_utility_coefs
    results["intercept_params"] = final_intercept_params
    results["shape_params"] = final_shape_params
    results["nest_params"] = None

    # Calculate the predicted probabilities
    # Note we don't include shape parameters because the clog-log model doesn't
    # have shape parameters
    probability_results = general_calc_probabilities(
                                                        final_utility_coefs,
                                                              design_matrix,
                                                              alt_id_vector,
                                                                 rows_to_obs,
                                                              rows_to_alts,
                                                     easy_utility_transform,
                                    intercept_params=final_intercept_params,
                                        chosen_row_to_obs=chosen_row_to_obs,
                                                     return_long_probs=True)

    prob_of_chosen_alternatives, long_probs = probability_results

    # Calculate the residual vector
    residuals = choice_vector - long_probs

    # Calculate the observation specific chi-squared components
    chi_squared_terms = np.square(residuals) / long_probs
    individual_chi_squareds = rows_to_obs.T.dot(chi_squared_terms)

    # Store the log-likelihood at zero
    results["log_likelihood_null"] = log_likelihood_at_zero

    # Calculate and store the rho-squared and rho-bar-squared
    results["rho_squared"] = 1.0 - (final_log_likelihood /
                                    log_likelihood_at_zero)
    results["rho_bar_squared"] = 1.0 - ((final_log_likelihood -
                                         results.x.shape[0]) /
                                        log_likelihood_at_zero)

    # Calculate and store the final gradient
    results["final_gradient"] = general_gradient(final_utility_coefs,
                                                 design_matrix,
                                                 alt_id_vector,
                                                 rows_to_obs,
                                                 rows_to_alts,
                                                 choice_vector,
                                        easy_utility_transform,
                                    _cloglog_transform_deriv_c,
                                               easy_calc_dh_dv,
                                               calc_dh_d_alpha,
                                        final_intercept_params,
                                            final_shape_params,
                                                 ridge)
    # Calculate and store the final hessian
    results["final_hessian"] = general_hessian(final_utility_coefs,
                                               design_matrix,
                                               alt_id_vector,
                                               rows_to_obs,
                                               rows_to_alts,
                                               easy_utility_transform,
                                               _cloglog_transform_deriv_c,
                                               easy_calc_dh_dv,
                                               calc_dh_d_alpha,
                                               block_matrix_indices,
                                               final_intercept_params,
                                               final_shape_params,
                                               ridge)

    # Calculate and store the final fischer information matrix
    results["fisher_info"] = cc.calc_fisher_info_matrix(
                                                          final_utility_coefs,
                                                                design_matrix,
                                                                alt_id_vector,
                                                                   rows_to_obs,
                                                                rows_to_alts,
                                                                choice_vector,
                                                       easy_utility_transform,
                                                   _cloglog_transform_deriv_c,
                                                              easy_calc_dh_dv,
                                                              calc_dh_d_alpha,
                                                       final_intercept_params,
                                                           final_shape_params,
                                                                        ridge)

    # Add all miscellaneous objects that we need to store to the results dict
    results["final_log_likelihood"] = final_log_likelihood
    results["chosen_probs"] = prob_of_chosen_alternatives
    results["long_probs"] = long_probs
    results["residuals"] = residuals
    results["ind_chi_squareds"] = individual_chi_squareds

    return results


class MNCL(base_mcm.MNDC_Model):
    """
    Parameters
    ----------
    data : string or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV file
        containing the long format data for this choice model. Note long format
        is has one row per available alternative for each observation. If
        pandas dataframe, the dataframe should be the long format data for the
        choice model.
    alt_id_col :str.
        Should denote the column in data which contains the alternative
        identifiers for each row.
    obs_id_col : str.
        Should denote the column in data which contains the observation
        identifiers for each row.
    choice_col : str.
        Should denote the column in data which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in `data`. Values are either a
        list or a single string, "all_diff" or "all_same". If a list, the
        elements should be:
            - single objects that are in the alternative ID column of `data`
            - lists of objects that are within the alternative ID column of
              `data`. For each single object in the list, a unique column will
              be created (i.e. there will be a unique coefficient for that
              variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification` values, a single column will be created for all
              the alternatives within the iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    intercept_ref_pos : int, optional.
         Valid only when the intercepts being estimated are not part of the
         index. Specifies the alternative in the ordered array of unique
         alternative ids whose intercept or alternative-specific constant is
         not estimated, to ensure model identifiability. Default == None.
    names : OrderedDict, optional.
        Should have the same keys as `specification`. For each key:
            - if the corresponding value in `specification` is "all_same", then
              there should be a single string as the value in names.
            - if the corresponding value in `specification` is "all_diff", then
              there should be a list of strings as the value in names. There
              should be one string in the value in names for each possible
              alternative.
            - if the corresponding value in `specification` is a list, then
              there should be a list of strings as the value in names. There
              should be one string the value in names per item in the value in
              `specification`.
        Default == None.
    intercept_names : list, or None, optional.
        If a list is passed, then the list should have the same number of
        elements as there are possible alternatives in data, minus 1. Each
        element of the list should be a string--the name of the corresponding
        alternative's intercept term, in sorted order of the possible
        alternative IDs. If None is passed, the resulting names that are shown
        in the estimation results will be
        `["Outside_ASC_{}".format(x) for x in shape_names]`. Default = None.
    **kwargs : optional.
        Any other keyword arguments that are passed to the class constructor
        will be directly given to the MNDC_Model class constructor.
    """
    def __init__(self,
                 data,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 intercept_ref_pos=None,
                 names=None,
                 intercept_names=None,
                 **kwargs):
        ##########
        # Print a helpful message for users who have included shape parameters
        # or shape names unneccessarily
        ##########
        for keyword in ["shape_names", "shape_ref_pos"]:
            if keyword in kwargs and kwargs[keyword] is not None:
                warnings.warn(_shape_ignore_msg)
                break

        ##########
        # Carry out the common instantiation process for all choice models
        ##########
        super(MNCL, self).__init__(data,
                                   alt_id_col,
                                   obs_id_col,
                                   choice_col,
                                   specification,
                                   intercept_ref_pos=intercept_ref_pos,
                                   names=names,
                                   intercept_names=intercept_names,
                                   model_type="Multinomial Clog-log Model")

        # Store the utility transform function
        self.utility_transform = partial(_cloglog_utility_transform,
                                         intercept_ref_pos=intercept_ref_pos)

        return None

    def fit_mle(self,
                init_vals,
                init_intercepts=None,
                init_coefs=None,
                print_res=True,
                method="BFGS",
                loss_tol=1e-06,
                gradient_tol=1e-06,
                maxiter=1000,
                ridge=None,
                **kwargs):
        """
        Parameters
        ----------
        init_vals : 1D ndarray.
            The initial values to start the optimization process with. There
            should be one value for each index coefficient and shape
            parameter being estimated. Shape parameters should come before
            intercept parameters, which should come before index coefficients.
            One can also pass None, and instead pass `init_shapes`, optionally
            `init_intercepts` if `"intercept"` is not in the utility
            specification, and `init_coefs`.
        init_intercepts : 1D ndarray or None, optional.
            The initial values of the intercept parameters. There should be one
            parameter per possible alternative id in the dataset, minus one.
            The passed values for this argument will be ignored if `init_vals`
            is not None. This keyword argument should only be used if
            `"intercept"` is not in the utility specification. Default == None.
        init_coefs : 1D ndarray or None, optional.
            The initial values of the index coefficients. There should be one
            coefficient per index variable. The passed values for this argument
            will be ignored if `init_vals` is not None. Default == None.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
            Default `== True`.
        method : str, optional.
            Should be a valid string for scipy.optimize.minimize. Determines
            the optimization algorithm that is used for this problem.
            Default `== 'bfgs'`.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next that is needed to determine
            convergence. Default `== 1e-06`.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
            Default `== 1e-06`.
        maxiter : int, optional.
            Determines the maximum number of iterations used by the optimizer.
            Default `== 1000`.
        ridge : int, float, long, or None, optional.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. The scalar should be greater than or equal to
            zero. Default `== None`.

        Returns
        -------
        None. Estimation results are saved to the model instance.
        """
        # Check integrity of passed arguments.
        if "init_shapes" in kwargs:
            msg = "Clog-log model does not use the 'init_shapes' kwarg. "
            msg_2 = "Remove such kwargs and pass a single init_vals argument "
            msg_3 = "or init_intercepts and init_coefs."
            raise ValueError(msg + msg_2 + msg_3)

        if ridge is not None:
            warnings.warn(_ridge_warning_msg)

        # Store the optimization method
        self.optimization_method = method

        # Store the ridge parameter
        self.ridge_param = ridge

        # Construct the mappings from alternatives to observations and from
        # chosen alternatives to observations
        mapping_res = self.get_mappings_for_fit()
        rows_to_obs = mapping_res["rows_to_obs"]
        rows_to_alts = mapping_res["rows_to_alts"]
        chosen_row_to_obs = mapping_res["chosen_row_to_obs"]

        # Create init_vals from init_coefs and init_intercepts if those
        # arguments are passed to the function and init_vals is None.
        if init_vals is None and any([x is not None for x in [init_intercepts,
                                                              init_coefs]]):
            ##########
            # Check the integrity of the parameter kwargs
            ##########
            num_alternatives = rows_to_alts.shape[1]
            try:
                assert init_intercepts.shape[0] == (num_alternatives - 1)
            except AssertionError:
                msg = "init_intercepts has length {} but should have length {}"
                raise ValueError(msg.format(init_intercepts.shape,
                                            num_alternatives - 1))

            try:
                assert init_coefs.shape[0] == self.design.shape[1]
            except AssertionError:
                msg = "init_coefs has length {} but should have length {}."
                raise ValueError(msg.format(init_coefs.shape,
                                            self.design.shape[1]))

            if init_intercepts is not None:
                init_vals = np.concatenate((init_intercepts,
                                            init_coefs), axis=0)
            else:
                init_vals = init_coefs

        # Get the estimation results
        estimation_res = _estimate(init_vals,
                                   self.design,
                                   self.alt_IDs,
                                   self.choices,
                                   rows_to_obs,
                                   rows_to_alts,
                                   chosen_row_to_obs,
                                   self.intercept_ref_position,
                                   print_results=print_res,
                                   method=method,
                                   loss_tol=loss_tol,
                                   gradient_tol=gradient_tol,
                                   maxiter=maxiter,
                                   ridge=ridge,
                                   **kwargs)

        # Store the estimation results
        self.store_fit_results(estimation_res)

        return None
