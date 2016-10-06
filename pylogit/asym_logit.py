# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:16:35 2016

@name:      MultiNomial Asymmetric Logit--version 3
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating multinomial asymmetric
            logit models (with the help of the "base_multinomial_cm.py" file)
@notes:     Differs from version 1 by how it defines the transformation for
            v_n < 0. Instead of ln(1-c_j), this file uses ln((1 - c_j)/ (J-1)).
            Differs from version 2 in how it partitions the vector of
            parameters to be estimated, using
            theta = (shape | intercept | beta) instead of
            theta = (shape | beta).
"""

from functools import partial
import warnings
import numpy as np
from scipy.sparse import diags


import choice_calcs as cc
import base_multinomial_cm_v2 as base_mcm
from estimation import LogitTypeEstimator
from estimation import estimate
from display_names import model_type_to_display_name

# Define the boundary values which are not to be exceeded ducing computation
max_comp_value = 1e300
min_comp_value = 1e-300

max_exp = 700
min_exp = -700

# Alias necessary functions from the base multinomial choice model module
general_log_likelihood = cc.calc_log_likelihood
general_gradient = cc.calc_gradient
general_calc_probabilities = cc.calc_probabilities
general_hessian = cc.calc_hessian

# Create a warning string that will be issued if ridge regression is performed.
_msg = "NOTE: An L2-penalized regression is being performed. The "
_msg_2 = "reported standard errors and robust standard errors "
_msg_3 = "***WILL BE INCORRECT***."
_ridge_warning_msg = _msg + _msg_2 + _msg_3

# Create a warning that tells users to pass the shape_ref_pos keyword argument.
_msg_4 = "shape_ref_pos must be an integer. It is not an optional "
_msg_5 = "parameter for the asymmetric logit model. All shape "
_msg_6 = "parameters are not identified."
_shape_ref_msg = _msg_4 + _msg_5 + _msg_6


def split_param_vec(param_vec, rows_to_alts, design):
    """
    Parameters
    ----------
    param_vec : 1D ndarray.
        Should have as many elements as there are parameters being estimated.
    rows_to_alts : 2D scipy sparse matrix.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.

    Returns
    -------
     tuple of three 1D ndarrays.
        The first element will be an array of the shape parameters for this
        model. The second element will either be an array of the "outside"
        intercept parameters for this model or None. The third element will be
        an array of the index coefficients for this model.
    """
    # Figure out how many shape parameters we should have for the model
    num_shapes = rows_to_alts.shape[1] - 1
    # Figure out how many parameters are in the index
    num_index_coefs = design.shape[1]

    # Isolate the initial shape parameters from the betas
    shapes = param_vec[:num_shapes]
    betas = param_vec[-1 * num_index_coefs:]

    # Get the remaining outside intercepts if there are any
    remaining_idx = param_vec.shape[0] - (num_shapes + num_index_coefs)
    if remaining_idx > 0:
        intercepts = param_vec[num_shapes: num_shapes + remaining_idx]
    else:
        intercepts = None

    return shapes, intercepts, betas

def _convert_eta_to_c(eta, ref_position):
    """
    Parameters
    ----------
    eta : 1D or 2D ndarray.
        The elements of the array should be this model's 'transformed' shape
        parameters, i.e. the natural log of (the corresponding shape parameter
        divided by the reference shape parameter). This array's elements will
        be real valued. If `eta` is 2D, then its shape should be
        (num_estimated_shapes, num_parameter_samples).
    ref_position : int.
        Specifies the position in the resulting array of shape ==
        `(eta.shape[0] + 1,)` that should be equal to 1 - the sum of the other
        elements in the resulting array.

    Returns
    -------
    c_vector : 1D or 2D ndarray based on `eta`.
        If `eta` is 1D then `c_vector` should have shape
        `(eta.shape[0] + 1, )`. If `eta` is 2D then `c_vector` should have
        shape `(eta.shape[0] + 1, eta.shape[1])`. The returned array will
        contains the 'natural' shape parameters that correspond to `eta`.
    """
    # Exponentiate eta
    exp_eta = np.exp(eta)

    # Guard against overflow
    exp_eta[np.isposinf(exp_eta)] = max_comp_value
    # Guard against underflow
    exp_eta[exp_eta == 0] = min_comp_value

    # Calculate the denominator in a logistic transformation
    # Note the +1 is for the reference alternative which has been
    # constrained so that its corresponding eta = 0 and exp(0) = 1
    denom = exp_eta.sum(axis=0) + 1

    # Get a list of all the indices (or row indices) corresponding to the
    # alternatives whose shape parameters are being estimated.
    replace_list = range(eta.shape[0] + 1)
    replace_list.remove(ref_position)

    # Initialize an array for the vector of shape parameters, c
    if len(eta.shape) > 1 and eta.shape[1] > 1:
        # Get an array of zeros with shape
        # (num_possible_alternatives, num_parameter_samples). This is used when
        # working with samples from a Bayesian posterior distribution
        c_vector = np.zeros((eta.shape[0] + 1,
                             eta.shape[1]))

        # Calculate the natural shape parameters
        c_vector[replace_list, :] = exp_eta / denom
        c_vector[ref_position, :] = 1.0 / denom
    else:
        # Get an array of zeros with shape (num_possible_alternatives,)
        c_vector = np.zeros(eta.shape[0] + 1)

        # Calculate the natural shape parameters
        c_vector[replace_list] = exp_eta / denom
        c_vector[ref_position] = 1.0 / denom

    return c_vector


def _calc_deriv_c_with_respect_to_eta(natural_shapes,
                                      ref_position,
                                      output_array=None):
    """
    Parameters
    ----------
    natural_shapes : 1D ndarray.
        Should have one element per available alternative in the dataset whose
        choice situations are being modeled. Should have at least
        `ref_position` elements in it.
    ref_position : int.
        Specifies the position in the array of natural shape parameters that
        should be equal to 1 - the sum of the other elements. Specifies the
        alternative in the ordered array of unique alternatives that is not
        having its shape parameter estimated (in order to ensure
        identifiability).
    output_array : 2D ndarray.
        This array is to have its data overwritten with the correct derivatives
        of the natural shape parameters with respect to transformed shape
        parameters. Should have shape ==
        `(natural_shapes.shape[0], natural_shapes.shape[0] - 1)`.

    Returns
    -------
    output_array : 2D ndarray.
        Has shape == (natural_shapes.shape[0], natural_shapes.shape[0] - 1).
        Will contain the derivative of the shape parameters, with
        respect to the underlying 'transformed' shape parameters.
    """
    # Generate a list of the indices which indicate the columns to be
    # selected from a 2D numpy array of
    # np.diag(natural_shapes) - np.outer(natural_shapes, natural_shapes)
    columns_to_be_kept = range(natural_shapes.shape[0])
    columns_to_be_kept.remove(ref_position)

    # Calculate and store the derivative of the natural shape parameters
    # with respect to the reduced shape parameters.
    output_array[:, :] = (np.diag(natural_shapes) -
                          np.outer(natural_shapes,
                                   natural_shapes))[:, columns_to_be_kept]

    return output_array


def _asym_utility_transform(systematic_utilities,
                            alt_IDs,
                            rows_to_alts,
                            eta,
                            intercept_params,
                            shape_ref_position=None,
                            intercept_ref_pos=None,
                            *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities : 1D ndarray.
        Contains the systematic utilities for each each available alternative
        for each observation. All elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
       All elements should be ints. There should be one row per obervation per
       available alternative for the given observation. Elements denote the
       alternative corresponding to the given row of the design matrix.
    rows_to_alts : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    eta : 1D ndarray.
        Each element should be an int, float, or long. There should be one
        value per transformed shape parameter. Note that if there are J
        possible alternatives in the dataset, then there should be J - 1
        elements in `eta`.
    intercept_params : 1D ndarray or None.
        If an array, each element should be an int, float, or long. For
        identifiability, there should be J- 1 elements where J is the total
        number of observed alternatives for this dataset.
    shape_ref_position : int.
        Specifies the position in the array of natural shape parameters that
        should be equal to 1 - the sum of the other elements. Specifies the
        alternative in the ordered array of unique alternatives that is not
        having its shape parameter estimated (to ensure identifiability).
    intercept_ref_pos : int, or None, optional.
        Specifies the index of the alternative, in the ordered array of unique
        alternatives, that is not having its intercept parameter estimated (in
        order to ensure identifiability). Should only be None if
        intercept_params is None. Default == None.

    Returns
    -------
    transformed_utilities : 2D ndarray.
        Should have shape `(systematic_utilities.shape[0], 1)`. The returned
        array contains the values of the transformed index for this model.
    """
    ##########
    # Convert the reduced shape parameters to the natural shape parameters
    ##########
    natural_shape_params = _convert_eta_to_c(eta, shape_ref_position)

    ##########
    # Calculate the transformed utilities from the natural shape parameters
    ##########
    # Create a vector which contains the appropriate shape for each row in
    # the design matrix
    long_shapes = rows_to_alts.dot(natural_shape_params)

    # Determine the total number of alternatives
    num_alts = rows_to_alts.shape[1]

    # Get the natural log of the long_shapes
    log_long_shapes = np.log(long_shapes)
    # Guard against underflow, aka long_shapes too close to zero
    log_long_shapes[np.isneginf(log_long_shapes)] = -1 * max_comp_value

    # Get the natural log of (1 - long_shapes) / (J - 1)
    log_1_sub_long_shapes = np.log((1 - long_shapes) / float(num_alts - 1))
    # Guard against underflow, aka 1 - long_shapes too close to zero
    small_idx = np.isneginf(log_1_sub_long_shapes)
    log_1_sub_long_shapes[small_idx] = -1 * max_comp_value

    # Compute the transformed utilities
    multiplier = ((systematic_utilities >= 0) * log_long_shapes +
                  (systematic_utilities < 0) * log_1_sub_long_shapes)
    transformed_utilities = log_long_shapes - systematic_utilities * multiplier

    # Perform a guard for shape --> 1 and V --> infinity.
    # It is DEFINITELY not clear if this is the correct thing to do. The limit
    # might not even exist, and there is no clear multivariate L'Hopital's
    # rule. So this is an arbitrary decision
    weird_case = np.isposinf(systematic_utilities) * (long_shapes == 1)
    transformed_utilities[weird_case] = 0

    # Account for the outside intercept parameters if there are any
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
        transformed_utilities += rows_to_alts.dot(all_intercepts)

    # Perform final guards against over/underflow in the transformations
    transformed_utilities[np.isposinf(transformed_utilities)] = max_comp_value
    transformed_utilities[np.isneginf(transformed_utilities)] = -max_comp_value


    # Be sure to return a 2D array since other functions will be expecting that
    if len(transformed_utilities.shape) == 1:
        transformed_utilities = transformed_utilities[:, np.newaxis]

    return transformed_utilities


def _asym_transform_deriv_v(systematic_utilities,
                            alt_IDs,
                            rows_to_alts,
                            eta,
                            ref_position=None,
                            output_array=None,
                            *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities : 1D ndarray.
        Contains the systematic utilities for each each available alternative
        for each observation. All elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
       All elements should be ints. There should be one row per obervation per
       available alternative for the given observation. Elements denote the
       alternative corresponding to the given row of the design matrix.
    rows_to_alts : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    eta : 1D ndarray.
        Each element should be an int, float, or long. There should be one
        value per transformed shape parameter. Note that if there are J
        possible alternatives in the dataset, then there should be J - 1
        elements in `eta`.
    ref_position : int.
        Specifies the position in the array of natural shape parameters that
        should be equal to 1 - the sum of the other elements. Specifies the
        alternative in the ordered array of unique alternatives that is not
        having its shape parameter estimated (to ensure identifiability).
    output_array : 2D scipy sparse matrix.
        This matrix's data is to be replaced with the correct derivatives of
        the transformation vector with respect to the vector of systematic
        utilities.

    Returns
    -------
    output_array : 2D scipy sparse matrix.
        Will be a square matrix with `systematic_utilities.shape[0]` rows and
        columns. `output_array` specifies the derivative of the transformed
        utilities with respect to the index, V.
    """
    ##########
    # Convert the reduced shape parameters to the natural shape parameters
    ##########
    natural_shape_params = _convert_eta_to_c(eta, ref_position)

    ##########
    # Calculate the derivative of the transformed utilities with respect to
    # the systematic utilities
    ##########
    # Create a vector which contains the appropriate shape for each row in the
    # design matrix
    long_shapes = rows_to_alts.dot(natural_shape_params)

    # Determine how many alternatives there are
    num_alts = rows_to_alts.shape[1]

    # Get the natural log of the long_shapes
    log_long_shapes = np.log(long_shapes)
    # Guard against underflow, aka long_shapes too close to zero.
    # I assume this should never happen because convert_eta_to_c never outputs
    # zeros, by design.
    log_long_shapes[np.isneginf(log_long_shapes)] = -1 * max_comp_value

    # Get the natural log of (1 - long_shapes) / (num_alts - 1)
    log_1_sub_long_shapes = np.log((1 - long_shapes) /
                                   (num_alts - 1))
    # Guard against underflow, aka 1 - long_shapes too close to zero.
    small_idx = np.isneginf(log_1_sub_long_shapes)
    log_1_sub_long_shapes[small_idx] = -1 * max_comp_value

    # Calculate the derivative of h_ij with respect to v_ij
    # Note that the derivative of h_ij with respect to any other systematic
    # utility is zero.
    derivs = -1 * ((systematic_utilities >= 0).astype(int) *
                   log_long_shapes +
                   (systematic_utilities < 0).astype(int) *
                   log_1_sub_long_shapes)

    output_array.data = derivs

    # Return the matrix of dh_dv. Note the off-diagonal entries are zero
    # because each transformation only depends on its value of v and no others
    return output_array


def _asym_transform_deriv_shape(systematic_utilities,
                                alt_IDs,
                                rows_to_alts,
                                eta,
                                ref_position=None,
                                dh_dc_array=None,
                                fill_dc_d_eta=None,
                                output_array=None,
                                *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities : 1D ndarray.
        Contains the systematic utilities for each each available alternative
        for each observation. All elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
       All elements should be ints. There should be one row per obervation per
       available alternative for the given observation. Elements denote the
       alternative corresponding to the given row of the design matrix.
    rows_to_alts : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    eta : 1D ndarray.
        Each element should be an int, float, or long. There should be one
        value per transformed shape parameter. Note that if there are J
        possible alternatives in the dataset, then there should be J - 1
        elements in `eta`.
    ref_position : int.
        Specifies the position in the array of natural shape parameters that
        should be equal to 1 - the sum of the other elements. Specifies the
        alternative in the ordered array of unique alternatives that is not
        having its shape parameter estimated (to ensure identifiability).
    dh_dc_array : 2D scipy sparse matrix.
        Its data is to be replaced with the correct derivatives of the
        transformed index vector with respect to the shape parameter vector.
        Should have shape
        `(systematic_utilities.shape[0], rows_to_alts.shape[1])`.
    fill_dc_d_eta : callable.
        Should accept `eta` and `ref_position` and return a 2D numpy array
        containing the derivatives of the 'natural' shape parameter vector with
        respect to the vector of transformed shape parameters.
    output_array : 2D numpy matrix.
        This matrix's data is to be replaced with the correct derivatives of
        the transformed systematic utilities with respect to the vector of
        transformed shape parameters. Should have shape 
        `(systematic_utilities.shape[0], shape_params.shape[0])`.

    Returns
    -------
    output_array : 2D ndarray.
        The shape of the returned array will be
        `(systematic_utilities.shape[0], shape_params.shape[0])`. The returned
        array specifies the derivative of the transformed utilities with
        respect to the shape parameters.
    """
    ##########
    # Convert the reduced shape parameters to the natural shape parameters
    ##########
    natural_shape_params = _convert_eta_to_c(eta, ref_position)

    ##########
    # Calculate the derivative of the transformed utilities with respect to
    # the vector of natural shape parameters, c
    ##########
    # Create a vector which contains the appropriate shape for each row in the
    # design matrix. Note as long as natural_shape_params is a numpy array,
    # then long_shapes will be a numpy array.
    long_shapes = rows_to_alts.dot(natural_shape_params)

    # Calculate d_ln(long_shape)_d_long_shape
    d_lnShape_dShape = 1.0 / long_shapes
    # Guard against overflow
    d_lnShape_dShape[np.isposinf(d_lnShape_dShape)] = max_comp_value

    # Calculate d_ln((1-long_shape)/(J-1))_d_long_shape
    d_lnShapeComp_dShape = -1.0 / (1 - long_shapes)
    # Guard against overflow
    d_lnShapeComp_dShape[np.isneginf(d_lnShapeComp_dShape)] = -max_comp_value

    # Differentiate the multiplier with respect to natural_shape_j.
    deriv_multiplier = ((systematic_utilities >= 0) * d_lnShape_dShape +
                        (systematic_utilities < 0) * d_lnShapeComp_dShape)
    # assert not np.isnan(deriv_multiplier).any()

    # Calculate the derivative of h_ij with respect to natural_shape_j.
    # Store these derivatives in their respective places in the dh_dc array
    # Note that d_hij_d_ck = 0 for k != j
    dh_dc_values = d_lnShape_dShape - systematic_utilities * deriv_multiplier
    # Guard against overflow
    dh_dc_values[np.isinf(dh_dc_values)] = -1 * max_comp_value
    # Assign the computed values to the scipy sparse array
    dh_dc_array.data = dh_dc_values

    ##########
    # Calculate the derivative of the natural shape parameters, c with
    # respect to the vector of reduced shape parameters, eta
    ##########
    # Return the matrix of dh_d_eta. Note the matrix should be of dimension
    # (systematic_utilities.shape[0], shape_params.shape[0])
    # Note the calculation is essentially dh_dc * dc_d_eta = dh_d_eta
    output_array[:, :] = dh_dc_array.dot(fill_dc_d_eta(natural_shape_params,
                                                       ref_position))
    return output_array


def _asym_transform_deriv_alpha(systematic_utilities,
                                alt_IDs,
                                rows_to_alts,
                                intercept_params,
                                output_array=None,
                                *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities : 1D ndarray.
        Contains the systematic utilities for each each available alternative
        for each observation. All elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
       All elements should be ints. There should be one row per obervation per
       available alternative for the given observation. Elements denote the
       alternative corresponding to the given row of the design matrix.
    rows_to_alts : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    intercept_params : 1D ndarray or None.
        If an array, each element should be an int, float, or long. For
        identifiability, there should be J- 1 elements where J is the total
        number of observed alternatives for this dataset.
    output_array : None or 2D scipy sparse matrix.
        If `output_array` is a 2D scipy sparse matrix, then it should contain
        the derivative of the vector of transformed utilities with respect to
        the intercept parameters outside of the index. This keyword argurment
        will be returned without alteration.

        If there are no intercept parameters outside of the index, then
        `output_array` should equal None.

        If there are intercept parameters outside of the index, then
        `output_array` should be rows_to_alts` without the column corresponding
        to the alternative whose intercept is not being estimated in order to
        ensure identifiability.

    Returns
    -------
    output_array.
    """
    return output_array


def create_calc_dh_dv(estimator):
    """
    Return the function that can be used in the various gradient and hessian
    calculations to calculate the derivative of the transformation with respect
    to the index.

    Parameters
    ----------
    estimator : an instance of the estimation.LogitTypeEstimator class.
        Should contain a `design` attribute that is a 2D ndarray representing
        the design matrix for this model and dataset.

    Returns
    -------
    Callable.
        Will accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of systematic utilities. The dimensions of the returned vector should
        be `(design.shape[0], design.shape[0])`.
    """
    dh_dv = diags(np.ones(estimator.design.shape[0]), 0, format='csr')
    # Create a function that will take in the pre-formed matrix, replace its
    # data in-place with the new data, and return the correct dh_dv on each
    # iteration of the minimizer
    calc_dh_dv = partial(_asym_transform_deriv_v,
                         ref_position=estimator.shape_ref_pos,
                         output_array=dh_dv)
    return calc_dh_dv


def create_calc_dh_d_shape(estimator):
    """
    Return the function that can be used in the various gradient and hessian
    calculations to calculate the derivative of the transformation with respect
    to the shape parameters.

    Parameters
    ----------
    estimator : an instance of the estimation.LogitTypeEstimator class.
        Should contain a `rows_to_alts` attribute that is a 2D scipy sparse
        matrix that maps the rows of the `design` matrix to the alternatives
        available in this dataset.

    Returns
    -------
    Callable.
        Will accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of shape parameters. The dimensions of the returned vector should
        be `(design.shape[0], num_alternatives)`.
    """
    num_alts = estimator.rows_to_alts.shape[1]
    pre_dc_d_eta = np.zeros((num_alts, num_alts - 1), dtype=float)

    pre_dh_dc = estimator.rows_to_alts.copy()

    pre_dh_d_eta = np.matrix(np.zeros((estimator.design.shape[0],
                                       num_alts - 1), dtype=float))

    easy_calc_dc_d_eta = partial(_calc_deriv_c_with_respect_to_eta,
                                 output_array=pre_dc_d_eta)

    # Create a function that will take in the pre-formed matrix, replace its
    # data in-place with the new data, and return the correct dh_dshape on each
    # iteration of the minimizer
    calc_dh_d_eta = partial(_asym_transform_deriv_shape,
                            ref_position=estimator.shape_ref_pos,
                            dh_dc_array=pre_dh_dc,
                            fill_dc_d_eta=easy_calc_dc_d_eta,
                            output_array=pre_dh_d_eta)
    return calc_dh_d_eta


def create_calc_dh_d_alpha(estimator):
    """
    Return the function that can be used in the various gradient and hessian
    calculations to calculate the derivative of the transformation with respect
    to the outside intercept parameters.

    Parameters
    ----------
    estimator : an instance of the estimation.LogitTypeEstimator class.
        Should contain a `rows_to_alts` attribute that is a 2D scipy sparse
        matrix that maps the rows of the `design` matrix to the alternatives
        available in this dataset. Should also contain an `intercept_ref_pos`
        attribute that is either None or an int. This attribute should denote
        which intercept is not being estimated (in the case of outside
        intercept parameters) for identification purposes.

    Returns
    -------
    Callable.
        Will accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of outside intercepts. The dimensions of the returned vector should
        be `(design.shape[0], num_alternatives - 1)`.
    """
    if estimator.intercept_ref_pos is not None:
        needed_idxs = range(estimator.rows_to_alts.shape[1])
        needed_idxs.remove(estimator.intercept_ref_pos)
        dh_d_alpha = (estimator.rows_to_alts
                               .copy()
                               .transpose()[needed_idxs, :]
                               .transpose())
    else:
        dh_d_alpha = None
    # Create a function that will take in the pre-formed matrix, replace its
    # data in-place with the new data, and return the correct dh_dalpha on each
    # iteration of the minimizer
    calc_dh_d_alpha = partial(_asym_transform_deriv_alpha,
                              output_array=dh_d_alpha)

    return calc_dh_d_alpha


class AsymEstimator(LogitTypeEstimator):
    """
    Estimation Object used to enforce uniformity in the estimation process
    across the various logit-type models.

    Parameters
    ----------
    model_obj : a pylogit.base_multinomial_cm_v2.MNDC_Model instance.
        Should contain the following attributes:

          - alt_IDs
          - choices
          - design
          - intercept_ref_position
          - shape_ref_position
          - utility_transform
    mapping_res : dict.
        Should contain the scipy sparse matrices that map the rows of the long
        format dataframe to various other objects such as the available
        alternatives, the unique observations, etc. The keys that it must have
        are `['rows_to_obs', 'rows_to_alts', 'chosen_row_to_obs']`
    ridge : int, float, long, or None.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. The scalar should be greater than or equal to
            zero..
    zero_vector : 1D ndarray.
        Determines what is viewed as a "null" set of parameters. It is
        explicitly passed because some parameters (e.g. parameters that must be
        greater than zero) have their null values at values other than zero.
    split_params : callable.
        Should take a vector of parameters, `mapping_res['rows_to_alts']`, and
        model_obj.design as arguments. Should return a tuple containing
        separate arrays for the model's shape, outside intercept, and index
        coefficients. For each of these arrays, if this model does not contain
        the particular type of parameter, the callable should place a `None` in
        its place in the tuple.
    """
    def set_derivatives(self):
        self.calc_dh_dv = create_calc_dh_dv(self)
        self.calc_dh_d_alpha = create_calc_dh_d_alpha(self)
        self.calc_dh_d_shape = create_calc_dh_d_shape(self)

    def check_length_of_initial_values(self, init_values):
        """
        Ensures that `init_values` is of the correct length. Raises a helpful
        ValueError if otherwise.

        Parameters
        ----------
        init_values : 1D ndarray.
            The initial values to start the optimization process with. There
            should be one value for each index coefficient, outside intercept
            parameter, and shape parameter being estimated.

        Returns
        -------
        None.
        """
        # Calculate the expected number of shape and index parameters
        # Note the asymmetric logit model has one shape parameter less than
        # the number of alternatives. The same is true of the number of outside
        # intercepts.
        num_alts = self.rows_to_alts.shape[1]
        num_index_coefs = self.design.shape[1]

        if self.intercept_ref_pos is not None:
            assumed_param_dimensions = num_index_coefs + 2 * (num_alts - 1)
        else:
            assumed_param_dimensions = num_index_coefs + num_alts - 1

        if init_values.shape[0] != assumed_param_dimensions:
            msg_1 = "The initial values are of the wrong dimension."
            msg_2 = "It should be of dimension {}"
            msg_3 = "But instead it has dimension {}"
            raise ValueError(msg_1 +
                             msg_2.format(assumed_param_dimensions) +
                             msg_3.format(init_values.shape[0]))

        return None


class MNAL(base_mcm.MNDC_Model):
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
    shape_ref_pos : int, optional.
        Specifies the alternative in the ordered array of unique alternative
        ids whose shape parameter is not estimated, to ensure model
        identifiability. Implemented as an optional parameter but MUST be
        passed for this model.
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
    shape_names : list, or None, optional.
        If a list is passed, then the list should have the same number of
        elements as there are possible alternative IDs in data. Each element of
        the list should be a string denoting the name of the corresponding
        shape parameter for the given alternative, in sorted order of the
        possible alternative IDs. The resulting names which are shown in the
        estimation results will be ["shape_{}".format(x) for x in shape_names].
        Default == None.
    """
    def __init__(self,
                 data,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 intercept_ref_pos=None,
                 shape_ref_pos=None,
                 names=None,
                 intercept_names=None,
                 shape_names=None,
                 **kwargs):

        ##########
        # Check that shape_ref_pos has been passed.
        ##########
        try:
            assert isinstance(shape_ref_pos, int)
        except AssertionError:
            raise ValueError(_shape_ref_msg)

        # Carry out the common instantiation process for all choice models
        super(MNAL, self).__init__(data,
                                   alt_id_col,
                                   obs_id_col,
                                   choice_col,
                                   specification,
                                   intercept_ref_pos=intercept_ref_pos,
                                   shape_ref_pos=shape_ref_pos,
                                   names=names,
                                   intercept_names=intercept_names,
                                   shape_names=shape_names,
                                   model_type=model_type_to_display_name["Asym"])

        # Store the utility transform function
        self.utility_transform = partial(_asym_utility_transform,
                                         shape_ref_position=shape_ref_pos,
                                         intercept_ref_pos=intercept_ref_pos)

        return None

    def fit_mle(self, init_vals,
                init_shapes=None,
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
        init_shapes : 1D ndarray or None, optional.
            The initial values of the shape parameters. All elements should be
            ints, floats, or longs. There should be one element less than the
            total number of possible alternatives in the dataset. This keyword
            argument will be ignored if `init_vals` is not None.
            Default == None.
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
        # Store the optimization method
        self.optimization_method = method

        # Store the ridge parameter
        self.ridge_param = ridge

        if ridge is not None:
            warnings.warn(_ridge_warning_msg)

        # Construct the mappings from alternatives to observations and from
        # chosen alternatives to observations
        mapping_res = self.get_mappings_for_fit()
        rows_to_alts = mapping_res["rows_to_alts"]

        # Create init_vals from init_coefs, init_intercepts, and init_shapes if
        # those arguments are passed to the function and init_vals is None.
        if init_vals is None and all([x is not None for x in [init_shapes,
                                                              init_coefs]]):
            ##########
            # Check the integrity of the parameter kwargs
            ##########
            num_alternatives = rows_to_alts.shape[1]
            try:
                assert init_shapes.shape[0] == num_alternatives - 1
            except AssertionError:
                msg = "init_shapes is of length {} but should be of length {}"
                raise ValueError(msg.format(init_shapes.shape,
                                            num_alternatives - 1))

            try:
                assert init_coefs.shape[0] == self.design.shape[1]
            except AssertionError:
                msg = "init_coefs has length {} but should have length {}"
                raise ValueError(msg.format(init_coefs.shape,
                                            self.design.shape[1]))

            try:
                if init_intercepts is not None:
                    assert init_intercepts.shape[0] == (num_alternatives - 1)
            except AssertionError:
                msg = "init_intercepts has length {} but should have length {}"
                raise ValueError(msg.format(init_intercepts.shape,
                                            num_alternatives - 1))

            # The code block below will limit users to only having 'inside'
            # OR 'outside' intercept parameters but not both.
            # try:
            #     condition_1 = "intercept" not in self.specification
            #     condition_2 = init_intercepts is None
            #     assert condition_1 or condition_2
            # except AssertionError as e:
            #     msg = "init_intercepts should only be used if 'intercept' is"
            #     msg_2 = " not in one's index specification."
            #     msg_3 = "Either make init_intercepts = None or remove "
            #     msg_4 = "'intercept' from the specification."
            #     print(msg + msg_2)
            #     print(msg_3 + msg_4)
            #     raise e

            if init_intercepts is not None:
                init_vals = np.concatenate((init_shapes,
                                            init_intercepts,
                                            init_coefs), axis=0)
            else:
                init_vals = np.concatenate((init_shapes,
                                            init_coefs), axis=0)

        elif init_vals is None:
            msg = "If init_vals is None, then users must pass both init_coefs "
            msg_2 = "and init_shapes."
            raise ValueError(msg + msg_2)

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
        asym_estimator = AsymEstimator(self,
                                       mapping_res,
                                       ridge,
                                       zero_vector,
                                       split_param_vec)
        # Set the derivative functions for estimation
        asym_estimator.set_derivatives()

        # Perform one final check on the length of the initial values
        asym_estimator.check_length_of_initial_values(init_vals)

        # Get the estimation results
        estimation_res = estimate(init_vals,
                                  asym_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res)

        # Store the estimation results
        self.store_fit_results(estimation_res)

        return None
