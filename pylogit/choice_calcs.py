# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:12:36 2016

@module:    choice_calcs.py
@name:      Choice Calculations
@author:    Timothy Brathwaite
@summary:   Contains generic functions necessary for calculating choice
            probabilities and for estimating the choice models.
"""
from __future__ import absolute_import

import numpy as np
import scipy.stats
import scipy.linalg
from scipy.linalg import block_diag
from scipy.sparse import hstack, issparse

try:
    # Python 3.x does not natively support xrange
    from past.builtins import xrange
except ImportError:
    pass

# Define the boundary values which are not to be exceeded during computation
min_exponent_val = -700
max_exponent_val = 700

max_comp_value = 1e300
min_comp_value = 1e-300
# The value below was determined since its the smallest value, x, for which
# 1 - x != 1
# min_comp_value = 1e-16


def calc_probabilities(beta,
                       design,
                       alt_IDs,
                       rows_to_obs,
                       rows_to_alts,
                       utility_transform,
                       intercept_params=None,
                       shape_params=None,
                       chosen_row_to_obs=None,
                       return_long_probs=False):
    """
    Parameters
    ----------
    beta : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each utility coefficient being estimated (i.e.
        num_features). If 2D, should have 1 column for each set of coefficients
        being used to predict the probabilities of each alternative being
        chosen. There should be one row per index coefficient.
    design : 2D or 3D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs. If `len(design.shape) == 3`,
        then beta MUST be 1D.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_alts : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    utility_transform : callable.
        Should accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 1D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated.
    intercept_params : 1D ndarray, or None, optional.
        If an array, each element should be an int, float, or long. For
        identifiability, there should be J- 1 elements where J is the total
        number of observed alternatives for this dataset. Default == None.
    shape_params : 1D ndarray, or None, optional.
        If an array, each element should be an int, float, or long. There
        should be one value per shape parameter of the model being used.
        Default == None.
    chosen_row_to_obs :  2D scipy sparse array, or None, optional.
        There should be one row per observation per available alternative and
        one column per observation. This matrix indicates, for each observation
        (on the columns), which rows of the design matrix were the realized
        outcome. If an array is passed then an array of shape
        (num_observations,) will be returned and each element will be the
        probability of the realized outcome of the given observation.
        Default == None.
    return_long_probs :  bool, optional.
        Indicates whether or not the long format probabilites (a 1D numpy array
        with one element per observation per available alternative) should be
        returned. Default == False.

    Returns
    -------
    numpy array or tuple of two numpy arrays.
        If `chosen_row_to_obs` is passed AND `return_long_probs is True`, then
        the tuple `(chosen_probs, long_probs)` is returned. If
        `return_long_probs is True` and `chosen_row_to_obs is None`, then
        `long_probs` is returned. If `chosen_row_to_obs` is passed and
        `return_long_probs is False` then `chosen_probs` is returned.

        `chosen_probs` is a 1D numpy array of shape (num_observations,). Each
        element is the probability of the corresponding observation being
        associated with its realized outcome.

        `long_probs` is a 1D numpy array with one element per observation per
        available alternative for that observation. Each element is the
        probability of the corresponding observation being associated with that
        rows corresponding alternative.

        If `beta` is a 2D array, `chosen_probs` and `long_probs` will also be
        2D arrays, with as many columns as there are sets of parameters being
        used to calculate probabilities with.

        It is NOT valid to have `chosen_row_to_obs == None` and
        `return_long_probs == False`.
    """
    # Check argument validity
    if (len(beta.shape) >= 2) and (len(design.shape) >= 3):
        msg_1 = "Cannot calculate probabilities with both 3D design matrix AND"
        msg_2 = " 2D coefficient array."
        raise ValueError(msg_1 + msg_2)
    if chosen_row_to_obs is None and return_long_probs is False:
        msg = "chosen_row_to_obs is None AND return_long_probs is False"
        raise ValueError(msg)

    # Calculate the systematic utility for each alternative for each individual
    sys_utilities = design.dot(beta)

    # Calculate the probability from the transformed utilities
    # The transformed utilities will be of shape (num_rows, 1)
    transformed_utilities = utility_transform(sys_utilities,
                                              alt_IDs,
                                              rows_to_alts,
                                              shape_params,
                                              intercept_params)

    # The following commands are to guard against numeric under/over-flow
    too_small_idx = transformed_utilities < min_exponent_val
    too_large_idx = transformed_utilities > max_exponent_val

    transformed_utilities[too_small_idx] = min_exponent_val
    transformed_utilities[too_large_idx] = max_exponent_val

    # Exponentiate the transformed utilities
    long_exponentials = np.exp(transformed_utilities)

    # long_probs will be of shape (num_rows,) Each element will provide the
    # probability of the observation associated with that row having the
    # alternative associated with that row as the observation's outcome
    individual_denominators = np.asarray(rows_to_obs.transpose().dot(
                                                    long_exponentials))
    long_denominators = np.asarray(rows_to_obs.dot(individual_denominators))
    if len(long_exponentials.shape) > 1 and long_exponentials.shape[1] > 1:
        long_probs = (long_exponentials / long_denominators)
    else:
        long_probs = (long_exponentials / long_denominators).ravel()

    # Guard against underflow
    long_probs[long_probs == 0] = min_comp_value

    if chosen_row_to_obs is None:
        chosen_probs = None
    else:
        # chosen_probs will be of shape (num_observations,)
        chosen_exponentials = np.asarray(
                         chosen_row_to_obs.transpose().dot(long_exponentials))
        if len(long_exponentials.shape) > 1 and long_exponentials.shape[1] > 1:
            chosen_probs = chosen_exponentials / individual_denominators
        else:
            chosen_probs = (chosen_exponentials /
                            individual_denominators).ravel()

    # Return the long form and chosen probabilities if desired
    if return_long_probs and chosen_probs is not None:
        return chosen_probs, long_probs
    # If working with predictions, return just the long form probabilities
    elif return_long_probs and chosen_probs is None:
        return long_probs
    # If estimating the model and storing fitted probabilities or testing the
    # model on data for which we know the chosen alternative, just return the
    # chosen probabilities.
    elif chosen_probs is not None:
        return chosen_probs


def calc_log_likelihood(beta,
                        design,
                        alt_IDs,
                        rows_to_obs,
                        rows_to_alts,
                        choice_vector,
                        utility_transform,
                        intercept_params=None,
                        shape_params=None,
                        ridge=None,
                        weights=None):
    """
    Parameters
    ----------
    beta : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features).
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_alts : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform:  callable.
        Should accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 1D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated.
    intercept_params:   1D ndarray, or None, optional.
        If an array, each element should be an int, float, or long. For
        identifiability, there should be J- 1 elements where J is the total
        number of observed alternatives for this dataset. Default == None.
    shape_params : 1D ndarray, or None, optional.
        If an array, each element should be an int, float, or long. There
        should be one value per shape parameter of the model being used.
        Default == None.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization. Default = None.
    weights : 1D ndarray or None, optional.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.

    Returns
    -------
    log_likelihood : float. The log likelihood of the multinomial choice model.
    """
    # Calculate the probability of each individual choosing each available
    # alternative for that individual.
    long_probs = calc_probabilities(beta,
                                    design,
                                    alt_IDs,
                                    rows_to_obs,
                                    rows_to_alts,
                                    utility_transform,
                                    intercept_params=intercept_params,
                                    shape_params=shape_params,
                                    return_long_probs=True)

    # Calculate the weights for the sample
    if weights is None:
        weights = 1

    # Calculate the log likelihood
    log_likelihood = choice_vector.dot(weights * np.log(long_probs))

    if ridge is None:
        return log_likelihood
    else:
        param_list = [x for x in [shape_params, intercept_params, beta]
                      if x is not None]
        if len(param_list) > 1:
            params = np.concatenate(param_list, axis=0)
        else:
            params = param_list[0]
        return log_likelihood - ridge * np.square(params).sum()


def calc_gradient(beta,
                  design,
                  alt_IDs,
                  rows_to_obs,
                  rows_to_alts,
                  choice_vector,
                  utility_transform,
                  transform_first_deriv_c,
                  transform_first_deriv_v,
                  transform_deriv_alpha,
                  intercept_params,
                  shape_params,
                  ridge,
                  weights):
    """
    Parameters
    ----------
    beta : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features).
    design : 2D ndarray.
        Tjere should be one row per observation per available alternative.
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
    rows_to_alts : 2D scipy sparse array
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 1D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated.
    transform_first_deriv_c : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, the `rows_to_alts` array, (shape parameters if there
        are any) and miscellaneous args and kwargs. Should return a 2D matrix
        or sparse array whose elements contain the derivative of the tranformed
        utility vector with respect to the vector of shape parameters. The
        dimensions of the returned vector should be
        `(design.shape[0], num_alternatives)`. If there are no shape parameters
        then the callable should return None.
    transform_first_deriv_v : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of systematic utilities. The dimensions of the returned vector should
        be `(design.shape[0], design.shape[0])`.
    transform_deriv_alpha : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, the `rows_to_alts` array, (intercept parameters if
        there are any) and miscellaneous args and kwargs. Should return a 2D
        array whose elements contain the derivative of the tranformed utility
        vector with respect to the vector of shape parameters. The dimensions
        of the returned vector should be
        `(design.shape[0], num_alternatives - 1)`. If there are no intercept
        parameters, the callable should return None.
    intercept_params : 1D numpy array or None.
        If an array, each element should be an int, float, or long. For
        identifiability, there should be J- 1 elements where J is the total
        number of observed alternatives for this dataset. Default == None.
    shape_params : 1D ndarray or None.
       If an array, each element should be an int, float, or long. There should
       be one value per shape parameter of the model being used.
       Default == None.
    ridge : int, float, long, or None.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization. Default = None.
    weights : 1D ndarray or None.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.

    Returns
    -------
    gradient : 1D ndarray.
       It's shape is (beta.shape[0], ). It is the second derivative of the log-
       likelihood with respect to beta.
    """
    # Calculate the systematic utility for each alternative for each individual
    sys_utilities = design.dot(beta)

    # Calculate the probability of each individual choosing each available
    # alternative for that individual.
    long_probs = calc_probabilities(beta,
                                    design,
                                    alt_IDs,
                                    rows_to_obs,
                                    rows_to_alts,
                                    utility_transform,
                                    intercept_params=intercept_params,
                                    shape_params=shape_params,
                                    return_long_probs=True)

    # Calculate the weights for the sample
    if weights is None:
        weights = 1

    ##########
    # Get the required matrices
    ##########
    # Differentiate the transformed utilities with respect to the shape params
    # Note that dh_dc should be a sparse array
    dh_dc = transform_first_deriv_c(sys_utilities, alt_IDs,
                                    rows_to_alts, shape_params)
    # Differentiate the transformed utilities by the intercept params
    # Note that dh_d_alpha should be a sparse array
    dh_d_alpha = transform_deriv_alpha(sys_utilities, alt_IDs,
                                       rows_to_alts, intercept_params)
    # Differentiate the transformed utilities with respect to the systematic
    # utilities. Note that dh_dv should be a sparse matrix
    dh_dv = transform_first_deriv_v(sys_utilities, alt_IDs,
                                    rows_to_alts, shape_params)
    # Differentiate the transformed utilities with respect to the utility
    # coefficients. Note that dh_db should be a dense **matrix**, not a dense
    # 2D array. This is because the dot product of a 2D scipy sparse array and
    # a 2D dense numpy array yields a 2D dense numpy matrix
    dh_db = dh_dv.dot(design)
    # Differentiate the log likelihood w/ respect to the transformed utilities
    # Note that d_ll_dh will be a dense 2D numpy array.
    d_ll_dh = np.multiply(weights, choice_vector - long_probs)[np.newaxis, :]

    # Calculate the gradient of the log-likelihood with respect to the betas
    d_ll_d_beta = d_ll_dh.dot(dh_db)

    ##########
    # Form and return the gradient
    ##########
    if shape_params is not None and intercept_params is not None:
        # Note that we use d_ll_dh * dh_dc and d_ll_dh * dh_d_alpha because
        # that is how one computes the dot product between a dense 2D numpy
        # array and a 2D sparse matrix. This is due to numpy ndarrays and
        # scipy sparse matrices not playing nicely together. However, numpy
        # ndarrays and numpy matrices can be dot producted together,
        # hence d_ll_dh.dot(dh_db).

        # Note that the 'np.asarray' is because dll_dh * dh_dc will be a row
        # matrix, but we want a 1D numpy array.
        gradient = np.concatenate((np.asarray(d_ll_dh * hstack((dh_dc,
                                                                dh_d_alpha),
                                                               format='csr')),
                                   d_ll_d_beta), axis=1).ravel()
        params = np.concatenate((shape_params, intercept_params, beta),
                                axis=0)

    elif shape_params is not None and intercept_params is None:
        # Note that we use d_ll_dh * dh_dc because that is how one computes
        # the dot product between a dense 2D numpy array and a 2D sparse matrix
        # This is due to numpy ndarrays and scipy sparse matrices not playing
        # nicely together. However, numpy ndarrays and numpy matrices can be
        # dot producted together, hence d_ll_dh.dot(dh_db).

        # Note that the 'np.asarray' is because dll_dh * dh_dc will be a row
        # matrix, but we want a 1D numpy array.
        gradient = np.concatenate((np.asarray(d_ll_dh * dh_dc), d_ll_d_beta),
                                  axis=1).ravel()
        params = np.concatenate((shape_params, beta), axis=0)

    elif shape_params is None and intercept_params is not None:
        # Note that we use d_ll_dh * dh_d_alpha because that's how one computes
        # the dot product between a dense 2D numpy array and a 2D sparse matrix
        # This is due to numpy ndarrays and scipy sparse matrices not playing
        # nicely together. However, numpy ndarrays and numpy matrices can be
        # dot producted together, hence d_ll_dh.dot(dh_db).

        # Note 'np.asarray' is used because dll_dh * dh_d_alpha will be a row
        # matrix, but we want a 1D numpy array.
        gradient = np.concatenate((np.asarray(d_ll_dh * dh_d_alpha),
                                   d_ll_d_beta), axis=1).ravel()
        params = np.concatenate((intercept_params, beta), axis=0)

    else:
        gradient = d_ll_d_beta.ravel()
        params = beta

    if ridge is not None:
        gradient -= 2 * ridge * params

    return gradient


def quadratic_prod_wrt_dp_ds(left,
                             right,
                             probs,
                             rows_to_obs,
                             weights=None):
    """
    Calculates `left * diag(weights) * dp_ds * right` in a memory efficient
    way, avoiding explicit computation of `dp_ds`. This allows in memory
    computation of the hessian.

    Parameters
    ----------
    left, right : 2D ndarray or 2D sparse matrix of scalars.
        Respectively, these arrays have the same number of columns and rows as
        there are rows in the long format data (or in `probs`.) They are to
        left- and right-multiply the `dp_ds` matrix.
    probs : 1D ndarray of floats in (0, 1).
        Represents the array of predicted probabilities for each available
        alternative in each choice situation.
    rows_to_obs : 2D compressed-sparse row matrix of zeros and ones.
        Maps each row to its corresponding observation.
    weights : 1D ndarray or None.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.

    Returns
    -------
    product : 2D ndarray.
        Will be the matrix produced by `left * dp_ds * right`.
    """
    # Calculate the weights for the sample
    if weights is None:
        weights = np.ones(probs.shape[0])
    # Convert matrixlib objects to ndarrays
    left = left.A if isinstance(left, np.matrixlib.defmatrix.matrix) else left
    right =\
        right.A if isinstance(right, np.matrixlib.defmatrix.matrix) else right
    # Determine properties of left and right
    left_is_ndarray = isinstance(left, np.ndarray)
    left_is_sparse, right_is_sparse = issparse(left), issparse(right)

    # Compute the needed matrices and arrays
    wt_probs = weights * probs
    # Note that probs is the same length as the COLUMNS of `left`, hence the
    # [None, :] slice instead of [:, None].
    left_times_wt_probs = (left.multiply(wt_probs[None, :]).tocsr()
                           if left_is_sparse else np.multiply(left, wt_probs))
    # Note that probs is the same length as the ROWS of `rows_to_obs`, hence
    # the [:, None] slice instead of [None, :].
    broadcasted_probs = rows_to_obs.multiply(probs[:, None]).tocsr()
    broadcasted_wt_probs = rows_to_obs.multiply(wt_probs[:, None]).tocsr()

    # Compute the terms of the desired product
    # Note that (AB)^T = (B^T)(A^T)
    term_1 = ((right.T.dot(left_times_wt_probs.T)).T
              if right_is_sparse else left_times_wt_probs.dot(right))
    term_2 = ((broadcasted_wt_probs.T.dot(left.T)).T
              if left_is_ndarray else left.dot(broadcasted_wt_probs))
    term_3 = broadcasted_probs.T.dot(right)
    term_4 = ((term_3.T.dot(term_2.T)).T if issparse(term_3)
              else term_2.dot(term_3))

    # Make sure the final terms are ndarrays so the result is an ndarray
    term_1 = term_1.toarray() if issparse(term_1) else term_1
    term_4 = term_4.toarray() if issparse(term_4) else term_4
    return np.asarray(term_1 - term_4)


def calc_hessian(beta,
                 design,
                 alt_IDs,
                 rows_to_obs,
                 rows_to_alts,
                 utility_transform,
                 transform_first_deriv_c,
                 transform_first_deriv_v,
                 transform_deriv_alpha,
                 intercept_params,
                 shape_params,
                 ridge,
                 weights):
    """
    Parameters
    ----------
    beta : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features).
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_alts: 2D ndarray.
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    utility_transform : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 1D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated.
    transform_first_deriv_c : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, the `rows_to_alts` array, (shape parameters if there
        are any) and miscellaneous args and kwargs. Should return a 2D array
        whose elements contain the derivative of the tranformed utilities with
        respect to the vector of shape parameters. The dimensions of the
        returned vector should be `(design.shape[0], num_alternatives)`.
    transform_first_deriv_v : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, (shape parameters if there are any) and miscellaneous
        args and kwargs. Should return a 2D array whose elements contain the
        derivative of the tranformed utility vector with respect to the vector
        of systematic utilities. The dimensions of the returned vector should
        be `(design.shape[0], design.shape[0])`.
    transform_deriv_alpha : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, the rows_to_alts array, (intercept parameters if
        there are any) and miscellaneous args and kwargs. Should return a 2D
        array whose elements contain the derivative of the tranformed utilities
        with respect to the vector of shape parameters. The dimensions of the
        returned vector should be `(design.shape[0], num_alternatives - 1)`. If
        `intercept_params == None`, the callable should return None.
    intercept_params : 1D ndarray.
        Each element should be an int, float, or long. For identifiability,
        there should be J- 1 elements where J is the total number of observed
        alternatives in the dataset.
    shape_params: None or 1D ndarray.
        If an array, each element should be an int, float, or long. There
        should be one value per shape parameter of the model being used.
        Default == None.
    ridge : int, float, long, or None.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization. Default = None.
    weights : 1D ndarray or None.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.

    Returns
    -------
    hess : 2D ndarray.
        It's shape is `(beta.shape[0], beta.shape[0])`. It is the second
        derivative of the log likelihood with respect to beta.
    """
    # Calculate the systematic utility for each alternative for each individual
    sys_utilities = design.dot(beta)

    # Calculate the probability of each individual choosing each available
    # alternative for that individual.
    long_probs = calc_probabilities(beta,
                                    design,
                                    alt_IDs,
                                    rows_to_obs,
                                    rows_to_alts,
                                    utility_transform,
                                    intercept_params=intercept_params,
                                    shape_params=shape_params,
                                    return_long_probs=True)

    # Calculate the weights for the sample
    if weights is None:
        weights = np.ones(design.shape[0])

    ##########
    # Get the required matrices
    ##########
    # Differentiate the transformed utilities with respect to the shape params
    # Note that dh_dc will be a 2D scipy sparse matrix
    dh_dc = transform_first_deriv_c(sys_utilities, alt_IDs,
                                    rows_to_alts, shape_params)
    # Differentiate the transformed utilities with respect to the systematic
    # utilities. Note that dh_dv will be a 2D scipy sparse matrix.
    dh_dv = transform_first_deriv_v(sys_utilities, alt_IDs,
                                    rows_to_alts, shape_params)
    # Differentiate the transformed utilities by the intercept params
    # Note that dh_d_alpha should be a sparse array
    dh_d_alpha = transform_deriv_alpha(sys_utilities, alt_IDs,
                                       rows_to_alts, intercept_params)
    # Differentiate the transformed utilities with respect to the utility
    # coefficients. Note that dh_db will be a 2D dense numpy matrix
    dh_db = dh_dv.dot(design)

    ##########
    # Calculate the second and mixed partial derivatives within the hessian
    ##########
    # Calculate the second derivative of the log-likelihood with repect to the
    # utility coefficients. Should have shape (design.shape[0],
    # design.shape[0]). Since dp_dh is a 2D dense numpy array and dh_db is a
    # 2D dense numpy matrix, using the .dot() syntax should work to compute
    # the dot product.
    # `d2_ll_db2 = -1 * dh_db.T.dot(dp_dh.dot(dh_db))`
    d2_ll_db2 = -1 * quadratic_prod_wrt_dp_ds(
                      dh_db.T, dh_db, long_probs, rows_to_obs, weights=weights)

    ##########
    # Form and return the hessian
    ##########
    if shape_params is not None and intercept_params is not None:
        # Calculate the second derivative of the log-likelihood with respect
        # to the shape parameters. Should have shape (shape_params.shape[0],
        # shape_params.shape[0]). Note that since dp_dh is a 2D dense numpy
        # array and dh_dc is a sparse matrix or dense numpy matrix, to
        # compute the dot product we need to use the * operator
        # `d2_ll_dc2 = -1 * dh_dc.T.dot(dp_dh * dh_dc)`
        d2_ll_dc2 = -1 * quadratic_prod_wrt_dp_ds(
                      dh_dc.T, dh_dc, long_probs, rows_to_obs, weights=weights)

        # Calculate the second derivative of the log-likelihood with respect
        # to the intercept parameters. Should have shape (J - 1, J - 1) where
        # J is the total number of observed alternatives for this dataset.
        # Note that since dp_dh is a 2D dense numpy array and dh_d_alpha is a
        # sparse matrix or dense numpy matrix, to compute the dot product
        # we need to use the * operator
        # `d2_ll_d_alpha2 = -1 * dh_d_alpha.T.dot(dp_dh * dh_d_alpha)`
        d2_ll_d_alpha2 = -1 * quadratic_prod_wrt_dp_ds(
            dh_d_alpha.T, dh_d_alpha, long_probs, rows_to_obs, weights=weights)

        # Calculate the mixed second derivative of the log-likelihood with
        # respect to the intercept and shape parameters. Should have shape
        # (dh_d_alpha.shape[1], dh_dc.shape[1]). Note that since dp_dh is a 2D
        # dense numpy array and dh_dc is a sparse matrix or dense numpy
        # matrix, to compute the dot product we need to use the * operator
        # `d2_ll_dc_d_alpha = -1 * dh_d_alpha.T.dot(dp_dh * dh_dc)`
        d2_ll_dc_d_alpha = -1 * quadratic_prod_wrt_dp_ds(
            dh_d_alpha.T, dh_dc, long_probs, rows_to_obs, weights=weights)

        # Calculate the mixed partial derivative of the log-likelihood with
        # respect to the utility coefficients and then with respect to the
        # shape parameters. Should have shape (design.shape[0],
        # shape_params.shape[0]). Note that since dp_dh is a 2D dense numpy
        # array and dh_dc is a sparse matrix or dense numpy matrix, to
        # compute the dot product we need to use the * operator
        # `d2_ll_dc_db = -1 * dh_db.T.dot(dp_dh * dh_dc)`
        d2_ll_dc_db = -1 * quadratic_prod_wrt_dp_ds(
                      dh_db.T, dh_dc, long_probs, rows_to_obs, weights=weights)

        # Calculate the mixed partial derivative of the log-likelihood with
        # respect to the utility coefficients and then with respect to the
        # intercept parameters. Should have shape (design.shape[0],
        # intercept_params.shape[0]). Note that since dp_dh is a 2D dense numpy
        # array and dh_d_alpha is a sparse matrix or dense numpy matrix, to
        # compute the dot product we need to use the * operator
        # `d2_ll_d_alpha_db = -1 * dh_db.T.dot(dp_dh * dh_d_alpha)`
        d2_ll_d_alpha_db = -1 * quadratic_prod_wrt_dp_ds(
                 dh_db.T, dh_d_alpha, long_probs, rows_to_obs, weights=weights)

        # Form the 3 by 3 partitioned hessian of 2nd derivatives
        top_row = np.concatenate((d2_ll_dc2,
                                  d2_ll_dc_d_alpha.T,
                                  d2_ll_dc_db.T), axis=1)
        middle_row = np.concatenate((d2_ll_dc_d_alpha,
                                     d2_ll_d_alpha2,
                                     d2_ll_d_alpha_db.T), axis=1)
        last_row = np.concatenate((d2_ll_dc_db,
                                   d2_ll_d_alpha_db,
                                   d2_ll_db2), axis=1)
        hess = np.concatenate((top_row,
                               middle_row,
                               last_row), axis=0)

    elif shape_params is not None and intercept_params is None:
        # Calculate the second derivative of the log-likelihood with respect
        # to the shape parameters. Should have shape (shape_params.shape[0],
        # shape_params.shape[0]). Note that since dp_dh is a 2D dense numpy
        # array and dh_dc is a sparse matrix or dense numpy matrix, to
        # compute the dot product we need to use the * operator
        # `d2_ll_dc2 = -1 * dh_dc.T.dot(dp_dh * dh_dc)`
        d2_ll_dc2 = -1 * quadratic_prod_wrt_dp_ds(
                      dh_dc.T, dh_dc, long_probs, rows_to_obs, weights=weights)

        # Calculate the mixed partial derivative of the log-likelihood with
        # respect to the utility coefficients and then with respect to the
        # shape parameters. Should have shape (design.shape[0],
        # shape_params.shape[0]). Note that since dp_dh is a 2D dense numpy
        # array and dh_dc is a sparse matrix or dense numpy matrix, to
        # compute the dot product we need to use the * operator
        # `d2_ll_dc_db = -1 * dh_db.T.dot(dp_dh * dh_dc)`
        d2_ll_dc_db = -1 * quadratic_prod_wrt_dp_ds(
                      dh_db.T, dh_dc, long_probs, rows_to_obs, weights=weights)


        hess = np.concatenate((np.concatenate((d2_ll_dc2,
                                               d2_ll_dc_db.T), axis=1),
                               np.concatenate((d2_ll_dc_db,
                                               d2_ll_db2), axis=1)), axis=0)

    elif shape_params is None and intercept_params is not None:
        # Calculate the second derivative of the log-likelihood with respect
        # to the intercept parameters. Should have shape (J - 1, J - 1) where
        # J is the total number of observed alternatives for this dataset.
        # Note that since dp_dh is a 2D dense numpy array and dh_d_alpha is a
        # sparse matrix or dense numpy matrix, to compute the dot product
        # we need to use the * operator
        # `d2_ll_d_alpha2 = -1 * dh_d_alpha.T.dot(dp_dh * dh_d_alpha)`
        d2_ll_d_alpha2 = -1 * quadratic_prod_wrt_dp_ds(
            dh_d_alpha.T, dh_d_alpha, long_probs, rows_to_obs, weights=weights)

        # Calculate the mixed partial derivative of the log-likelihood with
        # respect to the utility coefficients and then with respect to the
        # intercept parameters. Should have shape (design.shape[0],
        # intercept_params.shape[0]). Note that since dp_dh is a 2D dense numpy
        # array and dh_d_alpha is a sparse matrix or dense numpy matrix, to
        # compute the dot product we need to use the * operator
        # `d2_ll_d_alpha_db = -1 * dh_db.T.dot(dp_dh * dh_d_alpha)`
        d2_ll_d_alpha_db = -1 * quadratic_prod_wrt_dp_ds(
                 dh_db.T, dh_d_alpha, long_probs, rows_to_obs, weights=weights)

        hess = np.concatenate((np.concatenate((d2_ll_d_alpha2,
                                               d2_ll_d_alpha_db.T), axis=1),
                               np.concatenate((d2_ll_d_alpha_db,
                                               d2_ll_db2), axis=1)), axis=0)
    else:
        hess = d2_ll_db2

    if ridge is not None:
        hess -= 2 * ridge * np.identity(hess.shape[0])

    # Make sure we are returning standard numpy arrays
    if isinstance(hess, np.matrixlib.defmatrix.matrix):
        hess = np.asarray(hess)

    return hess


def calc_fisher_info_matrix(beta,
                            design,
                            alt_IDs,
                            rows_to_obs,
                            rows_to_alts,
                            choice_vector,
                            utility_transform,
                            transform_first_deriv_c,
                            transform_first_deriv_v,
                            transform_deriv_alpha,
                            intercept_params,
                            shape_params,
                            ridge,
                            weights):
    """
    Parameters
    ----------
    beta : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features).
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D ndarray.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_alts : 2D ndarray
        There should be one row per observation per available alternative and
        one column per possible alternative. This matrix maps the rows of the
        design matrix to the possible alternatives for this dataset.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a
        1D array whose elements contain the appropriately transformed
        systematic utility values, based on the current model being evaluated.
    transform_first_deriv_c : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, the `rows_to_alts` array, (shape parameters if there
        are any) and miscellaneous args and kwargs. Should return a 2D array
        whose elements contain the derivative of the tranformed utilities with
        respect to the vector of shape parameters. The dimensions of the
        returned vector should be `(design.shape[0], num_alternatives)`.
    transform_first_deriv_v : callable.
        Must accept a 1D array of systematic utility values, a 1D array of
         alternative IDs, (shape parameters if there are any) and miscellaneous
         args and kwargs. Should return a 2D array whose elements contain the
         derivative of the utility tranformation vector with respect to the
         vector of systematic utilities. The dimensions of the returned vector
         should be `(design.shape[0],  design.shape[0])`.
    shape_params : None or 1D ndarray.
        If an array, each element should be an int, float, or long. There
        should be one value per shape parameter of the model being used.
        Default == None.
    ridge : int, float, long, or None.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization. Default = None.
    weights : 1D ndarray or None.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.

    Returns
    -------
    fisher_matrix : 2D numpy array.
        It will be a square matrix, with one row and one column for each shape,
        intercept, and index coefficient. Contains the BHHH approximation to
        the Fisher Information matrix of the log likelihood.
    """
    # Calculate the systematic utility for each alternative for each individual
    sys_utilities = design.dot(beta)

    # Calculate the probability that the individual associated with a given row
    # chooses the alternative associated with a given row.
    long_probs = calc_probabilities(beta,
                                    design,
                                    alt_IDs,
                                    rows_to_obs,
                                    rows_to_alts,
                                    utility_transform,
                                    intercept_params=intercept_params,
                                    shape_params=shape_params,
                                    return_long_probs=True)

    # Calculate the weights for the sample
    if weights is None:
        weights = np.ones(design.shape[0])
    weights_per_obs =\
        np.max(rows_to_obs.toarray() * weights[:, None], axis=0)

    ##########
    # Get the required matrices
    ##########
    # Differentiate the transformed utilities with respect to the shape params
    dh_dc = transform_first_deriv_c(sys_utilities, alt_IDs,
                                    rows_to_alts, shape_params)
    # Differentiate the transformed utilities with respect to the systematic
    # utilities
    dh_dv = transform_first_deriv_v(sys_utilities, alt_IDs,
                                    rows_to_alts, shape_params)
    # Differentiate the transformed utilities by the intercept params
    # Note that dh_d_alpha should be a sparse array
    dh_d_alpha = transform_deriv_alpha(sys_utilities, alt_IDs,
                                       rows_to_alts, intercept_params)
    # Differentiate the transformed utilities with respect to the utility
    # coefficients. This should be a dense numpy array.
    dh_db = np.asarray(dh_dv.dot(design))
    # Differentiate the log likelihood w/ respect to the transformed utilities
    d_ll_dh = (choice_vector - long_probs)[np.newaxis, :]

    ##########
    # Create the matrix where each row represents the gradient of a particular
    # observations log-likelihood with respect to the shape parameters and
    # beta, depending on whether there are shape parameters being estimated
    ##########
    if shape_params is not None and intercept_params is not None:
        if isinstance(dh_dc, np.matrixlib.defmatrix.matrix):
            # Note that the '.A' transforms the matrix into a numpy ndarray
            gradient_vec = d_ll_dh.T * np.concatenate((dh_dc.A,
                                                       dh_d_alpha.toarray(),
                                                       dh_db), axis=1)
        else:
            gradient_vec = d_ll_dh.T * np.concatenate((dh_dc.toarray(),
                                                       dh_d_alpha.toarray(),
                                                       dh_db), axis=1)
    elif shape_params is not None and intercept_params is None:
        if isinstance(dh_dc, np.matrixlib.defmatrix.matrix):
            # Note that the '.A' transforms the matrix into a numpy ndarray
            gradient_vec = d_ll_dh.T * np.concatenate((dh_dc.A, dh_db), axis=1)
        else:
            gradient_vec = d_ll_dh.T * np.concatenate((dh_dc.toarray(),
                                                       dh_db), axis=1)
    elif shape_params is None and intercept_params is not None:
        # Note '.to_array()' is used because dh_d_alpha will be a sparse
        # matrix, but we want a 2D numpy array.
        gradient_vec = d_ll_dh.T * np.concatenate((dh_d_alpha.toarray(),
                                                   dh_db), axis=1)
    else:
        gradient_vec = d_ll_dh.T * dh_db

    # Make sure that we calculate the gradient PER OBSERVATION
    # and then take the outer products of those gradients.
    # Note that this is different than taking the outer products of the
    # gradient of the log-likelihood per available alternative per observation
    gradient_vec = rows_to_obs.T.dot(gradient_vec)

    # Compute and return the outer product of each row of the gradient
    # with itself. Then sum these individual matrices together. The line below
    # does the same computation just with less memory and time.
    fisher_matrix =\
        gradient_vec.T.dot(np.multiply(weights_per_obs[:, None], gradient_vec))

    if ridge is not None:
        # The rational behind adding 2 * ridge is that the fisher information
        # matrix should approximate the hessian and in the hessian we add
        # 2 * ridge at the end. I don't know if this is the correct way to
        # calculate the Fisher Information in ridge regression models.
        fisher_matrix -= 2 * ridge

    return fisher_matrix


def calc_asymptotic_covariance(hessian, fisher_info_matrix):
    """
    Parameters
    ----------
    hessian : 2D ndarray.
        It should have shape `(num_vars, num_vars)`. It is the matrix of second
        derivatives of the total loss across the dataset, with respect to each
        pair of coefficients being estimated.
    fisher_info_matrix : 2D ndarray.
        It should have a shape of `(num_vars, num_vars)`.  It is the
        approximation of the negative of the expected hessian formed by taking
        the outer product of (each observation's gradient of the loss function)
        with itself, and then summing across all observations.

    Returns
    -------
    huber_white_matrix : 2D ndarray.
        Will have shape `(num_vars, num_vars)`. The entries in the returned
        matrix are calculated by the following formula:
        `hess_inverse * fisher_info_matrix * hess_inverse`.
    """
    # Calculate the inverse of the hessian
    hess_inv = scipy.linalg.inv(hessian)

    return np.dot(hess_inv, np.dot(fisher_info_matrix, hess_inv))
