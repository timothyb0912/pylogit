"""
@author:    Timothy Brathwaite
@name:      Bootstrap ABC
@summary:   This module provides functions to calculate the approximate
            bootstrap confidence (ABC) intervals.
"""
import sys
from copy import deepcopy

import numpy as np
from scipy.stats import norm
from scipy.sparse import isspmatrix_csr

from .bootstrap_utils import check_conf_percentage_validity
from .bootstrap_utils import get_alpha_from_conf_percentage
from .bootstrap_utils import combine_conf_endpoints

# Create a value to be used as 'a very small number' when performing the finite
# difference calculations needed to estimate the empirical influence function.
EPSILON = sys.float_info.epsilon


def ensure_model_obj_has_mapping_constructor(model_obj):
    """
    Ensure that `model_obj` has a 'get_mappings_for_fit' method. Raises a
    helpful ValueError if otherwise.
    """
    if not hasattr(model_obj, "get_mappings_for_fit"):
        msg = "model_obj MUST have a 'get_mappings_for_fit' method."
        raise ValueError(msg)
    return None


def ensure_rows_to_obs_validity(rows_to_obs):
    """
    Ensure that `rows_to_obs` is None or a 2D scipy sparse CSR matrix. Raises a
    helpful ValueError if otherwise.
    """
    if rows_to_obs is not None and not isspmatrix_csr(rows_to_obs):
        msg = "rows_to_obs MUST be a 2D scipy sparse row matrix."
        raise ValueError(msg)
    return None


def ensure_wide_weights_is_1D_or_2D_ndarray(wide_weights):
    """
    Ensures that `wide_weights` is a 1D or 2D ndarray. Raises a helpful
    ValueError if otherwise.
    """
    if not isinstance(wide_weights, np.ndarray):
        msg = "wide_weights MUST be a ndarray."
        raise ValueError(msg)
    ndim = wide_weights.ndim
    if not 0 < ndim < 3:
        msg = "wide_weights MUST be a 1D or 2D ndarray."
        raise ValueError(msg)
    return None


def check_validity_of_long_form_args(model_obj, wide_weights, rows_to_obs):
    """
    Ensures the args to `create_long_form_weights` have expected properties.
    """
    # Ensure model_obj has the necessary method for create_long_form_weights
    ensure_model_obj_has_mapping_constructor(model_obj)
    # Ensure wide_weights is a 1D or 2D ndarray.
    ensure_wide_weights_is_1D_or_2D_ndarray(wide_weights)
    # Ensure rows_to_obs is a scipy sparse matrix
    ensure_rows_to_obs_validity(rows_to_obs)
    return None


def create_long_form_weights(model_obj, wide_weights, rows_to_obs=None):
    """
    Converts an array of weights with one element per observation (wide-format)
    to an array of weights with one element per observation per available
    alternative (long-format).

    Parameters
    ----------
    model_obj : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    wide_weights : 1D or 2D ndarray.
        Should contain one element or one column per observation in
        `model_obj.data`, depending on whether `wide_weights` is 1D or 2D
        respectively. These elements should be the weights for optimizing the
        model's objective function for estimation.
    rows_to_obs : 2D scipy sparse array.
        A mapping matrix of zeros and ones, were `rows_to_obs[i, j]` is one if
        row `i` of the long-format data belongs to observation `j` and zero
        otherwise.

    Returns
    -------
    long_weights : 1D or 2D ndarray.
        Should contain one element or one column per observation in
        `model_obj.data`, depending on whether `wide_weights` is 1D or 2D
        respectively. These elements should be the weights from `wide_weights`,
        simply mapping each observation's weight to the corresponding row in
        the long-format data.
    """
    # Ensure argument validity
    check_validity_of_long_form_args(model_obj, wide_weights, rows_to_obs)
    # Get a rows_to_obs mapping matrix.
    if rows_to_obs is None:
        rows_to_obs = model_obj.get_mappings_for_fit()['rows_to_obs']
    # Create a 2D version of
    wide_weights_2d =\
        wide_weights if wide_weights.ndim == 2 else wide_weights[:, None]
    long_weights = rows_to_obs.dot(wide_weights_2d)
    if wide_weights.ndim == 1:
        long_weights = long_weights.sum(axis=1)
    return long_weights


def calc_finite_diff_terms_for_abc(model_obj,
                                   mle_params,
                                   init_vals,
                                   epsilon,
                                   **fit_kwargs):
    """
    Calculates the terms needed for the finite difference approximations of
    the empirical influence and second order empirical influence functions.

    Parameters
    ----------
    model_obj : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    mle_params : 1D ndarray.
        Should contain the desired model's maximum likelihood point estimate.
    init_vals : 1D ndarray.
        The initial values used to estimate the desired choice model.
    epsilon : positive float.
        Should denote the 'very small' value being used to calculate the
        desired finite difference approximations to the various influence
        functions. Should be 'close' to zero.
    fit_kwargs : additional keyword arguments, optional.
        Should contain any additional kwargs used to alter the default behavior
        of `model_obj.fit_mle` and thereby enforce conformity with how the MLE
        was obtained. Will be passed directly to `model_obj.fit_mle`.

    Returns
    -------
    term_plus : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the finite difference term that comes from adding a small value
        to the observation corresponding to that elements respective row.
    term_minus : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the finite difference term that comes from subtracting a small
        value to the observation corresponding to that elements respective row.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equations 22.32 and 22.36.

    Notes
    -----
    The returned, symbolic value for `term_minus` does not explicitly appear in
    Equations 22.32 or 22.36. However, it is used to compute a midpoint / slope
    approximation to the finite difference derivative used to define the
    empirical influence function.
    """
    # Determine the number of observations in this dataset.
    num_obs = model_obj.data[model_obj.obs_id_col].unique().size
    # Determine the initial weights per observation.
    init_weights_wide = np.ones(num_obs, dtype=float) / num_obs
    # Initialize wide weights for elements of the second order influence array.
    init_wide_weights_plus = (1 - epsilon) * init_weights_wide
    init_wide_weights_minus = (1 + epsilon) * init_weights_wide
    # Initialize the second order influence array
    term_plus = np.empty((num_obs, init_vals.shape[0]), dtype=float)
    term_minus = np.empty((num_obs, init_vals.shape[0]), dtype=float)
    # Get the rows_to_obs mapping matrix for this model.
    rows_to_obs = model_obj.get_mappings_for_fit()['rows_to_obs']
    # Extract the initial weights from the fit kwargs
    new_fit_kwargs = deepcopy(fit_kwargs)
    if fit_kwargs is not None and 'weights' in fit_kwargs:
        orig_weights = fit_kwargs['weights']
        del new_fit_kwargs['weights']
    else:
        orig_weights = 1
    # Make sure we're just getting the point estimate
    new_fit_kwargs['just_point'] = True

    # Populate the second order influence array
    for obs in xrange(num_obs):
        # Note we create the long weights in a for-loop to avoid creating a
        # num_obs by num_obs matrix, which may be a problem for large datasets
        # Get the wide format weights for this observation
        current_wide_weights_plus = init_wide_weights_plus.copy()
        current_wide_weights_plus[obs] += epsilon

        current_wide_weights_minus = init_wide_weights_minus.copy()
        current_wide_weights_minus[obs] -= epsilon
        # Get the long format weights for this observation
        long_weights_plus =\
            (create_long_form_weights(model_obj, current_wide_weights_plus,
                                      rows_to_obs=rows_to_obs) * orig_weights)
        long_weights_minus =\
            (create_long_form_weights(model_obj,
                                      current_wide_weights_minus,
                                      rows_to_obs=rows_to_obs) * orig_weights)
        # Get the needed influence estimates.
        term_plus[obs] = model_obj.fit_mle(init_vals,
                                           weights=long_weights_plus,
                                           **new_fit_kwargs)['x']
        term_minus[obs] = model_obj.fit_mle(init_vals,
                                            weights=long_weights_minus,
                                            **new_fit_kwargs)['x']
    return term_plus, term_minus


def calc_empirical_influence_abc(term_plus,
                                 term_minus,
                                 epsilon):
    """
    Calculates the finite difference, midpoint / slope approximation to the
    empirical influence array needed to compute the approximate boostrap
    confidence (ABC) intervals.

    Parameters
    ----------
    term_plus : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the finite difference term that comes from adding a small value
        to the observation corresponding to that elements respective row.
    term_minus : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the finite difference term that comes from subtracting a small
        value to the observation corresponding to that elements respective row.
    epsilon : positive float.
        Should denote the 'very small' value being used to calculate the
        desired finite difference approximations to the various influence
        functions. Should be 'close' to zero.

    Returns
    -------
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.32.

    Notes
    -----
    This function is based off of the code in Efron's original Bootstrap
    library, written in S-plus. It is a finite difference, midpoint or slope
    approximation of Equation 22.32.
    """
    # Calculate the denominator of each row of the second order influence array
    denominator = 2 * epsilon
    # Calculate the empirical influence array.
    empirical_influence = np.zeros(term_plus.shape)
    diff_idx = ~np.isclose(term_plus, term_minus, atol=1e-12, rtol=0)
    if diff_idx.any():
        empirical_influence[diff_idx] =\
            (term_plus[diff_idx] - term_minus[diff_idx]) / denominator
    return empirical_influence


def calc_2nd_order_influence_abc(mle_params,
                                 term_plus,
                                 term_minus,
                                 epsilon):
    """
    Calculates either a 'positive' finite difference approximation or an
    approximation of a 'positive' finite difference approximation to the the
    2nd order empirical influence array needed to compute the approximate
    boostrap confidence (ABC) intervals. See the 'Notes' section for more
    information on the ambiguous function description.

    Parameters
    ----------
    mle_params : 1D ndarray.
        Should contain the desired model's maximum likelihood point estimate.
    term_plus : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the finite difference term that comes from adding a small value
        to the observation corresponding to that elements respective row.
    term_minus : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the finite difference term that comes from subtracting a small
        value to the observation corresponding to that elements respective row.
    epsilon : positive float.
        Should denote the 'very small' value being used to calculate the
        desired finite difference approximations to the various influence
        functions. Should be 'close' to zero.

    Returns
    -------
    second_order_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the second order empirical influence of the associated
        observation on the associated parameter.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.36.

    Notes
    -----
    This function is based on the code in Efron's original Bootstrap library
    written in S-plus. It is not equivalent to the 'positive' finite difference
    approximation to Equation 22.36, where epsilon is set to a small float. The
    reason for this discrepancy is becaue `term_minus` is not equal to the
    third term in the numerator in Equation 22.36. That term uses
    `(1 - epsilon)P^0` whereas `term_minus` uses `(1 + epsilon)P^0`. At the
    limit, both of these terms would be `P^0` and therefore equal. I think
    Efron's original code was making the assumption that the two terms are
    approximately equal to conserve computational resources. Either that or
    Equation 22.36, as printed, is incorrect because its third term really
    should be `(1 + epsilon)P^0`.
    """
    # Calculate the denominator of each row of the second order influence array
    denominator = epsilon**2
    # Initialize the second term used to calculate each row in the second order
    # influence array
    term_2 = np.broadcast_to(2 * mle_params, term_plus.shape)
    # Calculate the second order empirical influence array.
    second_order_influence = np.zeros(term_plus.shape, dtype=float)
    # Only perform the calculations if the values in the component terms are
    # sufficiently different. This is to prevent floating point errors. atol is
    # simply set to a 'reasonably small' number.
    diff_idx = ~np.isclose(term_plus + term_minus, term_2, atol=1e-12, rtol=0)
    if diff_idx.any():
        second_order_influence[diff_idx] =\
            ((term_plus[diff_idx] - term_2[diff_idx] + term_minus[diff_idx]) /
             denominator)
    return second_order_influence


def calc_influence_arrays_for_abc(model_obj,
                                  mle_est,
                                  init_values,
                                  epsilon,
                                  **fit_kwargs):
    """
    Calculates the empirical influence array and the 2nd order empirical
    influence array needed to compute the approximate boostrap confidence (ABC)
    intervals.

    Parameters
    ----------
    model_obj : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    mle_est : 1D ndarray.
        Should contain the desired model's maximum likelihood point estimate.
    init_vals : 1D ndarray.
        The initial values used to estimate the desired choice model.
    epsilon : positive float.
        Should denote the 'very small' value being used to calculate the
        desired finite difference approximations to the various influence
        functions. Should be 'close' to zero.
    fit_kwargs : additional keyword arguments, optional.
        Should contain any additional kwargs used to alter the default behavior
        of `model_obj.fit_mle` and thereby enforce conformity with how the MLE
        was obtained. Will be passed directly to `model_obj.fit_mle`.

    Returns
    -------
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.
    second_order_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the second order empirical influence of the associated
        observation on the associated parameter.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equations 22.32 and 22.36.
    """
    # Calculate the two arrays of finite difference terms needed for the
    # various influence arrays that we want to calculate.
    term_plus, term_minus = calc_finite_diff_terms_for_abc(model_obj,
                                                           mle_est,
                                                           init_values,
                                                           epsilon,
                                                           **fit_kwargs)
    # Calculate the empirical influence array.
    empirical_influence =\
        calc_empirical_influence_abc(term_plus, term_minus, epsilon)
    # Calculate the second order influence array.
    second_order_influence =\
        calc_2nd_order_influence_abc(mle_est, term_plus, term_minus, epsilon)
    # Return the desired terms
    return empirical_influence, second_order_influence


def calc_std_error_abc(empirical_influence):
    """
    Calculates the standard error of the MLE estimates for use in calculating
    the approximate bootstrap confidence (ABC) intervals.

    Parameters
    ----------
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.

    Returns
    -------
    std_error : 1D ndarray.
        Contains the standard error of the MLE estimates for use in the ABC
        confidence intervals.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.31.
    """
    num_obs = empirical_influence.shape[0]
    std_error = ((empirical_influence**2).sum(axis=0))**0.5 / num_obs
    return std_error


def calc_acceleration_abc(empirical_influence):
    """
    Calculates the acceleration constant for the approximate bootstrap
    confidence (ABC) intervals.

    Parameters
    ----------
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.

    Returns
    -------
    acceleration : 1D ndarray.
        Contains the ABC confidence intervals' estimated acceleration vector.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.34.
    """
    influence_cubed = empirical_influence**3
    influence_squared = empirical_influence**2
    numerator = influence_cubed.sum(axis=0)
    denominator = 6 * (influence_squared.sum(axis=0))**1.5
    acceleration = numerator / denominator
    return acceleration


def calc_bias_abc(second_order_influence):
    """
    Calculates the approximate bias of the MLE estimates for use in calculating
    the approximate bootstrap confidence (ABC) intervals.

    Parameters
    ----------
    second_order_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the second order empirical influence of the associated
        observation on the associated parameter.

    Returns
    -------
    bias : 1D ndarray.
        Contains the approximate bias of the MLE estimates for use in the ABC
        confidence intervals.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.35.
    """
    num_obs = second_order_influence.shape[0]
    constant = 2.0 * num_obs**2
    bias = second_order_influence.sum(axis=0) / constant
    return bias


def calc_quadratic_coef_abc(model_object,
                            mle_params,
                            init_vals,
                            empirical_influence,
                            std_error,
                            epsilon,
                            **fit_kwargs):
    """
    Calculates the quadratic coefficient needed to compute the approximate
    boostrap confidence (ABC) intervals.

    Parameters
    ----------
    model_object : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    mle_params : 1D ndarray.
        Should contain the desired model's maximum likelihood point estimate.
    init_vals : 1D ndarray.
        The initial values used to estimate the desired choice model.
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.
    std_error : 1D ndarray.
        Contains the standard error of the MLE estimates for use in the ABC
        confidence intervals.
    epsilon : positive float.
        Should denote the 'very small' value being used to calculate the
        desired finite difference approximations to the various influence
        functions. Should be 'close' to zero.
    fit_kwargs : additional keyword arguments, optional.
        Should contain any additional kwargs used to alter the default behavior
        of `model_obj.fit_mle` and thereby enforce conformity with how the MLE
        was obtained. Will be passed directly to `model_obj.fit_mle`.
    Returns
    -------
    quadratic_coef : 1D ndarray.
        Contains a measure of nonlinearity of the MLE estimation function as
        one moves in the 'least favorable direction.'

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.37.
    """
    # Determine the number of observations in this dataset.
    num_obs = float(empirical_influence.shape[0])
    # Calculate the standardized version of the empirical influence values
    # assuming mean zero. This is the mean of empirical influence values
    # (sum over n) divided by the standard error of the mean of the empiricial
    # influence values, i.e. the square root of [n^2 * variance] = n * std_err
    # where std_err is the standard error of the sampling distribution of the
    # influence values.
    standardized_influence =\
        empirical_influence / (num_obs**2 * std_error[None, :])
    # Determine the initial weights per observation.
    init_weights_wide = (np.ones(int(num_obs), dtype=float) / num_obs)[:, None]
    # Create the wide weights for the various terms of the quadratic_coef
    term_1_wide_weights =\
        (1 - epsilon) * init_weights_wide + epsilon * standardized_influence
    term_3_wide_weights =\
        (1 - epsilon) * init_weights_wide - epsilon * standardized_influence
    # Get the rows_to_obs mapping matrix for this model.
    rows_to_obs = model_object.get_mappings_for_fit()['rows_to_obs']
    # Initialize the terms for the quadratic coefficients.
    expected_term_shape = (init_vals.shape[0], init_vals.shape[0])
    term_1_array = np.empty(expected_term_shape, dtype=float)
    term_3_array = np.empty(expected_term_shape, dtype=float)
    # Extract the initial weights from the fit kwargs
    new_fit_kwargs = deepcopy(fit_kwargs)
    if fit_kwargs is not None and 'weights' in fit_kwargs:
        orig_weights = fit_kwargs['weights']
        del new_fit_kwargs['weights']
    else:
        orig_weights = 1
    # Make sure we're just getting the point estimate
    new_fit_kwargs['just_point'] = True
    # Calculate the various terms of the quadratic_coef
    for param_id in xrange(expected_term_shape[0]):
        # Get a 'long-format' array of the weights per observation
        term_1_long_weights =\
            (create_long_form_weights(model_object,
                                      term_1_wide_weights[:, param_id],
                                      rows_to_obs=rows_to_obs) * orig_weights)
        term_3_long_weights =\
            (create_long_form_weights(model_object,
                                      term_3_wide_weights[:, param_id],
                                      rows_to_obs=rows_to_obs) * orig_weights)
        # Populate the given row of the term_1 and term_3 arrays.
        term_1_array[param_id] =\
            model_object.fit_mle(init_vals,
                                 weights=term_1_long_weights,
                                 **new_fit_kwargs)['x']
        term_3_array[param_id] =\
            model_object.fit_mle(init_vals,
                                 weights=term_3_long_weights,
                                 **new_fit_kwargs)['x']
    # Extract the desired terms from their 2d arrays
    term_1 = np.diag(term_1_array)
    term_3 = np.diag(term_3_array)
    term_2 = 2 * mle_params
    # Calculate the quadratic coefficient
    quadratic_coef = np.zeros(term_1.shape, dtype=float)
    denominator = epsilon**2
    # Only perform the calculations if the values in the component terms are
    # sufficiently different. This is to prevent floating point errors. Note
    # that atol is simply set to a 'reasonably small' number.
    diff_idx = ~np.isclose(term_1 + term_3, term_2, atol=1e-10, rtol=0)
    if diff_idx.any():
        quadratic_coef[diff_idx] =\
            ((term_1[diff_idx] - term_2[diff_idx] + term_3[diff_idx]) /
             denominator)
    return quadratic_coef


def efron_quadratic_coef_abc(model_object,
                             mle_params,
                             init_vals,
                             empirical_influence,
                             std_error,
                             epsilon,
                             **fit_kwargs):
    """
    Calculates the quadratic coefficient needed to compute the approximate
    boostrap confidence (ABC) intervals.

    Parameters
    ----------
    model_object : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    mle_params : 1D ndarray.
        Should contain the desired model's maximum likelihood point estimate.
    init_vals : 1D ndarray.
        The initial values used to estimate the desired choice model.
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.
    std_error : 1D ndarray.
        Contains the standard error of the MLE estimates for use in the ABC
        confidence intervals.
    epsilon : positive float.
        Should denote the 'very small' value being used to calculate the
        desired finite difference approximations to the various influence
        functions. Should be 'close' to zero.
    fit_kwargs : additional keyword arguments, optional.
        Should contain any additional kwargs used to alter the default behavior
        of `model_obj.fit_mle` and thereby enforce conformity with how the MLE
        was obtained. Will be passed directly to `model_obj.fit_mle`.

    Returns
    -------
    quadratic_coef : 1D ndarray.
        Contains a measure of nonlinearity of the MLE estimation function as
        one moves in the 'least favorable direction.'

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.37.

    Notes
    -----
    This function does not directly implement Equation 22.37. Instead, it
    re-implements the calculations that Efron and Tibshirani use in their
    'abcnon.R' file within the 'bootstrap' library.
    """
    # Determine the number of observations in this dataset.
    num_obs = float(empirical_influence.shape[0])
    # Calculate the standardized version of the empirical influence values
    # assuming mean zero. This is the mean of empirical influence values
    # (sum over n) divided by the standard error of the mean of the empiricial
    # influence values, i.e. the square root of [n^2 * variance] = n * std_err
    # where std_err is the standard error of the sampling distribution of the
    # influence values.
    standardized_influence =\
        empirical_influence / (num_obs**2 * std_error[None, :])
    # Determine the initial weights per observation.
    init_weights_wide = (np.ones(int(num_obs), dtype=float) / num_obs)[:, None]
    # Create the wide weights for the various terms of the quadratic_coef
    term_1_wide_weights = init_weights_wide + epsilon * standardized_influence
    term_3_wide_weights = init_weights_wide - epsilon * standardized_influence
    # Get the rows_to_obs mapping matrix for this model.
    rows_to_obs = model_object.get_mappings_for_fit()['rows_to_obs']
    # Initialize the terms for the quadratic coefficients.
    expected_term_shape = (init_vals.shape[0], init_vals.shape[0])
    term_1_array = np.empty(expected_term_shape, dtype=float)
    term_3_array = np.empty(expected_term_shape, dtype=float)
    # Extract the initial weights from the fit kwargs
    new_fit_kwargs = deepcopy(fit_kwargs)
    if fit_kwargs is not None and 'weights' in fit_kwargs:
        orig_weights = fit_kwargs['weights']
        del new_fit_kwargs['weights']
    else:
        orig_weights = 1
    # Make sure we're just getting the point estimate
    new_fit_kwargs['just_point'] = True
    # Calculate the various terms of the quadratic_coef
    for param_id in xrange(expected_term_shape[0]):
        # Get a 'long-format' array of the weights per observation
        term_1_long_weights =\
            (create_long_form_weights(model_object,
                                      term_1_wide_weights[:, param_id],
                                      rows_to_obs=rows_to_obs) * orig_weights)
        term_3_long_weights =\
            (create_long_form_weights(model_object,
                                      term_3_wide_weights[:, param_id],
                                      rows_to_obs=rows_to_obs) * orig_weights)
        # Populate the given row of the term_1 and term_3 arrays.
        term_1_array[param_id] =\
            model_object.fit_mle(init_vals,
                                 weights=term_1_long_weights,
                                 **new_fit_kwargs)['x']
        term_3_array[param_id] =\
            model_object.fit_mle(init_vals,
                                 weights=term_3_long_weights,
                                 **new_fit_kwargs)['x']
    # Extract the desired terms from their 2d arrays
    term_1 = np.diag(term_1_array)
    term_3 = np.diag(term_3_array)
    term_2 = 2 * mle_params
    # Calculate the quadratic coefficient
    quadratic_coef = np.zeros(term_1.shape, dtype=float)
    denominator = 2 * std_error * epsilon**2
    # Only perform the calculations if the values in the component terms are
    # sufficiently different. This is to prevent floating point errors. Note
    # that atol is simply set to a 'reasonably small' number.
    diff_idx = ~np.isclose(term_1 + term_3, term_2, atol=1e-10, rtol=0)
    if diff_idx.any():
        quadratic_coef[diff_idx] =\
            ((term_1[diff_idx] - term_2[diff_idx] + term_3[diff_idx]) /
             denominator)
    return quadratic_coef


def calc_total_curvature_abc(bias, std_error, quadratic_coef):
    """
    Calculate the total curvature of the level surface of the weight vector,
    where the set of weights in the surface are those where the weighted MLE
    equals the original (i.e. the equal-weighted) MLE.

    Parameters
    ----------
    bias : 1D ndarray.
        Contains the approximate bias of the MLE estimates for use in the ABC
        confidence intervals.
    std_error : 1D ndarray.
        Contains the standard error of the MLE estimates for use in the ABC
        confidence intervals.
    quadratic_coef : 1D ndarray.
        Contains a measure of nonlinearity of the MLE estimation function as
        one moves in the 'least favorable direction.'

    Returns
    -------
    total_curvature : 1D ndarray of scalars.
        Denotes the total curvature of the level surface of the weight vector.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6. Equation 22.39.
    """
    total_curvature = (bias / std_error) - quadratic_coef
    return total_curvature


def calc_bias_correction_abc(acceleration, total_curvature):
    """
    Calculate the bias correction constant for the approximate bootstrap
    confidence (ABC) intervals.

    Parameters
    ----------
    acceleration : 1D ndarray of scalars.
        Should contain the ABC intervals' estimated acceleration constants.
    total_curvature : 1D ndarray of scalars.
        Should denote the ABC intervals' computred total curvature values.

    Returns
    -------
    bias_correction : 1D ndarray of scalars.
        Contains the computed bias correction for the MLE estimates that the
        ABC interval is being computed for.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.40, line 1.
    """
    inner_arg = 2 * norm.cdf(acceleration) * norm.cdf(-1 * total_curvature)
    bias_correction = norm.ppf(inner_arg)
    return bias_correction


def calc_endpoint_from_percentile_abc(model_obj,
                                      init_vals,
                                      percentile,
                                      bias_correction,
                                      acceleration,
                                      std_error,
                                      empirical_influence,
                                      **fit_kwargs):
    """
    Calculates the endpoint of the 1-tailed, (percentile)% confidence interval.
    Note this interval spans from negative infinity to the calculated endpoint.

    Parameters
    ----------
    model_obj : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    init_vals : 1D ndarray.
        The initial values used to estimate the desired choice model.
    percentile : scalar in (0.0, 100.0).
        Denotes the percentile of the standard normal distribution at which
        we'd like to evaluate the inverse cumulative distribution function and
        then convert this standardized value back to our approximate bootstrap
        distribution.
    bias_correction : 1D ndarray of scalars.
        Contains the computed bias correction for the MLE estimates that the
        ABC interval is being computed for.
    acceleration : 1D ndarray of scalars.
        Should contain the ABC intervals' estimated acceleration constants.
    std_error : 1D ndarray.
        Contains the standard error of the MLE estimates for use in the ABC
        confidence intervals.
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.
    fit_kwargs : additional keyword arguments, optional.
        Should contain any additional kwargs used to alter the default behavior
        of `model_obj.fit_mle` and thereby enforce conformity with how the MLE
        was obtained. Will be passed directly to `model_obj.fit_mle`.

    Returns
    -------
    endpoint : 1D ndarray.
        Contains the endpoint from our approximate bootstrap distribution's
        1-tailed, upper `percentile` confidence interval.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.33.
    """
    # Get the bias corrected standard normal value for the relevant percentile.
    # We multiply percentile by 0.01 because the ppf function requires decimals
    # in [0.0, 1.0].
    bias_corrected_z = bias_correction + norm.ppf(percentile * 0.01)
    # Calculate the multiplier for the least favorable direction
    # (i.e. the empirical influence function).
    lam = bias_corrected_z / (1 - acceleration * bias_corrected_z)**2
    multiplier = lam / std_error
    # Calculate the initial weights
    num_obs = empirical_influence.shape[0]
    init_weights_wide = np.ones(num_obs, dtype=float)[:, None] / num_obs
    # Get the necessary weight adjustment term for calculating the endpoint.
    weight_adjustment_wide = (multiplier[None, :] * empirical_influence)
    wide_weights_all_params = init_weights_wide + weight_adjustment_wide
    # Extract the initial weights from the fit kwargs
    new_fit_kwargs = deepcopy(fit_kwargs)
    if fit_kwargs is not None and 'weights' in fit_kwargs:
        orig_weights = fit_kwargs['weights']
        del new_fit_kwargs['weights']
    else:
        orig_weights = np.ones(model_obj.data.shape[0], dtype=float)
    # Make sure we're just getting the point estimate
    new_fit_kwargs['just_point'] = True
    # Get a long format version of the weights needed to compute the endpoints
    long_weights_all_params =\
        (create_long_form_weights(model_obj, wide_weights_all_params) *
         orig_weights[:, None])
    # Initialize the array to store the desired enpoints
    num_params = init_vals.shape[0]
    endpoint = np.empty(num_params, dtype=float)
    # Populate the endpoint array
    for param_id in xrange(num_params):
        current_weights = long_weights_all_params[:, param_id]
        current_estimate = model_obj.fit_mle(init_vals,
                                             weights=current_weights,
                                             **new_fit_kwargs)['x']
        endpoint[param_id] = current_estimate[param_id]
    return endpoint


def efron_endpoint_from_percentile_abc(model_obj,
                                       init_vals,
                                       percentile,
                                       bias_correction,
                                       acceleration,
                                       std_error,
                                       empirical_influence,
                                       **fit_kwargs):
    """
    Calculates the endpoint of the 1-tailed, (percentile)% confidence interval.
    Note this interval spans from negative infinity to the calculated endpoint.

    Parameters
    ----------
    model_obj : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    init_vals : 1D ndarray.
        The initial values used to estimate the desired choice model.
    percentile : scalar in (0.0, 100.0).
        Denotes the percentile of the standard normal distribution at which
        we'd like to evaluate the inverse cumulative distribution function and
        then convert this standardized value back to our approximate bootstrap
        distribution.
    bias_correction : 1D ndarray of scalars.
        Contains the computed bias correction for the MLE estimates that the
        ABC interval is being computed for.
    acceleration : 1D ndarray of scalars.
        Should contain the ABC intervals' estimated acceleration constants.
    std_error : 1D ndarray.
        Contains the standard error of the MLE estimates for use in the ABC
        confidence intervals.
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.
    fit_kwargs : additional keyword arguments, optional.
        Should contain any additional kwargs used to alter the default behavior
        of `model_obj.fit_mle` and thereby enforce conformity with how the MLE
        was obtained. Will be passed directly to `model_obj.fit_mle`.

    Returns
    -------
    endpoint : 1D ndarray.
        Contains the endpoint from our approximate bootstrap distribution's
        1-tailed, upper `percentile` confidence interval.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6, Equation 22.33.

    Notes
    -----
    This function does not directly implement Equation 22.33. Instead, it
    implements Efron's endpoint calculations from 'abcnon.R' in the 'bootstrap'
    library in R. It is not clear where these calculations come from, and
    if/how these calculations are equivalent to Equation 22.33.
    """
    # Get the bias corrected standard normal value for the relevant percentile.
    # We multiply percentile by 0.01 because the ppf function requires decimals
    # in [0.0, 1.0].
    bias_corrected_z = bias_correction + norm.ppf(percentile * 0.01)
    # Determine the number of observations
    num_obs = empirical_influence.shape[0]
    # Calculate the multiplier for the least favorable direction
    # (i.e. the empirical influence function).
    lam = bias_corrected_z / (1 - acceleration * bias_corrected_z)**2
    multiplier = lam / (std_error * num_obs**2)
    # Calculate the initial weights
    init_weights_wide = np.ones(num_obs, dtype=float)[:, None] / num_obs
    # Get the necessary weight adjustment term for calculating the endpoint.
    weight_adjustment_wide = (multiplier[None, :] * empirical_influence)
    wide_weights_all_params = init_weights_wide + weight_adjustment_wide
    # Extract the initial weights from the fit kwargs
    new_fit_kwargs = deepcopy(fit_kwargs)
    if fit_kwargs is not None and 'weights' in fit_kwargs:
        orig_weights = fit_kwargs['weights']
        del new_fit_kwargs['weights']
    else:
        orig_weights = np.ones(model_obj.data.shape[0], dtype=float)
    # Make sure we're just getting the point estimate
    new_fit_kwargs['just_point'] = True
    # Get a long format version of the weights needed to compute the endpoints
    long_weights_all_params =\
        (create_long_form_weights(model_obj, wide_weights_all_params) *
         orig_weights[:, None])
    # Initialize the array to store the desired enpoints
    num_params = init_vals.shape[0]
    endpoint = np.empty(num_params, dtype=float)
    # Populate the endpoint array
    for param_id in xrange(num_params):
        current_weights = long_weights_all_params[:, param_id]
        current_estimate = model_obj.fit_mle(init_vals,
                                             weights=current_weights,
                                             **new_fit_kwargs)['x']
        endpoint[param_id] = current_estimate[param_id]
    return endpoint


# def calc_endpoints_for_abc_confidence_interval(conf_percentage,
#                                                model_obj,
#                                                init_vals,
#                                                bias_correction,
#                                                acceleration,
#                                                std_error,
#                                                empirical_influence,
#                                                **fit_kwargs):
#     """
#     Calculates the endpoints of the equal-tailed, `conf_percentage`%
#     approximate bootstrap confidence (ABC) interval.
#
#     Parameters
#     ----------
#     conf_percentage : scalar in the interval (0.0, 100.0).
#         Denotes the confidence-level for the returned endpoints. For
#         instance, to calculate a 95% confidence interval, pass `95`.
#     model_obj : an instance or sublcass of the MNDC class.
#         Should be the model object that corresponds to the model we are
#         constructing the bootstrap confidence intervals for.
#     init_vals : 1D ndarray.
#         The initial values used to estimate the desired choice model.
#     bias_correction : 1D ndarray of scalars.
#         Contains the computed bias correction for the MLE estimates that the
#         ABC interval is being computed for.
#     acceleration : 1D ndarray of scalars.
#         Should contain the ABC intervals' estimated acceleration constants.
#     std_error : 1D ndarray.
#         Contains the standard error of the MLE estimates for use in the ABC
#         confidence intervals.
#     empirical_influence : 2D ndarray.
#         Should have one row for each observation. Should have one column for
#         each parameter in the parameter vector being estimated. Elements
#         should denote the empirical influence of the associated observation
#         on the associated parameter.
#     fit_kwargs : additional keyword arguments, optional.
#         Should contain any additional kwargs used to alter the default
#         behavior of `model_obj.fit_mle` and thereby enforce conformity with
#         how the MLE was obtained. Will be passed directly to
#         `model_obj.fit_mle`.
#
#     Returns
#     -------
#     lower_endpoint, upper_endpoint : 1D ndarray.
#         Contains the lower or upper endpoint, respectively, from our
#         `conf_percentage`% ABC interval.
#     """
#     # Calculate the percentiles for the lower and upper endpoints
#     alpha_percent = get_alpha_from_conf_percentage(conf_percentage)
#     # Note that we divide by 100 because scipy.stats.norm.ppf only accepts
#     # floats between 0.0 and 1.0
#     lower_percentile = alpha_percent / 2.0
#     upper_percentile = 100 - lower_percentile
#     # Calculate the lower endpoint
#     lower_endpoint = calc_endpoint_from_percentile_abc(model_obj,
#                                                        init_vals,
#                                                        lower_percentile,
#                                                        bias_correction,
#                                                        acceleration,
#                                                        std_error,
#                                                        empirical_influence,
#                                                        **fit_kwargs)
#     # Calculate the upper endpoint
#     upper_endpoint = calc_endpoint_from_percentile_abc(model_obj,
#                                                        init_vals,
#                                                        upper_percentile,
#                                                        bias_correction,
#                                                        acceleration,
#                                                        std_error,
#                                                        empirical_influence,
#                                                        **fit_kwargs)
#     return lower_endpoint, upper_endpoint


def efron_endpoints_for_abc_confidence_interval(conf_percentage,
                                                model_obj,
                                                init_vals,
                                                bias_correction,
                                                acceleration,
                                                std_error,
                                                empirical_influence,
                                                **fit_kwargs):
    """
    Calculates the endpoints of the equal-tailed, `conf_percentage`%
    approximate bootstrap confidence (ABC) interval.

    Parameters
    ----------
    conf_percentage : scalar in the interval (0.0, 100.0).
        Denotes the confidence-level for the returned endpoints. For instance,
        to calculate a 95% confidence interval, pass `95`.
    model_obj : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    init_vals : 1D ndarray.
        The initial values used to estimate the desired choice model.
    bias_correction : 1D ndarray of scalars.
        Contains the computed bias correction for the MLE estimates that the
        ABC interval is being computed for.
    acceleration : 1D ndarray of scalars.
        Should contain the ABC intervals' estimated acceleration constants.
    std_error : 1D ndarray.
        Contains the standard error of the MLE estimates for use in the ABC
        confidence intervals.
    empirical_influence : 2D ndarray.
        Should have one row for each observation. Should have one column for
        each parameter in the parameter vector being estimated. Elements should
        denote the empirical influence of the associated observation on the
        associated parameter.
    fit_kwargs : additional keyword arguments, optional.
        Should contain any additional kwargs used to alter the default behavior
        of `model_obj.fit_mle` and thereby enforce conformity with how the MLE
        was obtained. Will be passed directly to `model_obj.fit_mle`.

    Returns
    -------
    lower_endpoint, upper_endpoint : 1D ndarray.
        Contains the lower or upper endpoint, respectively, from our
        `conf_percentage`% ABC interval.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6.

    Notes
    -----
    This function does not directly implement Equation 22.33. Instead, it
    implements Efron's endpoint calculations from 'abcnon.R' in the 'bootstrap'
    library in R. It is not clear where these calculations come from, and
    if/how these calculations are equivalent to Equation 22.33.
    """
    # Calculate the percentiles for the lower and upper endpoints
    alpha_percent = get_alpha_from_conf_percentage(conf_percentage)
    # Note that we divide by 100 because scipy.stats.norm.ppf only accepts
    # floats between 0.0 and 1.0
    lower_percentile = alpha_percent / 2.0
    upper_percentile = 100 - lower_percentile
    # Calculate the lower endpoint
    lower_endpoint = efron_endpoint_from_percentile_abc(model_obj,
                                                        init_vals,
                                                        lower_percentile,
                                                        bias_correction,
                                                        acceleration,
                                                        std_error,
                                                        empirical_influence,
                                                        **fit_kwargs)
    # Calculate the upper endpoint
    upper_endpoint = efron_endpoint_from_percentile_abc(model_obj,
                                                        init_vals,
                                                        upper_percentile,
                                                        bias_correction,
                                                        acceleration,
                                                        std_error,
                                                        empirical_influence,
                                                        **fit_kwargs)
    return lower_endpoint, upper_endpoint


def calc_abc_interval(model_obj,
                      mle_params,
                      init_vals,
                      conf_percentage,
                      epsilon=0.001,
                      **fit_kwargs):
    """
    Calculate 'approximate bootstrap confidence' intervals.

    Parameters
    ----------
    model_obj : an instance or sublcass of the MNDC class.
        Should be the model object that corresponds to the model we are
        constructing the bootstrap confidence intervals for.
    mle_params : 1D ndarray.
        Should contain the desired model's maximum likelihood point estimate.
    init_vals : 1D ndarray.
        The initial values used to estimate the desired choice model.
    conf_percentage : scalar in the interval (0.0, 100.0).
        Denotes the confidence-level of the returned confidence interval. For
        instance, to calculate a 95% confidence interval, pass `95`.
    epsilon : positive float, optional.
        Should denote the 'very small' value being used to calculate the
        desired finite difference approximations to the various influence
        functions. Should be close to zero. Default == sys.float_info.epsilon.
    fit_kwargs : additional keyword arguments, optional.
        Should contain any additional kwargs used to alter the default behavior
        of `model_obj.fit_mle` and thereby enforce conformity with how the MLE
        was obtained. Will be passed directly to `model_obj.fit_mle`.

    Returns
    -------
    conf_intervals : 2D ndarray.
        The shape of the returned array will be `(2, samples.shape[1])`. The
        first row will correspond to the lower value in the confidence
        interval. The second row will correspond to the upper value in the
        confidence interval. There will be one column for each element of the
        parameter vector being estimated.

    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the Bootstrap.
        CRC press, 1994. Section 22.6.
    DiCiccio, Thomas J., and Bradley Efron. "Bootstrap confidence intervals."
        Statistical science (1996): 189-212.
    """
    # Check validity of arguments
    check_conf_percentage_validity(conf_percentage)
    # Calculate the empirical influence component and second order empirical
    # influence component for each observation
    empirical_influence, second_order_influence =\
        calc_influence_arrays_for_abc(model_obj,
                                      mle_params,
                                      init_vals,
                                      epsilon,
                                      **fit_kwargs)
    # Calculate the acceleration constant for the ABC interval.
    acceleration = calc_acceleration_abc(empirical_influence)
    # Use the delta method to calculate the standard error of the MLE parameter
    # estimate of the model using the original data.
    std_error = calc_std_error_abc(empirical_influence)
    # Approximate the bias of the MLE parameter estimates.
    bias = calc_bias_abc(second_order_influence)

    # Calculate the quadratic coefficient. Note we are using the 'efron'
    # version of the desired function because the direct implementation of the
    # formulas in the textbook don't return the correct results. The 'efron'
    # versions re-implement the calculations from 'abcnon.R' in Efron's
    # 'bootstrap' library in R.

    # quadratic_coef = calc_quadratic_coef_abc(model_obj,
    #                                          mle_params,
    #                                          init_vals,
    #                                          empirical_influence,
    #                                          std_error,
    #                                          epsilon,
    #                                          **fit_kwargs)
    quadratic_coef = efron_quadratic_coef_abc(model_obj,
                                              mle_params,
                                              init_vals,
                                              empirical_influence,
                                              std_error,
                                              epsilon,
                                              **fit_kwargs)

    # Calculate the total curvature of the level surface of the weight vector,
    # where the set of weights in the surface are those where the weighted MLE
    # equals the original (i.e. the equal-weighted) MLE.
    total_curvature = calc_total_curvature_abc(bias, std_error, quadratic_coef)
    # Calculate the bias correction constant.
    bias_correction = calc_bias_correction_abc(acceleration, total_curvature)

    # Calculate the lower limit of the conf_percentage confidence intervals
    # Note we are using the 'efron' version of the desired function because the
    # direct implementation of the formulas in the textbook don't return the
    # correct results. The 'efron' versions re-implement the calculations from
    # 'abcnon.R' in Efron's 'bootstrap' library in R.

    # lower_endpoint, upper_endpoint =\
    #     calc_endpoints_for_abc_confidence_interval(conf_percentage,
    #                                                model_obj,
    #                                                init_vals,
    #                                                bias_correction,
    #                                                acceleration,
    #                                                std_error,
    #                                                empirical_influence,
    #                                                **fit_kwargs)
    lower_endpoint, upper_endpoint =\
        efron_endpoints_for_abc_confidence_interval(conf_percentage,
                                                    model_obj,
                                                    init_vals,
                                                    bias_correction,
                                                    acceleration,
                                                    std_error,
                                                    empirical_influence,
                                                    **fit_kwargs)
    # Combine the enpoints into a single ndarray.
    conf_intervals = combine_conf_endpoints(lower_endpoint, upper_endpoint)
    return conf_intervals
