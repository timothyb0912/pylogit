"""
@author:    Timothy Brathwaite
@name:      Bootstrap ABC
@summary:   This module provides functions to calculate the approximate
            bootstrap confidence (ABC) intervals.
"""
import sys
import numpy as np
from scipy.stats import norm

from .bootstrap_utils import check_conf_percentage_validity
from .bootstrap_utils import get_alpha_from_conf_percentage
from .bootstrap_utils import combine_conf_endpoints


# Create a value to be used as 'a very small number' when performing the finite
# difference calculations needed to estimate the empirical influence function.
EPSILON = sys.float_info.epsilon


def create_long_form_weights(model_obj, wide_weights, rows_to_obs=None):
    if rows_to_obs is None:
        rows_to_obs = model_obj.get_mappings_for_fit()['rows_to_obs']
    wide_weights_2d =\
        wide_weights if wide_weights.ndim == 2 else wide_weights[:, None]
    long_weights = rows_to_obs.dot(wide_weights_2d)
    if isinstance(long_weights, np.matrixlib.defmatrix.matrix):
        long_weights = np.asarray(long_weights)
    if wide_weights.ndim == 1:
        long_weights = long_weights.sum(axis=1)
    return long_weights


def calc_finite_diff_terms_for_abc(model_obj,
                                   mle_params,
                                   init_vals,
                                   epsilon,
                                   **fit_kwargs):
    """
    Calculates the finite difference terms needed to compute the approximate
    boostrap confidence (ABC) intervals.

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
        was obtained. Will be passed directly to `model_obj.fit_mle`. Should
        NOT contain the key 'weights'.

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
    """
    # Determine the number of observations in this dataset.
    num_obs = model_obj.data[model_obj.obs_id_col].unique().size
    # Determine the initial weights per observation.
    init_weights_wide = np.ones(num_obs, dtype=float) / num_obs
    # Create the 'wide' identity matrix for this dataset
    wide_identity = np.eye(num_obs)
    # Create weights for the elements of the second order influence array.
    wide_weights_plus =\
        (1 - epsilon) * init_weights_wide[None, :] + epsilon * wide_identity
    wide_weights_minus =\
        (1 - epsilon) * init_weights_wide[None, :] - epsilon * wide_identity
    # Initialize the second order influence array
    term_plus = np.empty((num_obs, init_vals.shape[0]), dtype=float)
    term_minus = np.empty((num_obs, init_vals.shape[0]), dtype=float)
    # Get the rows_to_obs mapping matrix for this model.
    rows_to_obs = model_obj.get_mappings_for_fit()['rows_to_obs']
    # Populate the second order influence array
    for obs in xrange(num_obs):
        # Get the needed long format weights for this row
        long_weights_plus = create_long_form_weights(model_obj,
                                                     wide_weights_plus,
                                                     rows_to_obs=rows_to_obs)
        long_weights_minus = create_long_form_weights(model_obj,
                                                      wide_weights_minus,
                                                      rows_to_obs=rows_to_obs)
        # Get the needed influence estimates.
        term_plus[obs] = model_obj.fit_mle(init_vals,
                                           weights=long_weights_plus,
                                           **fit_kwargs)['x']
        term_minus[obs] = model_obj.fit_mle(init_vals,
                                            weights=long_weights_minus,
                                            **fit_kwargs)['x']
    return term_plus, term_minus


def calc_empirical_influence_abc(term_plus,
                                 term_minus,
                                 epsilon):
    """
    Calculates the empirical influence array needed to compute the approximate
    boostrap confidence (ABC) intervals.

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
    """
    # Calculate the denominator of each row of the second order influence array
    denominator = 2 * epsilon
    # Calculate the empirical influence array.
    empirical_influence = (term_plus - term_minus) / denominator
    return empirical_influence


def calc_2nd_order_influence_abc(mle_params,
                                 term_plus,
                                 term_minus,
                                 epsilon):
    """
    Calculates the 2nd order empirical influence array needed to compute the
    approximate boostrap confidence (ABC) intervals.

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
    """
    # Calculate the denominator of each row of the second order influence array
    denominator = epsilon**2
    # Initialize the second term used to calculate each row in the second order
    # influence array
    term_2 = (2 * mle_params)[None, :]
    # Calculate the second order empirical influence array.
    second_order_influence = (term_plus - term_2 + term_minus) / denominator
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
        was obtained. Will be passed directly to `model_obj.fit_mle`. Should
        NOT contain the key 'weights'.

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
    influence_squared = empirical_influence**2
    num_obs = empirical_influence.shape[0]
    constant = num_obs**2
    std_error = (influence_squared.sum(axis=0) / constant)**0.5
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
        was obtained. Will be passed directly to `model_obj.fit_mle`. Should
        NOT contain the key 'weights'.

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
    # Determine the initial weights per observation.
    init_weights_wide = np.ones(num_obs, dtype=float) / num_obs
    # Get the rows_to_obs mapping matrix for this model.
    rows_to_obs = model_object.get_mappings_for_fit()['rows_to_obs']
    # Get a 'long-format' array of the weights per observation
    init_weights_long = create_long_form_weights(model_object,
                                                 init_weights_wide,
                                                 rows_to_obs=rows_to_obs)
    # Calculate the standardized version of the empirical influence values
    # assuming mean zero. This is the mean of empirical influence values
    # (sum over n) divided by the standard error of the mean of the empiricial
    # influence values, i.e. the square root of [n^2 * variance] = n * std_err
    # where std_err is the standard error of the sampling distribution of the
    # influence values.
    standardized_influence =\
        empirical_influence / (num_obs**2 * std_error[None, :])
    standardized_influence_long =\
        create_long_form_weights(model_object,
                                 standardized_influence,
                                 rows_to_obs=rows_to_obs)
    # Calculate the weights for the various terms of the quadratic_coef
    # Note that these arrays have shape (num_long_rows, num_params)
    term_1_weights = ((1 - epsilon) * init_weights_long[:, None] +
                      epsilon * standardized_influence_long)
    term_3_weights = ((1 - epsilon) * init_weights_long[:, None] -
                      epsilon * standardized_influence_long)
    # Initialize the terms for the quadratic coefficients.
    expected_term_shape = (init_vals.shape[0], init_vals.shape[0])
    term_1_array = np.empty(expected_term_shape, dtype=float)
    term_3_array = np.empty(expected_term_shape, dtype=float)
    # Calculate the various terms of the quadratic_coef
    for param_id in xrange(expected_term_shape[0]):
        term_1_array[param_id] =\
            model_object.fit_mle(init_vals,
                                 weights=term_1_weights[:, param_id],
                                 **fit_kwargs)['x']
        term_3_array[param_id] =\
            model_object.fit_mle(init_vals,
                                 weights=term_3_weights[:, param_id],
                                 **fit_kwargs)['x']
    # Extract the desired terms from their 2d arrays
    term_1 = np.diag(term_1_array)
    term_3 = np.diag(term_3_array)
    term_2 = 2 * mle_params
    # Calculate the quadratic coefficient
    numerator = term_1 - term_2 + term_3
    denominator = epsilon**2
    quadratic_coef = numerator / denominator
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
        CRC press, 1994. Section 22.6, Equation 22.40.

    Note
    ----
    This function implements an analytic, simplified version of the first line
    of Equation 22.40 of Efron and Tibshirani (1994). Note that the formula on
    the second line of Equation 22.40 appears incorrect, and this line is not
    used in the current function.
    """
    constant = np.log(np.pi / 2.0)
    bias_correction = (constant + acceleration**2 + total_curvature**2)**0.5
    return bias_correction


def calc_endpoint_from_percentile_abc(model_obj,
                                      init_vals,
                                      percentile,
                                      bias_correction,
                                      acceleration,
                                      std_error,
                                      empirical_influence,
                                      **fit_kwargs):
    # Get the bias corrected standard normal value for the relevant percentile.
    biased_corrected_z = bias_correction + norm.ppf(percentile)
    # Calculate the multiplier for the least favorable direction
    # (i.e. the empirical influence function).
    lam = biased_corrected_z / (1 - acceleration * biased_corrected_z)**2
    multiplier = lam / std_error
    # Calculate the initial weights
    num_obs = empirical_influence.shape[0]
    init_weights_wide = np.ones(num_obs, dtype=float) / num_obs
    # Get the necessary weight adjustment term for calculating the endpoint.
    weight_adjustment_wide = (multiplier[None, :] * empirical_influence)
    wide_weights_all_params = init_weights_wide + weight_adjustment_wide
    # Get a long format version of the weights needed to compute the endpoints
    long_weights_all_params =\
        create_long_form_weights(model_obj, wide_weights_all_params)
    # Initialize the array to store the desired enpoints
    num_params = init_vals.shape[0]
    endpoint = np.empty(num_params, dtype=float)
    # Populate the endpoint array
    for param_id in xrange(num_params):
        current_weights = long_weights_all_params[:, param_id]
        current_estimate = model_obj.fit_mle(init_vals,
                                             weights=current_weights,
                                             **fit_kwargs)['x']
        endpoint[param_id] = current_estimate[param_id]
    return endpoint


def calc_endpoints_for_abc_confidence_endpoint(conf_percentage,
                                               model_obj,
                                               init_vals,
                                               bias_correction,
                                               acceleration,
                                               std_error,
                                               empirical_influence,
                                               **fit_kwargs):
    # Calculate the percentiles for the lower and upper endpoints
    alpha_percent = get_alpha_from_conf_percentage(conf_percentage)
    lower_percentile = alpha_percent / 2.0
    upper_percentile = 100 - lower_percentile
    # Calculate the lower endpoint
    lower_endpoint = calc_endpoint_from_percentile_abc(model_obj,
                                                       init_vals,
                                                       lower_percentile,
                                                       bias_correction,
                                                       acceleration,
                                                       std_error,
                                                       empirical_influence,
                                                       **fit_kwargs)
    # Calculate the upper endpoint
    upper_endpoint = calc_endpoint_from_percentile_abc(model_obj,
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
                      epsilon=EPSILON,
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
        was obtained. Will be passed directly to `model_obj.fit_mle`. Should
        NOT contain the key 'weights'.

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
    # Calculate the quadratic coefficient.
    quadratic_coef = calc_quadratic_coef_abc(model_obj,
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
    lower_endpoint, upper_endpoint =\
        calc_endpoints_for_abc_confidence_endpoint(conf_percentage,
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
