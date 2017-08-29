"""
@author:    Timothy Brathwaite
@name:      Bootstrap Calculations
@summary:   This module provides functions to calculate the bootstrap
            confidence intervals using the 'percentile' and
            'bias-corrected and accelerated' methods.
"""
import numpy as np
from scipy.stats import norm

from .bootstrap_utils import check_conf_percentage_validity
from .bootstrap_utils import ensure_samples_is_2d_ndarray
from .bootstrap_utils import get_alpha_from_conf_percentage
from .bootstrap_utils import combine_conf_endpoints

# Create a value to be used to avoid numeric underflow.
MIN_COMP_VALUE = 1e-16


def calc_percentile_interval(bootstrap_samples, conf_percentage):
    """
    Calculate bootstrap confidence intervals based on raw percentiles of the
    bootstrap distribution of samples.

    Parameters
    ----------
    bootstrap_samples : 2D ndarray.
        Each row should correspond to a different bootstrap parameter sample.
        Each column should correspond to an element of the parameter vector
        being estimated.
    conf_percentage : scalar in the interval (0.0, 100.0).
        Denotes the confidence-level of the returned confidence interval. For
        instance, to calculate a 95% confidence interval, pass `95`.

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
        CRC press, 1994. Section 12.5 and Section 13.3.

    Notes
    -----
    This function differs slightly from the actual percentile bootstrap
    procedure described in Efron and Tibshirani (1994). To ensure that the
    returned endpoints of one's bootstrap confidence intervals are actual
    values that were observed in the bootstrap distribution, both the procedure
    of Efron and Tibshirani and this function make more conservative confidence
    intervals. However, this function uses a simpler (and in some cases less
    conservative) correction than that of Efron and Tibshirani.
    """
    # Check validity of arguments
    check_conf_percentage_validity(conf_percentage)
    ensure_samples_is_2d_ndarray(bootstrap_samples)
    # Get the alpha * 100% value
    alpha = get_alpha_from_conf_percentage(conf_percentage)
    # Get the lower and upper percentiles that demarcate the desired interval.
    lower_percent = alpha / 2.0
    upper_percent = 100.0 - lower_percent
    # Calculate the lower and upper endpoints of the confidence intervals.
    # Note that the particular choices of interpolation methods are made in
    # order to produce conservatively wide confidence intervals and ensure that
    # all returned endpoints in the confidence intervals are actually observed
    # in the bootstrap distribution. This is in accordance with the spirit of
    # Efron and Tibshirani (1994).
    lower_endpoint = np.percentile(bootstrap_samples,
                                   lower_percent,
                                   interpolation='lower',
                                   axis=0)
    upper_endpoint = np.percentile(bootstrap_samples,
                                   upper_percent,
                                   interpolation='upper',
                                   axis=0)
    # Combine the enpoints into a single ndarray.
    conf_intervals = combine_conf_endpoints(lower_endpoint, upper_endpoint)
    return conf_intervals


def calc_bias_correction_bca(bootstrap_samples, mle_estimate):
    numerator = (bootstrap_samples < mle_estimate[None, :]).sum(axis=0)
    denominator = float(bootstrap_samples.shape[0])
    bias_correction = norm.ppf(numerator / denominator)
    return bias_correction


def calc_acceleration_bca(jackknife_samples):
    # Get the mean of the bootstrapped statistics.
    jackknife_mean = jackknife_samples.mean(axis=0)[None, :]
    # Calculate the differences between the mean of the bootstrapped statistics
    differences = jackknife_mean - jackknife_samples
    numerator = (differences**3).sum(axis=0)
    denominator = 6 * ((differences**2).sum(axis=0))**1.5
    # guard against division by zero. Note that this guard shouldn't distort
    # the computational results since the numerator should be zero whenever the
    # denominator is zero.
    zero_denom = np.where(denominator == 0)
    denominator[zero_denom] = MIN_COMP_VALUE
    # Compute the acceleration.
    acceleration = numerator / denominator
    return acceleration


def calc_lower_bca_percentile(alpha_percent, bias_correction, acceleration):
    z_lower = norm.ppf(alpha_percent / (100.0 * 2))
    numerator = bias_correction + z_lower
    denominator = 1 - acceleration * numerator
    lower_percentile =\
        norm.cdf(bias_correction + numerator / denominator) * 100
    return lower_percentile


def calc_upper_bca_percentile(alpha_percent, bias_correction, acceleration):
    z_upper = norm.ppf(1 - alpha_percent / (100.0 * 2))
    numerator = bias_correction + z_upper
    denominator = 1 - acceleration * numerator
    upper_percentile =\
        norm.cdf(bias_correction + numerator / denominator) * 100
    return upper_percentile


def calc_bca_interval(bootstrap_samples,
                      jackknife_samples,
                      mle_params,
                      conf_percentage):
    """
    Calculate 'bias-corrected and accelerated' bootstrap confidence intervals.

    Parameters
    ----------
    bootstrap_samples : 2D ndarray.
        Each row should correspond to a different bootstrap parameter sample.
        Each column should correspond to an element of the parameter vector
        being estimated.
    jackknife_samples : 2D ndarray.
        Each row should correspond to a different jackknife parameter sample,
        formed by deleting a particular observation and then re-estimating the
        desired model. Each column should correspond to an element of the
        parameter vector being estimated.
    mle_params : 1D ndarray.
        The original dataset's maximum likelihood point estimate. Should have
        the same number of elements as `samples.shape[1]`.
    conf_percentage : scalar in the interval (0.0, 100.0).
        Denotes the confidence-level of the returned confidence interval. For
        instance, to calculate a 95% confidence interval, pass `95`.

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
        CRC press, 1994. Section 14.3.
    DiCiccio, Thomas J., and Bradley Efron. "Bootstrap confidence intervals."
        Statistical science (1996): 189-212.
    """
    # Check validity of arguments
    check_conf_percentage_validity(conf_percentage)
    ensure_samples_is_2d_ndarray(bootstrap_samples)
    ensure_samples_is_2d_ndarray(jackknife_samples, name='jackknife')
    # Calculate the alpha * 100% value
    alpha_percent = get_alpha_from_conf_percentage(conf_percentage)
    # Estimate the bias correction for the bootstrap samples
    bias_correction = calc_bias_correction_bca(bootstrap_samples, mle_params)
    # Estimate the acceleration
    acceleration = calc_acceleration_bca(jackknife_samples)
    # Get the lower and upper percent value for the raw bootstrap samples.
    lower_percents =\
        calc_lower_bca_percentile(alpha_percent, bias_correction, acceleration)
    upper_percents =\
        calc_upper_bca_percentile(alpha_percent, bias_correction, acceleration)
    # Get the lower and upper endpoints for the desired confidence intervals.
    lower_endpoints = np.diag(np.percentile(bootstrap_samples,
                                            lower_percents,
                                            interpolation='lower',
                                            axis=0))
    upper_endpoints = np.diag(np.percentile(bootstrap_samples,
                                            upper_percents,
                                            interpolation='upper',
                                            axis=0))
    # Combine the enpoints into a single ndarray.
    conf_intervals = combine_conf_endpoints(lower_endpoints, upper_endpoints)
    return conf_intervals
