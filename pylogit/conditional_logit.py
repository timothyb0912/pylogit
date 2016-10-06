# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 07:19:49 2016

@name:      MultiNomial Logit
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating multinomial logit
            models (with the help of the "base_multinomial_cm.py" file).
            Differs from version one since it works with the shape, intercept,
            index coefficient partitioning of estimated parameters as opposed
            to the shape, index coefficient partitioning scheme of version 1.
"""
import warnings
import numpy as np
from scipy.sparse import diags

import choice_calcs as cc
import base_multinomial_cm_v2 as base_mcm
from estimation import LogitTypeEstimator
from estimation import estimate
from display_names import model_type_to_display_name

# Create a variable that will be printed if there is a non-fatal error
# in the MNL class construction
_msg_1 = "The Multinomial Logit Model has no shape parameters. "
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


def split_param_vec(beta, *args, **kwargs):
    """
    Parameters
    ----------
    beta:       1D numpy array. All elements should by ints, floats, or
                longs. Should have 1 element for each utility
                coefficient being estimated (i.e. num_features).

    Returns
    -------
    tuple.
        `(None, None, beta)`. This function is merely for compatibility with
        the other choice model files.
    """
    return None, None, beta


def _mnl_utility_transform(systematic_utilities, *args, **kwargs):
    """
    Parameters
    ----------
    systematic_utilities:   1D numpy array of the systematic utilities for each
                            each available alternative for each observation

    Returns
    -------
    `systematic_utilities[:, None]`
    """
    # Be sure to return a 2D array since other functions will be expecting this
    if len(systematic_utilities.shape) == 1:
        systematic_utilities = systematic_utilities[:, np.newaxis]

    return systematic_utilities


def _mnl_transform_deriv_c(*args, **kwargs):
    """
    Returns None.

    This is a place holder function since the MNL model has no shape
    parameters.
    """
    # This is a place holder function since the MNL model has no shape
    # parameters.
    return None


def _mnl_transform_deriv_alpha(*args, **kwargs):
    """
    Returns None.

    This is a place holder function since the MNL model has no intercept
    parameters outside of the index.
    """
    # This is a place holder function since the MNL model has no intercept
    # parameters outside the index.
    return None


class MNLEstimator(LogitTypeEstimator):
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
        # Pre-calculate the derivative of the transformation vector with
        # respect to the vector of systematic utilities
        dh_dv = diags(np.ones(self.design.shape[0]), 0, format='csr')
        # Create a function to calculate dh_dv which will return the
        # pre-calculated result when called
        calc_dh_dv = lambda *args: dh_dv

        self.calc_dh_dv = calc_dh_dv
        self.calc_dh_d_alpha = _mnl_transform_deriv_alpha
        self.calc_dh_d_shape = _mnl_transform_deriv_c

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
        # Calculate the expected number of index parameters
        num_index_coefs = self.design.shape[1]

        if init_values.shape[0] != num_index_coefs:
            msg_1 = "The initial values are of the wrong dimension."
            msg_2 = "It should be of dimension {}"
            msg_3 = "But instead it has dimension {}"
            raise ValueError(msg_1 +
                             msg_2.format(num_index_coefs) +
                             msg_3.format(init_values.shape[0]))

        return None


class MNL(base_mcm.MNDC_Model):
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

    """
    def __init__(self,
                 data,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 names=None,
                 *args, **kwargs):
        ##########
        # Print a helpful message for users who have included shape parameters
        # or shape names unneccessarily
        ##########
        for keyword in ["shape_names", "shape_ref_pos"]:
            if keyword in kwargs and kwargs[keyword] is not None:
                warnings.warn(_shape_ignore_msg)
                break

        if "intercept_ref_pos" in kwargs:
            if kwargs["intercept_ref_pos"] is not None:
                msg = "The MNL model should have all intercepts in the index."
                raise ValueError(msg)

        # Carry out the common instantiation process for all choice models
        super(MNL, self).__init__(data,
                                  alt_id_col,
                                  obs_id_col,
                                  choice_col,
                                  specification,
                                  names=names,
                                  model_type=model_type_to_display_name["MNL"])

        # Store the utility transform function
        self.utility_transform = _mnl_utility_transform

        return None

    def fit_mle(self,
                init_vals,
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
            should be one value for each utility coefficient being estimated.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
        method : str, optional.
            Should be a valid string that can be passed to
            scipy.optimize.minimize. Determines the optimization algorithm that
            is used for this problem. If 'em' is passed, a custom coded EM
            algorithm will be used. Default `== 'newton-cg'`.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next that is needed to determine
            convergence. Default `== 1e-06`.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
        ridge : int, float, long, or None, optional.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. Default `== None`.

        Returns
        -------
        None. Estimation results are saved to the model instance.
        """
        # Check integrity of passed arguments
        kwargs_to_be_ignored = ["init_shapes", "init_intercepts", "init_coefs"]
        if any([x in kwargs for x in kwargs_to_be_ignored]):
            msg = "MNL model does not use of any of the following kwargs:\n{}"
            msg_2 = "Remove such kwargs and pass a single init_vals argument"
            raise ValueError(msg.format(kwargs_to_be_ignored) + msg_2)

        if ridge is not None:
            warnings.warn(_ridge_warning_msg)

        # Store the optimization method
        self.optimization_method = method

        # Store the ridge parameter
        self.ridge_param = ridge

        # Construct the mappings from alternatives to observations and from
        # chosen alternatives to observations
        mapping_res = self.get_mappings_for_fit()

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
        mnl_estimator = MNLEstimator(self,
                                     mapping_res,
                                     ridge,
                                     zero_vector,
                                     split_param_vec)
        # Set the derivative functions for estimation
        mnl_estimator.set_derivatives()

        # Perform one final check on the length of the initial values
        mnl_estimator.check_length_of_initial_values(init_vals)

        # Get the estimation results
        estimation_res = estimate(init_vals,
                                  mnl_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res)

        # Store the estimation results
        self.store_fit_results(estimation_res)

        return None
