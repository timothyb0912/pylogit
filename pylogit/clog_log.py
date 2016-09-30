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
import warnings
import numpy as np
from scipy.sparse import diags

import choice_calcs as cc
import base_multinomial_cm_v2 as base_mcm
from estimation import EstimationObj
from estimation import estimate

# Define the boundary values which are not to be exceeded ducing computation
max_comp_value = 1e300
min_comp_value = 1e-300

max_exp = 700
min_exp = -700

# Create a variable that will be printed if there is a non-fatal error
# in the Clog-log class construction
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


def create_calc_dh_dv(estimator):
    """
    Return the function that can be used in the various gradient and hessian
    calculations to calculate the derivative of the transformation with respect
    to the index.

    Parameters
    ----------
    estimator : an instance of the estimation.EstimationObj class.
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
    calc_dh_dv = partial(_cloglog_transform_deriv_v, output_array=dh_dv)
    return calc_dh_dv


def create_calc_dh_d_alpha(estimator):
    """
    Return the function that can be used in the various gradient and hessian
    calculations to calculate the derivative of the transformation with respect
    to the outside intercept parameters.

    Parameters
    ----------
    estimator : an instance of the estimation.EstimationObj class.
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
    calc_dh_d_alpha = partial(_cloglog_transform_deriv_alpha,
                              output_array=dh_d_alpha)

    return calc_dh_d_alpha


class ClogEstimator(EstimationObj):
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
        self.calc_dh_d_shape = _cloglog_transform_deriv_c

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
        # Note the clog-log model has one less outside intercept than number
        # of alternatives.
        num_alts = self.rows_to_alts.shape[1]
        num_index_coefs = self.design.shape[1]

        if self.intercept_ref_pos is not None:
            assumed_param_dimensions = num_index_coefs + num_alts - 1
        else:
            assumed_param_dimensions = num_index_coefs

        if init_values.shape[0] != assumed_param_dimensions:
            msg_1 = "The initial values are of the wrong dimension."
            msg_2 = "It should be of dimension {}"
            msg_3 = "But instead it has dimension {}"
            raise ValueError(msg_1 +
                             msg_2.format(assumed_param_dimensions) +
                             msg_3.format(init_values.shape[0]))

        return None


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
        rows_to_alts = mapping_res["rows_to_alts"]

        # Create init_vals from init_coefs and init_intercepts if those
        # arguments are passed to the function and init_vals is None.
        if init_vals is None and init_coefs is not None:
            ##########
            # Check the integrity of the parameter kwargs
            ##########
            num_alternatives = rows_to_alts.shape[1]
            try:
                assert init_coefs.shape[0] == self.design.shape[1]
            except AssertionError:
                msg = "init_coefs has length {} but should have length {}."
                raise ValueError(msg.format(init_coefs.shape,
                                            self.design.shape[1]))
            if init_intercepts is not None:
                if init_intercepts.shape[0] != (num_alternatives - 1):
                    msg = "init_intercepts has length {} "
                    msg_2 = "but it should have length {}"
                    raise ValueError(msg.format(init_intercepts.shape) +
                                     msg_2.format(num_alternatives - 1))

                init_vals = np.concatenate((init_intercepts,
                                            init_coefs), axis=0)
            else:
                init_vals = init_coefs
        elif init_vals is None and init_coefs is None:
            msg = "If init_vals is None, then users must pass init_coefs "
            raise ValueError(msg)

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
        clog_estimator = ClogEstimator(self,
                                       mapping_res,
                                       ridge,
                                       zero_vector,
                                       split_param_vec)
        # Set the derivative functions for estimation
        clog_estimator.set_derivatives()

        # Perform one final check on the length of the initial values
        clog_estimator.check_length_of_initial_values(init_vals)

        # Get the estimation results
        estimation_res = estimate(init_vals,
                                  clog_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res)

        # Store the estimation results
        self.store_fit_results(estimation_res)

        return None
