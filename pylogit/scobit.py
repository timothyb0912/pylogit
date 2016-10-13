# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:06:41 2016

@name:      MultiNomial Scobit Model
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating multinomial scobit
            models (with the help of the "base_multinomial_cm.py" file).
            Differs from version one since it works with the shape, intercept,
            index coefficient partitioning of estimated parameters as opposed
            to the shape, index coefficient partitioning scheme of version 1.
"""

from functools import partial
import warnings
import numpy as np
from scipy.sparse import diags

import base_multinomial_cm_v2 as base_mcm
from estimation import LogitTypeEstimator
from estimation import estimate
from display_names import model_type_to_display_name as display_name_dict

# Define the boundary values which are not to be exceeded during computation
max_comp_value = 1e300
min_comp_value = 1e-300

max_exp = 700
min_exp = -700

# Create a warning string that will be issued if ridge regression is performed.
_msg = "NOTE: An L2-penalized regression is being performed. The "
_msg_2 = "reported standard errors and robust standard errors "
_msg_3 = "***WILL BE INCORRECT***."
_ridge_warning_msg = _msg + _msg_2 + _msg_3

# Create a warning string that will be issued if "shape_ref_pos" is passed to
# the MNSL constructor as a kwarg
_msg_4 = "The Multinomial Scobit model estimates all shape parameters"
_msg_5 = ", so shape_ref_pos will be ignored if passed."
_shape_ref_msg = _msg_4 + _msg_5


def split_param_vec(param_vec, rows_to_alts, design, return_all_types=False):
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
    return_all_types : bool, optional.
        Determines whether or not a tuple of 4 elements will be returned (with
        one element for the nest, shape, intercept, and index parameters for
        this model). If False, a tuple of 3 elements will be returned, as
        described below.

    Returns
    -------
    `(shapes, intercepts, betas)` : tuple of 1D ndarrays.
        The first element will be an array of the shape parameters for this
        model. The second element will either be an array of the "outside"
        intercept parameters for this model or None. The third element will be
        an array of the index coefficients for this model.

    Note
    ----
    If `return_all_types == True` then the function will return a tuple of four
    objects. In order, these objects will either be None or the arrays
    representing the arrays corresponding to the nest, shape, intercept, and
    index parameters.
    """
    # Figure out how many possible alternatives exist in the dataset
    num_shapes = rows_to_alts.shape[1]
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

    if return_all_types:
        return None, shapes, intercepts, betas
    else:
        return shapes, intercepts, betas


def _scobit_utility_transform(systematic_utilities,
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
        If an array, each element should be an int, float, or long. There
        should be one value per shape parameter of the model being used.
    intercept_params : None or 1D ndarray.
        If an array, each element should be an int, float, or long. If J is the
        total number of possible alternatives for the dataset being modeled,
        there should be J-1 elements in the array.
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
        elements should be ints, floats, or longs.
    """
    # Figure out what indices are to be filled in
    if intercept_ref_pos is not None and intercept_params is not None:
        needed_idxs = range(intercept_params.shape[0] + 1)
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
    else:
        # Create a full set of intercept parameters including the intercept
        # constrained to zero
        all_intercepts = np.zeros(rows_to_alts.shape[1])

    # Figure out what intercept values correspond to each row of the
    # systematic utilities
    long_intercepts = rows_to_alts.dot(all_intercepts)

    # Convert the shape parameters back into their 'natural parametrization'
    natural_shapes = np.exp(shape_params)
    natural_shapes[np.isposinf(natural_shapes)] = max_comp_value
    # Figure out what shape values correspond to each row of the
    # systematic utilities
    long_natural_shapes = rows_to_alts.dot(natural_shapes)

    # Calculate the data dependent part of the transformation
    # Also, along the way, guard against numeric underflow or overflow
    exp_neg_v = np.exp(-1 * systematic_utilities)
    exp_neg_v[np.isposinf(exp_neg_v)] = max_comp_value

    powered_term = np.power(1 + exp_neg_v, long_natural_shapes)
    powered_term[np.isposinf(powered_term)] = max_comp_value

    term_2 = np.log(powered_term - 1)
    # Guard against overvlow
    too_big_idx = np.isposinf(powered_term)
    term_2[too_big_idx] = (-1 * long_natural_shapes[too_big_idx] *
                           systematic_utilities[too_big_idx])

    transformations = long_intercepts - term_2
    # Guard against overflow
    transformations[np.isposinf(transformations)] = max_comp_value
    transformations[np.isneginf(transformations)] = -1 * max_comp_value

    # Be sure to return a 2D array since other functions will be expecting that
    if len(transformations.shape) == 1:
        transformations = transformations[:, np.newaxis]

    return transformations


def _scobit_transform_deriv_v(systematic_utilities,
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
    # Note  the np.exp is  needed because the raw curvature params are the log
    # of the 'natural' curvature params. This is to ensure the natural shape
    # params are always positive
    curve_shapes = np.exp(shape_params)
    curve_shapes[np.isposinf(curve_shapes)] = max_comp_value
    long_curve_shapes = rows_to_alts.dot(curve_shapes)

    # Generate the needed terms for the derivative of the transformation with
    # respect to the systematic utility and guard against underflow or overflow
    exp_neg_v = np.exp(-1 * systematic_utilities)

    powered_term = np.power(1 + exp_neg_v, long_curve_shapes)

    small_powered_term = np.power(1 + exp_neg_v, long_curve_shapes - 1)

    derivs = (long_curve_shapes *
              exp_neg_v *
              small_powered_term /
              (powered_term - 1))
    # Use L'Hopitals rule to deal with overflow from v --> -inf
    # From plots, the assignment below may also correctly handle cases where we
    # have overflow from moderate v (say |v| <= 10) and large shape parameters.
    too_big_idx = (np.isposinf(derivs) +
                   np.isposinf(exp_neg_v) +
                   np.isposinf(powered_term) +
                   np.isposinf(small_powered_term)).astype(bool)
    derivs[too_big_idx] = long_curve_shapes[too_big_idx]
    # Use L'Hopitals rule to deal with underflow from v --> inf
    too_small_idx = np.where((exp_neg_v == 0) | (powered_term - 1 == 0))
    derivs[too_small_idx] = 1.0

    # Assign the calculated derivatives to the output array
    output_array.data = derivs
    assert output_array.shape == (systematic_utilities.shape[0],
                                  systematic_utilities.shape[0])

    # Return the matrix of dh_dv. Note the off-diagonal entries are zero
    # because each transformation only depends on its value of v and no others
    return output_array


def _scobit_transform_deriv_shape(systematic_utilities,
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
        The array should have shape `(systematic_utilities.shape[0],
        shape_params.shape[0])`. It's data is to be replaced with the correct
        derivatives of the transformation vector with respect to the vector of
        shape parameters. This argument is NOT optional.

    Returns
    -------
    output_array : 2D scipy sparse array.
        The shape of the returned array is `(systematic_utilities.shape[0],
        shape_params.shape[0])`. The returned array specifies the derivative of
        the transformed utilities with respect to the shape parameters. All
        elements are ints, floats, or longs.
    """
    # Note the np.exp is needed because the raw curvature params are the log
    # of the 'natural' curvature params. This is to ensure the natural shape
    # params are always positive.
    curve_shapes = np.exp(shape_params)
    curve_shapes[np.isposinf(curve_shapes)] = max_comp_value
    long_curve_shapes = rows_to_alts.dot(curve_shapes)

    # Generate the needed terms for the derivative of the transformation with
    # respect to the systematic utility and guard against underflow or overflow
    exp_neg_v = np.exp(-1 * systematic_utilities)

    powered_term = np.power(1 + exp_neg_v, long_curve_shapes)

    # Note the "* long_curve_shapes" accounts for the fact that the derivative
    # of the natural shape params (i.e. exp(shape_param)) with respect to the
    # shape param is simply exp(shape_param) = natural shape param. The
    # multipication comes from the chain rule.
    curve_derivs = (-1 * np.log1p(exp_neg_v) *
                    powered_term / (powered_term - 1)) * long_curve_shapes
    # Note that all of the following overflow and underflow guards are based
    # on L'Hopital's rule.
    # Guard against underflow from v --> inf. Note we end up with '-1' after
    # accounting for the jacobian.
    too_big_idx = np.where((powered_term - 1) == 0)
    curve_derivs[too_big_idx] = -1
    # Guard against overflow from v --> -inf.
    too_small_idx = np.isposinf(exp_neg_v)
    curve_derivs[too_small_idx] = max_comp_value
    # Guard against overflow from moderate v but shape --> inf.
    shape_too_big_idx = np.where((np.abs(systematic_utilities) <= 10) &
                                 np.isposinf(powered_term))
    curve_derivs[shape_too_big_idx] = -1 * np.log1p(exp_neg_v)

    # Assign the calculated derivatives to the output array
    output_array.data = curve_derivs

    return output_array


def _scobit_transform_deriv_alpha(systematic_utilities,
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
    calc_dh_dv = partial(_scobit_transform_deriv_v, output_array=dh_dv)
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
    dh_d_shape = estimator.rows_to_alts.copy()
    # Create a function that will take in the pre-formed matrix, replace its
    # data in-place with the new data, and return the correct dh_dshape on each
    # iteration of the minimizer
    calc_dh_d_shape = partial(_scobit_transform_deriv_shape,
                              output_array=dh_d_shape)
    return calc_dh_d_shape


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
    calc_dh_d_alpha = partial(_scobit_transform_deriv_alpha,
                              output_array=dh_d_alpha)

    return calc_dh_d_alpha


class ScobitEstimator(LogitTypeEstimator):
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
        # Note the scobit model has one shape parameter per alternative.
        num_alts = self.rows_to_alts.shape[1]
        num_index_coefs = self.design.shape[1]

        if self.intercept_ref_pos is not None:
            assumed_param_dimensions = num_index_coefs + 2 * num_alts - 1
        else:
            assumed_param_dimensions = num_index_coefs + num_alts

        if init_values.shape[0] != assumed_param_dimensions:
            msg_1 = "The initial values are of the wrong dimension."
            msg_2 = "It should be of dimension {}"
            msg_3 = "But instead it has dimension {}"
            raise ValueError(msg_1 +
                             msg_2.format(assumed_param_dimensions) +
                             msg_3.format(init_values.shape[0]))

        return None


class MNSL(base_mcm.MNDC_Model):
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
                 names=None,
                 intercept_names=None,
                 shape_names=None,
                 **kwargs):
        ##########
        # Print a helpful message if shape_ref_pos has been included
        ##########
        if "shape_ref_pos" in kwargs and kwargs["shape_ref_pos"] is not None:
            warnings.warn(_shape_ref_msg)

        ##########
        # Carry out the common instantiation process for all choice models
        ##########
        super(MNSL, self).__init__(data,
                                   alt_id_col,
                                   obs_id_col,
                                   choice_col,
                                   specification,
                                   intercept_ref_pos=intercept_ref_pos,
                                   names=names,
                                   intercept_names=intercept_names,
                                   shape_names=shape_names,
                                   model_type=display_name_dict["Scobit"])

        ##########
        # Store the utility transform function
        ##########
        self.utility_transform = partial(_scobit_utility_transform,
                                         intercept_ref_pos=intercept_ref_pos)

        return None

    def fit_mle(self,
                init_vals,
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
            ints, floats, or longs. There should be one parameter per possible
            alternative id in the dataset. This keyword argument will be
            ignored if `init_vals` is not None. Default == None.
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
                assert init_shapes.shape[0] == num_alternatives
            except AssertionError:
                msg = "init_shapes is of length {} but should be of length {}"
                raise ValueError(msg.format(init_shapes.shape,
                                            num_alternatives))

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
            #     print(msg + msg_2 )
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
        scobit_estimator = ScobitEstimator(self,
                                           mapping_res,
                                           ridge,
                                           zero_vector,
                                           split_param_vec)
        # Set the derivative functions for estimation
        scobit_estimator.set_derivatives()

        # Perform one final check on the length of the initial values
        scobit_estimator.check_length_of_initial_values(init_vals)

        # Get the estimation results
        estimation_res = estimate(init_vals,
                                  scobit_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res)

        # Store the estimation results
        self.store_fit_results(estimation_res)

        return None
