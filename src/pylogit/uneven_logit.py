# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 08:02:45 2016

@name:      MultiNomial Uneven Logit 3
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating multinomial asymmetric
            logit models (with the help of the "base_multinomial_cm.py" file).

            Version 3 differs from version two since it reparametrizes the
            shape parameters as ln(original shape parameters) to allow
            unconstrained optimization.

            Version 2 differed from version 1 by using with the shape,
            intercept, and index coefficient partitioning for estimated
            parameters as opposed to the shape, index coefficient partitioning
            scheme of version 1.
"""
from __future__ import absolute_import

from functools import partial
import warnings
import numpy as np
from scipy.sparse import diags

from . import base_multinomial_cm_v2 as base_mcm
from .estimation import LogitTypeEstimator
from .estimation import estimate
from .display_names import model_type_to_display_name as display_name_dict

try:
    # in Python 3 range returns an iterator instead of list
    # to maintain backwards compatibility use "old" version of range
    from past.builtins import range
except ImportError:
    pass


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
_msg_4 = "The Multinomial Uneven Logit model estimates all shape parameters"
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


def _uneven_utility_transform(systematic_utilities,
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
    transformed_utilities : 2D ndarray.
        Should have shape `(systematic_utilities.shape[0], 1)`. The returned
        array contains the transformed utility values for this model. All
        elements will be ints, longs, or floats.
    """
    # Convert the shape parameters back into their 'natural parametrization'
    natural_shapes = np.exp(shape_params)
    natural_shapes[np.isposinf(natural_shapes)] = max_comp_value
    # Figure out what shape values correspond to each row of the
    # systematic utilities
    long_natural_shapes = rows_to_alts.dot(natural_shapes)

    # Get the exponentiated neative utilities
    exp_neg_utilities = np.exp(-1 * systematic_utilities)

    # Get the log of 1 + exponentiated negative utilities
    log_1_plus_exp_neg_utilitiles = np.log1p(exp_neg_utilities)

    # Guard against overflow. Underflow not a problem since we add one to a
    # near zero number and log of one will evaluate to zero
    inf_idx = np.isinf(log_1_plus_exp_neg_utilitiles)
    log_1_plus_exp_neg_utilitiles[inf_idx] = -1 * systematic_utilities[inf_idx]

    # Get the exponentiated (negative utilities times the shape parameter)
    exp_neg_shape_utilities = np.exp(-1 *
                                     long_natural_shapes *
                                     systematic_utilities)

    # Get the log of 1 + exponentiated (negative utiltiies times the shape)
    log_1_plus_exp_neg_shape_utilities = np.log1p(exp_neg_shape_utilities)

    ##########
    # Guard against overflow
    ##########
    # Check for any values which have gone off to positive infinity
    inf_idx = np.isinf(log_1_plus_exp_neg_shape_utilities)
    # Replace those values with an approximation of the true values by ignoring
    # the "1." The idea is that 1 + infinity ~ infinity so the effect of the +1
    # on the log is minimal.
    if np.any(inf_idx):
        log_1_plus_exp_neg_shape_utilities[inf_idx] =\
              -1 * long_natural_shapes[inf_idx] * systematic_utilities[inf_idx]

    # Calculate the transformed utility values
    transformed_utilities = (systematic_utilities +
                             log_1_plus_exp_neg_utilitiles -
                             log_1_plus_exp_neg_shape_utilities)
    # Perform a final guard against numbers that are too large to deal with
    transformed_utilities[np.isposinf(transformed_utilities)] = max_comp_value
    transformed_utilities[np.isneginf(transformed_utilities)] = -max_comp_value
    transformed_utilities[np.isneginf(systematic_utilities)] = -max_comp_value

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
        transformed_utilities += rows_to_alts.dot(all_intercepts)

    # Be sure to return a 2D array since other functions will be expecting that
    if len(transformed_utilities.shape) == 1:
        transformed_utilities = transformed_utilities[:, np.newaxis]

    return transformed_utilities


def _uneven_transform_deriv_v(systematic_utilities,
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
    # Convert the shape parameters back into their 'natural parametrization'
    natural_shapes = np.exp(shape_params)
    natural_shapes[np.isposinf(natural_shapes)] = max_comp_value
    # Figure out what shape values correspond to each row of the
    # systematic utilities
    long_shapes = rows_to_alts.dot(natural_shapes)

    # Get the exponentiated neative utilities
    exp_neg_utilities = np.exp(-1 * systematic_utilities)

    # Get the exponentiated (utilities times the shape parameter)
    exp_shape_utilities = np.exp(long_shapes * systematic_utilities)

    # Calculate the derivative of h_ij with respect to v_ij
    # Note that the derivative of h_ij with respect to any other systematic
    # utility is zero.
    derivs = (1.0 / (1.0 + exp_neg_utilities) +
              long_shapes / (1.0 + exp_shape_utilities))

    output_array.data = derivs

    # Return the matrix of dh_dv. Note the off-diagonal entries are zero
    # because each transformation only depends on its value of v and no others
    return output_array


def _uneven_transform_deriv_shape(systematic_utilities,
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
    # Convert the shape parameters back into their 'natural parametrization'
    natural_shapes = np.exp(shape_params)
    natural_shapes[np.isposinf(natural_shapes)] = max_comp_value
    # Figure out what shape values correspond to each row of the
    # systematic utilities
    long_shapes = rows_to_alts.dot(natural_shapes)

    # Get the exponentiated (utilities times the shape parameter)
    exp_shape_utilities = np.exp(long_shapes * systematic_utilities)

    # Calculate the derivative of h_ij with respect to shape_j.
    derivs = (systematic_utilities / (1.0 + exp_shape_utilities))
    # Guard against overflow. Only for cases of systematic_utilities becomming
    # huge. It is unlikely this safeguard will be needed.
    derivs[np.isposinf(systematic_utilities)] = 0
    # Guard against underflow from v --> -inf.
    huge_index = np.isneginf(systematic_utilities)
    derivs[huge_index] = -max_comp_value

    # Return the matrix of dh_dshapes. Note the matrix should be of dimension
    # (systematic_utilities.shape[0], shape_params.shape[0])
    # Note that the "* long_shapes" accounts for the fact that the derivative
    # of the natural shape parameters with resepect to the actual shape
    # parameters being estimated is simply
    # exp(actual shape parameters) = natural shape parameters. The
    # multiplication comes from the chain rule.
    output_array.data = derivs * long_shapes
    return output_array


def _uneven_transform_deriv_alpha(systematic_utilities,
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
    calc_dh_dv = partial(_uneven_transform_deriv_v, output_array=dh_dv)
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
    calc_dh_d_shape = partial(_uneven_transform_deriv_shape,
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
    calc_dh_d_alpha = partial(_uneven_transform_deriv_alpha,
                              output_array=dh_d_alpha)

    return calc_dh_d_alpha


class UnevenEstimator(LogitTypeEstimator):
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
            zero.
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
    constrained_pos : list or None, optional.
        Denotes the positions of the array of estimated parameters that are
        not to change from their initial values. If a list is passed, the
        elements are to be integers where no such integer is greater than
        `num_params` Default == None.
    weights : 1D ndarray or None, optional.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.
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
            The initial values to start the optimizatin process with. There
            should be one value for each index coefficient, outside intercept
            parameter, and shape parameter being estimated.

        Returns
        -------
        None.
        """
        # Calculate the expected number of shape and index parameters
        # Note the uneven logit model has one shape parameter per alternative.
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


class MNUL(base_mcm.MNDC_Model):
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
                 shape_names=None,
                 **kwargs):
        ##########
        # Print a helpful message for users who have included shape parameters
        # or shape names unneccessarily
        ##########
        if "shape_ref_pos" in kwargs and kwargs["shape_ref_pos"] is not None:
            warnings.warn(_shape_ref_msg)

        # Carry out the common instantiation process for all choice models
        super(MNUL, self).__init__(data,
                                   alt_id_col,
                                   obs_id_col,
                                   choice_col,
                                   specification,
                                   intercept_ref_pos=intercept_ref_pos,
                                   names=names,
                                   intercept_names=intercept_names,
                                   shape_names=shape_names,
                                   model_type=display_name_dict["Uneven"])

        # Store the utility transform function
        self.utility_transform = partial(_uneven_utility_transform,
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
                constrained_pos=None,
                just_point=False,
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
        constrained_pos : list or None, optional.
            Denotes the positions of the array of estimated parameters that are
            not to change from their initial values. If a list is passed, the
            elements are to be integers where no such integer is greater than
            `init_vals.size.` Default == None.
        just_point : bool, optional.
            Determines whether (True) or not (False) calculations that are non-
            critical for obtaining the maximum likelihood point estimate will
            be performed. If True, this function will return the results
            dictionary from scipy.optimize. Default == False.

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
                init_vals = np.concatenate((init_shapes, init_coefs), axis=0)

        elif init_vals is None:
            msg = "If init_vals is None, then users must pass both init_coefs "
            msg_2 = "and init_shapes."
            raise ValueError(msg + msg_2)

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
        uneven_estimator = UnevenEstimator(self,
                                           mapping_res,
                                           ridge,
                                           zero_vector,
                                           split_param_vec,
                                           constrained_pos=constrained_pos)
        # Set the derivative functions for estimation
        uneven_estimator.set_derivatives()

        # Perform one final check on the length of the initial values
        uneven_estimator.check_length_of_initial_values(init_vals)

        # Get the estimation results
        estimation_res = estimate(init_vals,
                                  uneven_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res,
                                  just_point=just_point)

        if not just_point:
            # Store the estimation results
            self.store_fit_results(estimation_res)

            return None
        else:
            return estimation_res
