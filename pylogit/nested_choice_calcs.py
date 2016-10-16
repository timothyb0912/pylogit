# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:57:29 2016

@module:    nested_coice_calcs.py
@name:      Nested Choice Model Calculations
@author:    Timothy Brathwaite
@summary:   Contains generic functions necessary for calculating 'nested'
            choice probabilities and for estimating the 'nested' choice models.
"""
import numpy as np

# Define the boundary values which are not to be exceeded ducing computation
min_exponent_val = -700
max_exponent_val = 700

max_comp_value = 1e300
min_comp_value = 1e-300


def calc_nested_probs(nest_coefs,
                      index_coefs,
                      design,
                      rows_to_obs,
                      rows_to_nests,
                      chosen_row_to_obs=None,
                      return_type="long_probs",
                      *args,
                      **kwargs):
    """
    Parameters
    ----------
    nest_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each nesting coefficient being estimated. If 2D, should
        have 1 column for each set of nesting coefficients being used to
        predict the probabilities of each alternative being chosen. There
        should be one row per nesting coefficient. Elements denote the inverse
        of the scale coefficients for each of the lower level nests.
    index_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each utility coefficient being estimated (i.e.
        num_features). If 2D, should have 1 column for each set of coefficients
        being used to predict the probabilities of each alternative being
        chosen. There should be one row per index coefficient.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should with one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).
    chosen_row_to_obs : 2D scipy sparse array, or None, optional.
        There should be one row per observation per available alternative and
        one column per observation. This matrix indicates, for each observation
        (on the columns), which rows of the design matrix were the realized
        outcome. If an array is passed then an array of shape
        (num_observations,) can be returned and each element will be the
        probability of the realized outcome of the given observation.
        Default == None.
    return_type : str, optional.
        Indicates what object(s) are to be returned from the function. Valid
        values are: `['long_probs', 'chosen_probs', 'long_and_chosen_probs',
        'all_prob_dict']`. If `long_probs`, the long format probabilities (a 1D
        numpy array with one element per observation per available alternative)
        will be returned. If `chosen_probs`, a 1D numpy array with one element
        per observation will be returned, where the values are the
        probabilities of the chosen alternative for the given observation. If
        `long_and_chosen_probs`, a tuple of chosen_probs and long_probs will be
        returned. If `all_prob_dict`, a dictionary will be returned. The values
        will all be 1D numpy arrays of probabilities dictated by the value's
        corresponding key. The keys will be `long_probs`, `nest_choice_probs`,
        `prob_given_nest`, and `chosen_probs`. If chosen_row_to_obs is None,
        then `chosen_probs` will be None. If `chosen_row_to_obs` is passed,
        then `chosen_probs` will be a 1D array as described above.
        `nest_choice_probs` is of the same shape as `rows_to_nests` and it
        denotes the probability of each individual choosing each of the
        possible nests. `prob_given_nest` is of the same shape as `long_probs`
        and it denotes the probability of the individual associated with a
        given row choosing the alternative associated with that row, given that
        the individual chooses the nest that contains the given alternative.
        Default == `long_probs`.

    Returns
    -------
    See above for documentation of the `return_type` kwarg.
    """
    # Check for 2D index coefficients or nesting coefficients
    try:
        assert len(index_coefs.shape) <= 2
        assert (len(index_coefs.shape) == 1) or (index_coefs.shape[1] == 1)
        assert len(nest_coefs.shape) <= 2
        assert (len(nest_coefs.shape) == 1) or (nest_coefs.shape[1] == 1)
    except AssertionError:
        msg = "Support for 2D index_coefs or nest_coefs not yet implemented."
        raise NotImplementedError(msg)

    # Check for kwarg validity
    valid_return_types = ['long_probs',
                          'chosen_probs',
                          'long_and_chosen_probs',
                          'all_prob_dict']
    if return_type not in valid_return_types:
        msg = "return_type must be one of the following values: "
        raise ValueError(msg + str(valid_return_types))

    chosen_probs_needed = ['chosen_probs', 'long_and_chosen_probs']
    if chosen_row_to_obs is None and return_type in chosen_probs_needed:
        msg = "chosen_row_to_obs is None AND return_type in {}."
        raise ValueError(msg.format(chosen_probs_needed) +
                         "\nThis is invalid.")

    # Calculate the index for each alternative for each individual, V = X*beta
    index_vals = design.dot(index_coefs)

    # Get the long format nest parameters for each row of the design matrix
    long_nest_coefs = rows_to_nests.dot(nest_coefs)

    # Calculate the scaled index values (index / nest_param = V / lambda)
    scaled_index = index_vals / long_nest_coefs

    # Guard against overflow
    pos_inf_idx = np.isposinf(scaled_index)
    neg_inf_idx = np.isneginf(scaled_index)
    scaled_index[pos_inf_idx] = max_comp_value
    scaled_index[neg_inf_idx] = -1 * max_comp_value

    # Calculate the e^(scaled-index) = exp(V / lambda)
    exp_scaled_index = np.exp(scaled_index)

    # Guard against overflow
    inf_idx = np.isposinf(exp_scaled_index)
    exp_scaled_index[inf_idx] = max_comp_value
    # Guard against underflow. Note that I'm not sure this is the best place or
    # best way to perform such guarding. If all of an observations indices
    # suffer underflow, then we'll have 0 / 0 when calculating the
    # probabilities and I should use L'Hopital's rule to get the correct
    # probability. However, replacing underflowed values here may result in
    # incorrectly assigning probabilities of either zero for all alternatives
    # or 1 / num_alternatives for all alternatives.
    zero_idx = (exp_scaled_index == 0)
    exp_scaled_index[zero_idx] = min_comp_value

    # Calculate the log-sum for each nest, for each observation. Note that the
    # "*" is used to compute the dot produce between the mapping matrix which
    # is a scipy.sparse matrix and the second term which is a dense numpy
    # matrix. Note the dimensions of ind_log_sums_per_nest are (obs, nests).
    # Calculates sum _{j \in C_m} exp(V_{ij} / \lambda_m) for each nest m.
    ind_exp_sums_per_nest = (rows_to_obs.T *
                             rows_to_nests.multiply(exp_scaled_index[:, None]))
    # Guard against overflow
    inf_idx = np.isposinf(ind_exp_sums_per_nest)
    ind_exp_sums_per_nest[inf_idx] = max_comp_value

    # Get the long-format representation of ind_log_sums_per_nest. Each row
    # will have two columns, one for each nest. The entries of the matrix will
    # be the log-sum for each nest, for the individual associated with the
    # given row. The "*" is used to perform the dot product since rows_to_obs
    # is a sparse matrix & ind_exp_sums_per_nest is a dense numpy matrix.
    long_exp_sums_per_nest = rows_to_obs * ind_exp_sums_per_nest

    # Get the relevant log-sum for each row of the long-format data
    # Note the .A converts the numpy matrix into a numpy array
    # This is sum _{j \in C_m} exp(V_{ij} / \lambda_m) for the nest
    # belonging to each row
    long_exp_sums = (rows_to_nests.multiply(long_exp_sums_per_nest)
                                  .sum(axis=1)
                                  .A).ravel()


    # Get the denominators for each individual
    ind_denom = (np.power(ind_exp_sums_per_nest,
                          nest_coefs[None, :])
                   .sum(axis=1))
    # Guard against overflow and underflow
    inf_idx = np.isposinf(ind_denom)
    ind_denom[inf_idx] = max_comp_value

    zero_idx = (ind_denom == 0)
    ind_denom[zero_idx] = min_comp_value

    # Get the long format denominators
    long_denom = rows_to_obs.dot(ind_denom).A.ravel()

    # Get the long format numerators
    long_numerators = (exp_scaled_index *
                       np.power(long_exp_sums,
                                (long_nest_coefs - 1)))
    # Guard agains overflow and underflow
    inf_idx = np.isposinf(long_numerators)
    long_numerators[inf_idx] = max_comp_value

    zero_idx = (long_numerators == 0)
    long_numerators[zero_idx] = min_comp_value

    # Calculate and return the long-format probabilities
    long_probs = (long_numerators / long_denom).ravel()
    # Guard against underflow
    long_probs[np.where(long_probs == 0)] = min_comp_value

    # If desired, isolate the probabilities of the chosen alternatives
    if chosen_row_to_obs is None:
        chosen_probs = None
    else:
        # chosen_probs will be of shape (num_observations,)
        chosen_probs = (chosen_row_to_obs.transpose()
                                         .dot(long_probs))
        chosen_probs = np.asarray(chosen_probs).ravel()

    # Return the long form and chosen probabilities if desired
    if return_type == 'long_and_chosen_probs':
        return chosen_probs, long_probs
    # If working with predictions, return just the long form probabilities
    elif return_type == 'long_probs':
        return long_probs
    # If estimating the model and storing fitted probabilities or testing the
    # model on data for which we know the chosen alternative, just return the
    # chosen probabilities.
    elif return_type == 'chosen_probs':
        return chosen_probs
    # If we want all the factors of the probability (e.g. as when calculating
    # the gradient)
    elif return_type == 'all_prob_dict':
        # Create the dictionary of the various probabilities to be returned
        prob_dict = {}
        prob_dict["long_probs"] = long_probs
        prob_dict["chosen_probs"] = chosen_probs

        # Calculate the 'prob_given_nest' array
        prob_given_nest = exp_scaled_index / long_exp_sums
        # Guard against underflow
        zero_idx = (prob_given_nest == 0)
        prob_given_nest[zero_idx] = min_comp_value

        # Calculate the 'nest_choice_probs'. Note ind_denom is a matrix with
        # shape (num_obs, 1) so no need to explicitly broadcast
        nest_choice_probs = (np.power(ind_exp_sums_per_nest,
                                      nest_coefs[None, :]) /
                             ind_denom)
        # Guard against underflow
        zero_idx = (nest_choice_probs == 0)
        nest_choice_probs[zero_idx] = min_comp_value
        # Return dictionary.
        # Note the ".A" converts the numpy matrix into a numpy array
        prob_dict["prob_given_nest"] = prob_given_nest
        prob_dict["nest_choice_probs"] = nest_choice_probs.A
        prob_dict["ind_sums_per_nest"] = ind_exp_sums_per_nest

        return prob_dict


def calc_nested_log_likelihood(nest_coefs,
                               index_coefs,
                               design,
                               rows_to_obs,
                               rows_to_nests,
                               choice_vector,
                               ridge=None,
                               *args,
                               **kwargs):
    """
    Parameters
    ----------
    nest_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each nesting coefficient being estimated. If 2D, should
        have 1 column for each set of nesting coefficients being used to
        predict the probabilities of each alternative being chosen. There
        should be one row per nesting coefficient. Elements denote the inverse
        of the scale coefficients for each of the lower level nests.
    index_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each utility coefficient being estimated
        (i.e. num_features). If 2D, should have 1 column for each set of
        coefficients being used to predict the probabilities of choosing each
        alternative. There should be one row per index coefficient.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization. Default = None.

    Returns
    -------
    log_likelihood : float.
        The log likelihood of the nested logit model. Includes ridge penalty if
        a penalized regression is being performed.
    """
    # Calculate the probability of each individual choosing each available
    # alternative for that individual.
    long_probs = calc_nested_probs(nest_coefs,
                                   index_coefs,
                                   design,
                                   rows_to_obs,
                                   rows_to_nests,
                                   return_type='long_probs')

    # Calculate the log likelihood
    log_likelihood = choice_vector.dot(np.log(long_probs))

    if ridge is None:
        return log_likelihood
    else:
        # Note that the 1.0 is used since the 'null' nest coefficient is equal
        # to 1.0.
        params = np.concatenate(((nest_coefs - 1.0), index_coefs), axis=0)

        return log_likelihood - ridge * np.square(params).sum()


# Create a function to create the various arrays that are needed when
# calculating the gradient of the nested logit model
def prep_vectors_for_gradient(nest_coefs,
                              index_coefs,
                              design,
                              choice_vec,
                              rows_to_obs,
                              rows_to_nests,
                              *args,
                              **kwargs):
    """
    Parameters
    ----------
    nest_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each nesting coefficient being estimated. If 2D, should
        have 1 column for each set of nesting coefficients being used to
        predict the probabilities of each alternative being chosen. There
        should be one row per nesting coefficient. Elements denote the inverse
        of the scale coefficients for each of the lower level nests.
    index_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each utility coefficient being estimated
        (i.e. num_features). If 2D, should have 1 column for each set of
        coefficients being used to predict the probabilities of choosing each
        alternative. There should be one row per index coefficient.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated.
        All elements should be ints, floats, or longs.
    choice_vec : 1D ndarray.
        All elements should by ints, floats, or longs. Each element represents
        whether the individual associated with the given row chose the
        alternative associated with the given row. Should have the same number
        of rows as `design`.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).

    Returns
    -------
    desired_arrays : dict.
        Will contain the arrays necessary for calculating the gradient of the
        nested logit log-likelihood. The keys will be:
            `["long_nest_params", "scaled_y", "long_chosen_nest",
              "obs_to_chosen_nests", "p_tilde_given_nest", "long_probs",
              "prob_given_nest", "nest_choice_probs", "ind_sums_per_nest"]`
    """
    # Create the "long_nest_parameters" which is an array with one element per
    # alternative per observation, where each element is the nest parameter for
    # the alternative corresponding to the given row
    long_nest_params = (rows_to_nests.multiply(nest_coefs[None, :])
                                     .sum(axis=1)
                                     .A
                                     .ravel())

    # Calculate y-tilde
    scaled_y = choice_vec / long_nest_params
    # Guard against overflow
    inf_index = np.isinf(scaled_y)
    scaled_y[inf_index] = max_comp_value

    # Determine which nest was chosen by each row's individual.
    # Resulting matrix has shape (num_rows, num_nests)
    obs_to_chosen_nests = (rows_to_obs.T *
                           rows_to_nests.multiply(choice_vec[:, None])).A
    row_to_chosen_nest = rows_to_obs * obs_to_chosen_nests
    # Determine whether the given row is part of the nest that was chosen
    long_chosen_nest = (rows_to_nests.multiply(row_to_chosen_nest)
                                     .sum(axis=1)
                                     .A
                                     .ravel())

    # Get the various probabilities
    prob_dict = calc_nested_probs(nest_coefs,
                                  index_coefs,
                                  design,
                                  rows_to_obs,
                                  rows_to_nests,
                                  return_type='all_prob_dict')

    # Calculate p_tilde_ij_given_nest
    p_tilde_row_given_nest = (prob_dict["prob_given_nest"] *
                              long_chosen_nest /
                              long_nest_params)
    # Guard against overflow
    inf_index = np.isinf(p_tilde_row_given_nest)
    p_tilde_row_given_nest[inf_index] = max_comp_value

    # Return all the desired matrices and arrays
    desired_arrays = {}
    desired_arrays["long_nest_params"] = long_nest_params.ravel()
    desired_arrays["scaled_y"] = scaled_y.ravel()
    desired_arrays["long_chosen_nest"] = long_chosen_nest
    desired_arrays["obs_to_chosen_nests"] = obs_to_chosen_nests
    desired_arrays["p_tilde_given_nest"] = p_tilde_row_given_nest
    desired_arrays["long_probs"] = prob_dict["long_probs"]
    desired_arrays["prob_given_nest"] = prob_dict["prob_given_nest"]
    desired_arrays["nest_choice_probs"] = prob_dict["nest_choice_probs"]
    desired_arrays["ind_sums_per_nest"] = prob_dict["ind_sums_per_nest"].A

    return desired_arrays


# Define a function to convert the transformed nest coefs
# (i.e. the logit of the nest coefficients) to the
# 'standard' nest coefficients
def naturalize_nest_coefs(nest_coef_estimates):
    """
    Parameters
    ----------
    nest_coef_estimates : 1D ndarray.
        Should contain the estimated logit's
        (`ln[nest_coefs / (1 - nest_coefs)]`) of the true nest coefficients.
        All values should be ints, floats, or longs.

    Returns
    -------
    nest_coefs : 1D ndarray.
        Will contain the 'natural' nest coefficients:
        `1.0 / (1.0 + exp(-nest_coef_estimates))`.
    """
    # Calculate the exponential term of the
    # logistic transformation
    exp_term = np.exp(-1 * nest_coef_estimates)

    # Guard against_overflow
    inf_idx = np.isinf(exp_term)
    exp_term[inf_idx] = max_comp_value

    # Calculate the 'natural' nest coefficients
    nest_coefs = 1.0 / (1.0 + exp_term)

    # Guard against underflow
    zero_idx = (nest_coefs == 0)
    nest_coefs[zero_idx] = min_comp_value

    return nest_coefs


# Create the actual function used to calculate the gradient
def calc_nested_gradient(orig_nest_coefs,
                         index_coefs,
                         design,
                         choice_vec,
                         rows_to_obs,
                         rows_to_nests,
                         ridge=None,
                         use_jacobian=True,
                         *args,
                         **kwargs):
    """
    Parameters
    ----------
    orig_nest_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each nesting coefficient being estimated. If 2D, should
        have 1 column for each set of nesting coefficients being used to
        predict the probabilities of each alternative being chosen. There
        should be one row per nesting coefficient. Elements denote the logit of
        the inverse of the scale coefficients for each lower level nests.
    index_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each utility coefficient being estimated
        (i.e. num_features). If 2D, should have 1 column for each set of
        coefficients being used to predict the probabilities of choosing each
        alternative. There should be one row per index coefficient.
    design : 2D ndarray.
       There should be one row per observation per available alternative. There
       should be one column per utility coefficient being estimated. All
       elements should be ints, floats, or longs.
    choice_vec : 1D ndarray.
        All elements should by ints, floats, or longs. Each element represents
        whether the individual associated with the given row chose the
        alternative associated with the given row.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization. Default `== None`.
    use_jacobian : bool, optional.
        Determines whether or not the jacobian will be used when calculating
        the gradient. When performing model estimation, `use_jacobian` should
        be `True` if the values being estimated are actually the logit of the
        nest coefficients. Default `== True`.

    Returns
    -------
    gradient : 1D numpy array.
       The gradient of the log-likelihood with respect to the given nest
       coefficients and index coefficients.
    """
    # Transform the nest coefficients into their "always positive" versions
    nest_coefs = naturalize_nest_coefs(orig_nest_coefs)

    # Get the vectors and matrices needed to calculate the gradient
    vector_dict = prep_vectors_for_gradient(nest_coefs,
                                            index_coefs,
                                            design,
                                            choice_vec,
                                            rows_to_obs,
                                            rows_to_nests)

    # Calculate the index for each alternative for each person
    sys_utility = design.dot(index_coefs)

    # Calculate w_ij
    long_w = sys_utility / vector_dict["long_nest_params"]
    # Guard against overflow
    inf_index = np.isposinf(long_w)
    long_w[inf_index] = max_comp_value

    ##########
    # Calculate d_log_likelihood_d_nest_params
    ##########
    # Calculate the term that onlny depends on nest level values
    log_exp_sums = np.log(vector_dict["ind_sums_per_nest"])
    # Guard against overflow
    log_exp_sums[np.isneginf(log_exp_sums)] = -1 * max_comp_value

    # Calculate the first term of the derivative of the log-liikelihood
    # with respect to the nest parameters
    nest_gradient_term_1 = ((vector_dict["obs_to_chosen_nests"] -
                             vector_dict["nest_choice_probs"]) *
                            log_exp_sums).sum(axis=0)

    # Calculate the second term of the derivative of the log-liikelihood
    # with respect to the nest parameters
    half_deriv = ((vector_dict["long_probs"] -
                   vector_dict["long_chosen_nest"] *
                   vector_dict["prob_given_nest"]) *
                  long_w)
    nest_gradient_term_2 = (rows_to_nests.transpose()
                                         .dot(half_deriv)[:, None]).ravel()

    # Calculate the third term of the derivative of the log-likelihood
    # with respect to the nest parameters
    nest_gradient_term_3a = (choice_vec -
                             vector_dict["long_chosen_nest"] *
                             vector_dict["prob_given_nest"])
    nest_gradient_term_3b = ((-1 * nest_gradient_term_3a * long_w) /
                             vector_dict["long_nest_params"])
    # Guard against overflow
    inf_idx = np.isposinf(nest_gradient_term_3b)
    nest_gradient_term_3b[inf_idx] = max_comp_value

    neg_inf_idx = np.isneginf(nest_gradient_term_3b)
    nest_gradient_term_3b[neg_inf_idx] = -1 * max_comp_value

    # Get the nest-wide version of this piece of the gradient
    nest_gradient_term_3 = (rows_to_nests.transpose()
                                         .dot(nest_gradient_term_3b)).ravel()

    # Combine the two terms. Note the "nest_coefs * (1 - nest_coefs)" is due to
    # the fact that we're estimating the logit of the nest coefficients instead
    # of the nest coefficient itself. We therefore need to multiply by
    # d_nest_coef_d_estimated_variable to get the correct gradient.
    # d_nest_coef_d_estimated_variable == nest_coefs * (1 - nest_coefs).
    if use_jacobian:
        jacobian = nest_coefs * (1.0 - nest_coefs)
    else:
        jacobian = 1
    nest_gradient = ((nest_gradient_term_1 +
                      nest_gradient_term_2 +
                      nest_gradient_term_3) *
                     jacobian)[None, :]

    ##########
    # Calculate d_loglikelihood_d_beta
    ##########
    beta_gradient_term_1 = (vector_dict["scaled_y"] -
                            vector_dict["p_tilde_given_nest"] +
                            vector_dict["p_tilde_given_nest"] *
                            vector_dict["long_nest_params"] -
                            vector_dict["long_probs"])[None, :]
    #####
    # Calculate the derivative with respect to beta
    #####
    beta_gradient = beta_gradient_term_1.dot(design)

    #####
    # Combine the gradient pieces and account for ridge parameter
    #####
    gradient = np.concatenate((nest_gradient, beta_gradient), axis=1).ravel()

    if ridge is not None:
        # Note that the 20 is used in place of 'infinity' since I would really
        # like to specify the expected value of the nest coefficient to 1, but
        # that would make the logit of the nest parameter infinity. Instead I
        # use 20 as a close enough value-- (1 + exp(-20))**-1 is approx. 1.
        params = np.concatenate(((nest_coefs - 20), index_coefs), axis=0)

        gradient -= 2 * ridge * params

    return gradient


# Define a function that will calculate the BHHH approximation of the hessian
# This is essentially the sum over all individuals, of the outer product of
# the gradient with itself.
def calc_bhhh_hessian_approximation(orig_nest_coefs,
                                    index_coefs,
                                    design,
                                    choice_vec,
                                    rows_to_obs,
                                    rows_to_nests,
                                    ridge=None,
                                    use_jacobian=False,
                                    *args,
                                    **kwargs):
    """
    Parameters
    ----------
    orig_nest_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each nesting coefficient being estimated. If 2D, should
        have 1 column for each set of nesting coefficients being used to
        predict the probabilities of each alternative being chosen. There
        should be one row per nesting coefficient. Elements denote the inverse
        of the scale coefficients for each of the lower level nests.
    index_coefs : 1D or 2D ndarray.
        All elements should by ints, floats, or longs. If 1D, should have 1
        element for each utility coefficient being estimated
        (i.e. num_features). If 2D, should have 1 column for each set of
        coefficients being used to predict the probabilities of choosing each
        alternative. There should be one row per index coefficient.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated. All
        elements should be ints, floats, or longs.
    choice_vec : 1D ndarray.
        All elements should by ints, floats, or longs. Each element represents
        whether the individual associated with the given row chose the
        alternative associated with the given row.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If an int,
        float or long is passed, then that scalar determines the ridge penalty
        for the optimization. Default == None. Note that if this parameter is
        passed, the values of the BHHH matrix MAY BE INCORRECT since it is not
        100% clear how penalization affects the information matrix.
    use_jacobian : bool, optional.
        Determines whether or not the jacobian will be used when calculating
        the gradient. When performing model estimation, `use_jacobian` should
        be `True` if the values that are actually being estimated are the
        logit of the nest coefficients. Default `== False`.

    Returns
    -------
    bhhh_matrix : 2D ndarray.
       The negative of the sum of the outer products of the gradient of the
       log-likelihood function for each observation.
    """
    # Transform the nest coefficients into their "always positive" versions
    nest_coefs = naturalize_nest_coefs(orig_nest_coefs)

    # Get the vectors and matrices needed to calculate the gradient
    vector_dict = prep_vectors_for_gradient(nest_coefs,
                                            index_coefs,
                                            design,
                                            choice_vec,
                                            rows_to_obs,
                                            rows_to_nests)

    # Calculate the index for each alternative for each person
    sys_utility = design.dot(index_coefs)

    # Calculate w_ij
    long_w = sys_utility / vector_dict["long_nest_params"]
    # Guard against overflow
    inf_index = np.isposinf(long_w)
    long_w[inf_index] = max_comp_value

    ##########
    # Calculate d_log_likelihood_d_nest_params
    ##########
    # Calculate the term that onlny depends on nest level values
    log_exp_sums = np.log(vector_dict["ind_sums_per_nest"])
    # Guard against overflow
    log_exp_sums[np.isneginf(log_exp_sums)] = -1 * max_comp_value

    # Calculate the first term of the derivative of the log-liikelihood
    # with respect to the nest parameters. Note we do not sum this object
    # because we want the values at the 'individual' level, which they already
    # are
    nest_gradient_term_1 = ((vector_dict["obs_to_chosen_nests"] -
                             vector_dict["nest_choice_probs"]) *
                            log_exp_sums)

    # Calculate the second term of the derivative of the log-liikelihood
    # with respect to the nest parameters
    half_deriv = ((vector_dict["long_probs"] -
                   vector_dict["long_chosen_nest"] *
                   vector_dict["prob_given_nest"]) *
                  long_w)[:, None]
    # "Spread out" the second term across the appropriate nests
    spread_half_deriv = rows_to_nests.multiply(half_deriv)
    # Aggregate the spread out half-derivatives to the individual level
    # This object should have shape (num_obs, num_nests)
    nest_gradient_term_2 = rows_to_obs.transpose().dot(spread_half_deriv).A

    # Calculate the third term of the derivative of the log-likelihood
    # with respect to the nest parameters
    nest_gradient_term_3a = (choice_vec -
                             vector_dict["long_chosen_nest"] *
                             vector_dict["prob_given_nest"])
    nest_gradient_term_3b = ((-1 * nest_gradient_term_3a * long_w) /
                             vector_dict["long_nest_params"])
    # Guard against overflow
    inf_idx = np.isposinf(nest_gradient_term_3b)
    nest_gradient_term_3b[inf_idx] = max_comp_value

    neg_inf_idx = np.isneginf(nest_gradient_term_3b)
    nest_gradient_term_3b[neg_inf_idx] = -1 * max_comp_value

    # Get the nest-wide version of this piece of the gradient
    spread_out_term_3b = rows_to_nests.multiply(nest_gradient_term_3b[:, None])
    nest_gradient_term_3 = rows_to_obs.transpose().dot(spread_out_term_3b)

    # Combine the two terms. Note the "nest_coefs * (1 - nest_coefs)" is due to
    # the fact that we're estimating the logit of the nest coefficients instead
    # of the nest coefficient itself. We therefore need to multiply by
    # d_nest_coef_d_estimated_variable to get the correct gradient.
    # d_nest_coef_d_estimated_variable == nest_coefs * (1 - nest_coefs).
    # As with the various nest_gradient_terms, the nest_gradient should be of
    # shape (num_obs, num_nests)
    if use_jacobian:
        jacobian = (nest_coefs * (1.0 - nest_coefs))[None, :]
    else:
        jacobian = 1
    nest_gradient = ((nest_gradient_term_1 +
                      nest_gradient_term_2 +
                      nest_gradient_term_3).A *
                     jacobian)

    ##########
    # Calculate d_loglikelihood_d_beta
    ##########
    beta_gradient_term_1 = (vector_dict["scaled_y"] -
                            vector_dict["p_tilde_given_nest"] +
                            vector_dict["p_tilde_given_nest"] *
                            vector_dict["long_nest_params"] -
                            vector_dict["long_probs"])[:, None]
    #####
    # Calculate the derivative with respect to beta
    #####
    beta_gradient = rows_to_obs.T.dot(beta_gradient_term_1 * design)

    #####
    # Combine the gradient pieces
    #####
    gradient_matrix = np.concatenate((nest_gradient, beta_gradient), axis=1)

    #####
    # Compute and return the outer product of each row of the gradient
    # with itself. Then sum these individual matrices together.
    #####
#    fisher_matrix = (gradient_vec[:, :, np.newaxis] *
#                     gradient_vec[:, np.newaxis, :]).sum(axis=0)
    # The next five lines replicate the procedure accomplished in two
    # lines by the vectorized code above. However, when the design
    # matrix is large, the vectorized code will cause memory problems
    # by creating matrices that are even larger than the design matrix.
    # The for-loop based calculation avoids memory issues.
    bhhh_matrix = np.zeros((gradient_matrix.shape[1],
                            gradient_matrix.shape[1]))
    for row_idx in range(gradient_matrix.shape[0]):
        bhhh_matrix += np.outer(gradient_matrix[row_idx],
                                gradient_matrix[row_idx])

    if ridge is not None:
        # The rational behind subtracting 2 * ridge is that the information
        # matrix should approximate the hessian and in the hessian we subtract
        # 2 * ridge at the end. I don't know if this is the correct way to
        # calculate the Fisher Information Matrix in ridge regression models.
        bhhh_matrix -= 2 * ridge

    # Note the "-1" is because we are approximating the Fisher information
    # matrix which has a negative one in the front of it?
    return -1 * bhhh_matrix
