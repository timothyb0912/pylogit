# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:56:47 2016

@module:    mixed_logit_calcs.py
@name:      Mixed Logit Choice Calculations
@author:    Timothy Brathwaite
@summary:   Contains generic functions necessary for calculating choice
            probabilities and for estimating Mixed MNL models.

Current Assumptions (to be relaxed later):
    - The probability function being mixed (i.e. the kernel) is the MNL model.
    - Mixing is only permitted for index coefficients.
    - The number of draws from each parameter's distributions must be equal.
    - Only the normal distribution is supported as a mixing distribution
      (at the moment).
    - Only psuedo-random draws are supported at the moment. Halton draws are
      not yet supported.
    - It is assumed that individuals are the units being mixed over
      (i.e. parameters are randomly distributed over observations).
"""
from __future__ import absolute_import

import scipy.stats
import numpy as np
from . import choice_calcs as cc

try:
    # Python 3.x does support xrange
    from past.builtins import xrange
except ImportError:
    pass


# Define the boundary values which are not to be exceeded ducing computation
min_exponent_val = -700
max_exponent_val = 700

max_comp_value = 1e300
min_comp_value = 1e-300

# Alias necessary functions from the base multinomial choice model module
general_calc_probabilities = cc.calc_probabilities


def get_normal_draws(num_mixers,
                     num_draws,
                     num_vars,
                     seed=None):
    """
    Parameters
    ----------
    num_mixers : int.
        Should be greater than zero. Denotes the number of observations for
        which we are making draws from a normal distribution for. I.e. the
        number of observations with randomly distributed coefficients.
    num_draws : int.
        Should be greater than zero. Denotes the number of draws that are to be
        made from each normal distribution.
    num_vars : int.
        Should be greater than zero. Denotes the number of variables for which
        we need to take draws from the normal distribution.
    seed : int or None, optional.
        If an int is passed, it should be greater than zero. Denotes the value
        to be used in seeding the random generator used to generate the draws
        from the normal distribution. Default == None.

    Returns
    -------
    all_draws : list of 2D ndarrays.
        The list will have num_vars elements. Each element will be a num_mixers
        by num_draws numpy array of draws from a normal distribution with mean
        zero and standard deviation of one.
    """
    # Check the validity of the input arguments
    assert all([isinstance(x, int) for x in [num_mixers, num_draws, num_vars]])
    assert all([x > 0 for x in [num_mixers, num_draws, num_vars]])
    if seed is not None:
        assert isinstance(seed, int) and seed > 0

    normal_dist = scipy.stats.norm(loc=0.0, scale=1.0)
    all_draws = []
    if seed:
        np.random.seed(seed)
    for i in xrange(num_vars):
        all_draws.append(normal_dist.rvs(size=(num_mixers, num_draws)))
    return all_draws


def convert_mixing_names_to_positions(mixing_names, ind_var_names):
    """
    Parameters
    ----------
    mixing_names : list of strings.
        Denotes the names of the index variables that are being treated as
        random variables.
    ind_var_names : list of strings.
        Each string should represent (in order) the variables in the index.

    Returns
    -------
    list. All elements should be ints. Elements will be the position in
        `ind_var_names` of each of the elements in `mixing_names`.
    """
    return [ind_var_names.index(name) for name in mixing_names]


def create_expanded_design_for_mixing(design,
                                      draw_list,
                                      mixing_pos,
                                      rows_to_mixers):
    """
    Parameters
    ----------
    design : 2D ndarray.
        All elements should be ints, floats, or longs. Each row corresponds to
        an available alternative for a given individual. There should be one
        column per index coefficient being estimated.
    draw_list : list of 2D ndarrays.
        All numpy arrays should have the same number of columns (`num_draws`)
        and the same number of rows (`num_mixers`). All elements of the numpy
        arrays should be ints, floats, or longs. Should have as many elements
        as there are lements in `mixing_pos`.
    mixing_pos : list of ints.
        Each element should denote a column in design whose associated index
        coefficient is being treated as a random variable.
    rows_to_mixers : 2D scipy sparse array.
        All elements should be zeros and ones. Will map the rows of the design
        matrix to the particular units that the mixing is being performed over.
        Note that in the case of panel data, this matrix will be different from
        `rows_to_obs`.

    Returns
    -------
    design_3d : 3D numpy array.
        Each slice of the third dimension will contain a copy of the design
        matrix corresponding to a given draw of the random variables being
        mixed over.
    """
    if len(mixing_pos) != len(draw_list):
        msg = "mixing_pos == {}".format(mixing_pos)
        msg_2 = "len(draw_list) == {}".format(len(draw_list))
        raise ValueError(msg + "\n" + msg_2)

    # Determine the number of draws being used. Note the next line assumes an
    # equal number of draws from each random coefficient's mixing distribution.
    num_draws = draw_list[0].shape[1]
    orig_num_vars = design.shape[1]

    # Initialize the expanded design matrix that replicates the columns of the
    # variables that are being mixed over.
    arrays_for_mixing = design[:, mixing_pos]
    expanded_design = np.concatenate((design, arrays_for_mixing),
                                     axis=1).copy()
    design_3d = np.repeat(expanded_design[:, None, :],
                          repeats=num_draws,
                          axis=1)

    # Multiply the columns that are being mixed over by their appropriate
    # draws from the normal distribution
    for pos, idx in enumerate(mixing_pos):
        rel_draws = draw_list[pos]
        # Note that rel_long_draws will be a dense, 2D numpy array of shape
        # (num_rows, num_draws).
        rel_long_draws = rows_to_mixers.dot(rel_draws)
        # Create the actual column in design 3d that should be used.
        # It should be the multiplication of the draws random variable and the
        # independent variable associated with the param that is being mixed.
        # NOTE THE IMPLICIT ASSUMPTION THAT ONLY INDEX COEFFICIENTS ARE MIXED.
        # Also, the final axis is selected on because the final axis sepecifies
        # the particular variable being multiplied by the draws. We select with
        # orig_num_vars + pos since the variables being mixed over were added,
        # in order so we simply need to start at the first position after all
        # the original variables (i.e. at orig_num_vars) and iterate.
        design_3d[:, :, orig_num_vars + pos] *= rel_long_draws

    return design_3d


def calc_choice_sequence_probs(prob_array,
                               choice_vec,
                               rows_to_mixers,
                               return_type=None):
    """
    Parameters
    ----------
    prob_array : 2D ndarray.
        All elements should be ints, floats, or longs. All elements should be
        between zero and one (exclusive). Each element should represent the
        probability of the corresponding alternative being chosen by the
        corresponding individual during the given choice situation, given the
        particular draw of coefficients being considered. There should be one
        column for each draw of the coefficients.
    choice_vec : 1D ndarray.
        All elements should be zeros or ones. Should denote the rows that were
        chosen by the individuals corresponding to those rows.
    rows_to_mixers : 2D scipy sparse array.
        All elements should be zeros and ones. Will map the rows of the design
        matrix to the particular units that the mixing is being performed over.
        Note that in the case of panel data, this matrix will be different from
        `rows_to_obs`.
    return_type : `'all'` or None, optional.
        If `'all'` is passed, then a tuple will be returned. The first element
        will be a 1D numpy array of shape `(num_mixing_units,)`. Each value
        will be the average probability of predicting the associated mixing
        unit's probability of making the observed sequence of choices. The
        second element of the tuple will be a 2D numpy array with shape
        `(num_mixing_units, num_draws)`, where
        `num_draws == prob_array.shape[1]`. Each value will be the probability
        of predicting the associated mixing unit's probability of making the
        observed sequence of choices, given the associated draw of the mixing
        distribution for the given individual. If None, only the first
        element of the tuple described above will be returned. Default == None.

    Returns
    -------
    See `return_type` kwarg.
    """
    if return_type not in [None, 'all']:
        raise ValueError("return_type must be None or 'all'.")

    log_chosen_prob_array = choice_vec[:, None] * np.log(prob_array)
    # Create a 2D array with shape (num_mixing_units, num_random_draws)
    # Each element will be the log of the probability of the sequence of
    # choices, given the random draw of the coefficients
    expanded_log_sequence_probs = rows_to_mixers.T.dot(log_chosen_prob_array)
    # Calculate the probability of the sequence of choices for each mixing
    # unit, given the random draw of the coefficients
    expanded_sequence_probs = np.exp(expanded_log_sequence_probs)
    # Guard against underflow since none of the probabilities are actually zero
    zero_idx = np.where(expanded_sequence_probs == 0)
    expanded_sequence_probs[zero_idx] = min_comp_value
    # Calculate the simulated probability of the sequence of choices of each
    # mixing unit
    sequence_probs = expanded_sequence_probs.mean(axis=1)

    if return_type is None:
        return sequence_probs
    elif return_type == 'all':
        return sequence_probs, expanded_sequence_probs


def calc_mixed_log_likelihood(params,
                              design_3d,
                              alt_IDs,
                              rows_to_obs,
                              rows_to_alts,
                              rows_to_mixers,
                              choice_vector,
                              utility_transform,
                              ridge=None,
                              weights=None):
    """
    Parameters
    ----------
    params : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features  +
        num_coefs_being_mixed).
    design_3d : 3D ndarray.
        All elements should be ints, floats, or longs. Should have one row per
        observation per available alternative. The second axis should have as
        many elements as there are draws from the mixing distributions of the
        coefficients. The last axis should have one element per index
        coefficient being estimated.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D scipy sparse array.
        All elements should be zeros and ones. There should be one row per
        observation per available alternative and one column per observation.
        This matrix maps the rows of the design matrix to the unique
        observations (on the columns).
    rows_to_alts : 2D scipy sparse array.
        All elements should be zeros and ones. There should be one row per
        observation per available alternative and one column per possible
        alternative. This matrix maps the rows of the design matrix to the
        possible alternatives for this dataset.
    rows_to_mixers : 2D scipy sparse array.
        All elements should be zeros and ones. Will map the rows of the design
        matrix to the particular units that the mixing is being performed over.
        Note that in the case of panel data, this matrix will be different from
        `rows_to_obs`.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform :  callable.
        Should accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 2D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated and the
        given draw of the random coefficients. There should be one column for
        each draw of the random coefficients. There should have one row per
        individual per choice situation per available alternative.
    ridge : scalar or None, optional.
        Determines whether or not ridge regression is performed. If a scalar is
        passed, then that scalar determines the ridge penalty for the
        optimization. Default = None.
    weights : 1D ndarray or None.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.

    Returns
    -------
    log_likelihood: float.
        The log-likelihood of the mixed logit model evaluated at the passed
        values of `params`.
    """
    # Calculate the weights for the sample
    if weights is None:
        weights = np.ones(design_3d.shape[0])
    weights_per_obs =\
        np.max(rows_to_mixers.toarray() * weights[:, None], axis=0)

    # Calculate the regular probability array. Note the implicit assumption
    # that params == index coefficients.
    prob_array = general_calc_probabilities(params,
                                            design_3d,
                                            alt_IDs,
                                            rows_to_obs,
                                            rows_to_alts,
                                            utility_transform,
                                            return_long_probs=True)

    # Calculate the simulated probability of correctly predicting each persons
    # sequence of choices. Note that this function implicitly assumes that the
    # mixing unit is the individual
    simulated_sequence_probs = calc_choice_sequence_probs(prob_array,
                                                          choice_vector,
                                                          rows_to_mixers)

    # Calculate the log-likelihood of the dataset
    log_likelihood = weights_per_obs.dot(np.log(simulated_sequence_probs))

    # Adjust for the presence of a ridge estimator. Again, note that we are
    # implicitly assuming that the only model being mixed is the MNL model,
    # such that params == index coefficients.
    if ridge is None:
        return log_likelihood
    else:
        return log_likelihood - ridge * np.square(params).sum()


def calc_mixed_logit_gradient(params,
                              design_3d,
                              alt_IDs,
                              rows_to_obs,
                              rows_to_alts,
                              rows_to_mixers,
                              choice_vector,
                              utility_transform,
                              ridge=None,
                              weights=None):
    """
    Parameters
    ----------
    params : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated
        (i.e. num_features  + num_coefs_being_mixed).
    design_3d : 3D ndarray.
        All elements should be ints, floats, or longs. Should have one row per
        observation per available alternative. The second axis should have as
        many elements as there are draws from the mixing distributions of the
        coefficients. The last axis should have one element per index
        coefficient being estimated.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D scipy sparse array.
        All elements should be zeros and ones. Should have one row per
        observation per available alternative and one column per observation.
        This matrix maps the rows of the design matrix to the unique
        observations (on the columns).
    rows_to_alts : 2D scipy sparse array.
        All elements should be zeros and ones. Should have one row per
        observation per available alternative and one column per possible
        alternative. This matrix maps the rows of the design matrix to the
        possible alternatives for this dataset.
    rows_to_mixers : 2D scipy sparse array.
        All elements should be zeros and ones. Will map the rows of the design
        matrix to the particular units that the mixing is being performed over.
        Note that in the case of panel data, this matrix will be different from
        `rows_to_obs`.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform :  callable.
        Should accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 2D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated and the
        given draw of the random coefficients. There should be one column for
        each draw of the random coefficients. There should have one row per
        individual per choice situation per available alternative.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If a float is
        passed, then that float determines the ridge penalty for the
        optimization. Default = None.
    weights : 1D ndarray or None.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.

    Returns
    -------
    gradient : ndarray of shape (design_3d.shape[2],).
        The returned array is the gradient of the log-likelihood of the mixed
        MNL model with respect to `params`.
    """
    # Calculate the weights for the sample
    if weights is None:
        weights = np.ones(design_3d.shape[0])

    # Calculate the regular probability array. Note the implicit assumption
    # that params == index coefficients.
    prob_array = general_calc_probabilities(params,
                                            design_3d,
                                            alt_IDs,
                                            rows_to_obs,
                                            rows_to_alts,
                                            utility_transform,
                                            return_long_probs=True)

    # Calculate the simulated probability of correctly predicting each persons
    # sequence of choices. Note that this function implicitly assumes that the
    # mixing unit is the individual
    prob_results = calc_choice_sequence_probs(prob_array,
                                              choice_vector,
                                              rows_to_mixers,
                                              return_type="all")
    # Calculate the sequence probabilities given random draws
    # and calculate the overal simulated probabilities
    sequence_prob_array = prob_results[1]
    simulated_probs = prob_results[0]

    # Convert the various probabilties to long format
    long_sequence_prob_array = rows_to_mixers.dot(sequence_prob_array)
    long_simulated_probs = rows_to_mixers.dot(simulated_probs)
    # Scale sequence probabilites given random draws by simulated probabilities
    scaled_sequence_probs = (long_sequence_prob_array /
                             long_simulated_probs[:, None])
    # Calculate the scaled error. Will have shape == (num_rows, num_draws)
    scaled_error = ((choice_vector[:, None] - prob_array) *
                    scaled_sequence_probs)

    # Calculate the gradient. Note that the lines below assume that we are
    # taking the gradient of an MNL model. Should refactor to make use of the
    # built in gradient function for logit-type models. Should also refactor
    # the gradient function for logit-type models to be able to handle 2D
    # systematic utility arrays.
    gradient = (scaled_error[:, :, None] *
                design_3d *
                weights[:, None, None]).sum(axis=0)

    gradient = gradient.mean(axis=0)

    # Account for the ridge parameter if an L2 penalization is being performed
    if ridge is not None:
        gradient -= 2 * ridge * params

    return gradient.ravel()


def calc_neg_log_likelihood_and_neg_gradient(beta,
                                             design_3d,
                                             alt_IDs,
                                             rows_to_obs,
                                             rows_to_alts,
                                             rows_to_mixers,
                                             choice_vector,
                                             utility_transform,
                                             constrained_pos,
                                             ridge=None,
                                             weights=None,
                                             *args):
    """
    Parameters
    ----------
    beta : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features  +
        num_coefs_being_mixed).
    design_3d : 3D ndarray.
        All elements should be ints, floats, or longs. Should have one row per
        observation per available alternative. The second axis should have as
        many elements as there are draws from the mixing distributions of the
        coefficients. The last axis should have one element per index
        coefficient being estimated.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D scipy sparse array.
        All elements should be zeros and ones. Should have one row per
        observation per available alternative and one column per observation.
        This matrix maps the rows of the design matrix to the unique
        observations (on the columns).
    rows_to_alts : 2D scipy sparse array.
        All elements should be zeros and ones. Should have one row per
        observation per available alternative and one column per possible
        alternative. This matrix maps the rows of the design matrix to the
        possible alternatives for this dataset.
    rows_to_mixers : 2D scipy sparse array.
        All elements should be zeros and ones. Will map the rows of the design
        matrix to the particular units that the mixing is being performed over.
        Note that in the case of panel data, this matrix will be different from
        `rows_to_obs`.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform :  callable.
        Should accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 2D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated and the
        given draw of the random coefficients. There should be one column for
        each draw of the random coefficients. There should have one row per
        individual per choice situation per available alternative.
    constrained_pos : list of ints, or None, optional.
        Each int denotes a position in the array of estimated parameters that
        are not to change from their initial values. None of the integers
        should be greater than `beta.size`.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If a float is
        passed, then that float determines the ridge penalty for the
        optimization. Default = None.
    weights : 1D ndarray or None.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.

    Returns
    -------
    tuple. (`neg_log_likelihood`, `neg_beta_gradient_vec`).
        The first element is a float. The second element is a 1D numpy array of
        shape (design.shape[1],). The first element is the negative
        log-likelihood of this model evaluated at the passed values of beta.
        The second element is the gradient of the negative log- likelihood with
        respect to the vector of utility coefficients.
    """
    neg_log_likelihood = -1 * calc_mixed_log_likelihood(beta,
                                                        design_3d,
                                                        alt_IDs,
                                                        rows_to_obs,
                                                        rows_to_alts,
                                                        rows_to_mixers,
                                                        choice_vector,
                                                        utility_transform,
                                                        ridge=ridge,
                                                        weights=weights)

    neg_beta_gradient_vec = -1 * calc_mixed_logit_gradient(beta,
                                                           design_3d,
                                                           alt_IDs,
                                                           rows_to_obs,
                                                           rows_to_alts,
                                                           rows_to_mixers,
                                                           choice_vector,
                                                           utility_transform,
                                                           ridge=ridge,
                                                           weights=weights)

    if constrained_pos is not None:
        neg_beta_gradient_vec[constrained_pos] = 0

    return neg_log_likelihood, neg_beta_gradient_vec


def calc_bhhh_hessian_approximation_mixed_logit(params,
                                                design_3d,
                                                alt_IDs,
                                                rows_to_obs,
                                                rows_to_alts,
                                                rows_to_mixers,
                                                choice_vector,
                                                utility_transform,
                                                ridge=None,
                                                weights=None):
    """
    Parameters
    ----------
    params : 1D ndarray.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features  +
        num_coefs_being_mixed).
    design_3d : 3D ndarray.
        All elements should be ints, floats, or longs. Should have one row per
        observation per available alternative. The second axis should have as
        many elements as there are draws from the mixing distributions of the
        coefficients. The last axis should have one element per index
        coefficient being estimated.
    alt_IDs : 1D ndarray.
        All elements should be ints. There should be one row per obervation per
        available alternative for the given observation. Elements denote the
        alternative corresponding to the given row of the design matrix.
    rows_to_obs : 2D scipy sparse array.
        All elements should be zeros and ones. Should have one row per
        observation per available alternative and one column per observation.
        This matrix maps the rows of the design matrix to the unique
        observations (on the columns).
    rows_to_alts : 2D scipy sparse array.
        All elements should be zeros and ones. Should have one row per
        observation per available alternative and one column per possible
        alternative. This matrix maps the rows of the design matrix to the
        possible alternatives for this dataset.
    rows_to_mixers : 2D scipy sparse array.
        All elements should be zeros and ones. Will map the rows of the design
        matrix to the particular units that the mixing is being performed over.
        Note that in the case of panel data, this matrix will be different from
        `rows_to_obs`.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    utility_transform :  callable.
        Should accept a 1D array of systematic utility values, a 1D array of
        alternative IDs, and miscellaneous args and kwargs. Should return a 2D
        array whose elements contain the appropriately transformed systematic
        utility values, based on the current model being evaluated and the
        given draw of the random coefficients. There should be one column for
        each draw of the random coefficients. There should have one row per
        individual per choice situation per available alternative.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If a float is
        passed, then that float determines the ridge penalty for the
        optimization. Default = None.
    weights : 1D ndarray or None, optional.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class. Default == None.

    Returns
    -------
    bhhh_matrix : 2D ndarray of shape `(design.shape[1], design.shape[1])`.
        The returned array is the BHHH approximation of the Fisher Information
        Matrix. I.e it is the negative of the sum of the outer product of
        each individual's gradient with itself.
    """
    # Calculate the weights for the sample
    if weights is None:
        weights = np.ones(design_3d.shape[0])
        
    M = rows_to_obs.multiply(weights.reshape(-1,1))
    weights_per_obs = np.max(M, axis=0).toarray().reshape(-1)
    # Calculate the regular probability array. Note the implicit assumption
    # that params == index coefficients.
    prob_array = general_calc_probabilities(params,
                                            design_3d,
                                            alt_IDs,
                                            rows_to_obs,
                                            rows_to_alts,
                                            utility_transform,
                                            return_long_probs=True)

    # Calculate the simulated probability of correctly predicting each persons
    # sequence of choices. Note that this function implicitly assumes that the
    # mixing unit is the individual
    prob_results = calc_choice_sequence_probs(prob_array,
                                              choice_vector,
                                              rows_to_mixers,
                                              return_type="all")
    # Calculate the sequence probabilities given random draws
    # and calculate the overal simulated probabilities
    sequence_prob_array = prob_results[1]
    simulated_probs = prob_results[0]

    # Convert the various probabilties to long format
    long_sequence_prob_array = rows_to_mixers.dot(sequence_prob_array)
    long_simulated_probs = rows_to_mixers.dot(simulated_probs)
    # Scale sequence probabilites given random draws by simulated probabilities
    scaled_sequence_probs = (long_sequence_prob_array /
                             long_simulated_probs[:, None])
    # Calculate the scaled error. Will have shape == (num_rows, num_draws)
    scaled_error = ((choice_vector[:, None] - prob_array) *
                    scaled_sequence_probs)

    # Calculate the gradient. Note that the lines below assume that we are
    # taking the gradient of an MNL model. Should refactor to make use of the
    # built in gradient function for logit-type models. Should also refactor
    # the gradient function for logit-type models to be able to handle 2D
    # systematic utility arrays. `gradient` will have shape
    # (design_3d.shape[0], design_3d.shape[2])
    gradient = (scaled_error[:, :, None] * design_3d).mean(axis=1)

    gradient_per_obs = rows_to_mixers.T.dot(gradient)

    bhhh_matrix =\
        gradient_per_obs.T.dot(weights_per_obs[:, None] * gradient_per_obs)

    if ridge is not None:
        bhhh_matrix -= 2 * ridge * np.identity(bhhh_matrix.shape[0])

    # Note the "-1" is because we are approximating the Fisher information
    # matrix which has a negative one in the front of it?
    return -1 * bhhh_matrix
