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


import time, sys
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags

import choice_calcs as cc
import base_multinomial_cm_v2 as base_mcm
import choice_model_em

# Alias necessary functions from the base multinomial choice model module
general_log_likelihood = cc.calc_log_likelihood
general_gradient = cc.calc_gradient
general_calc_probabilities = cc.calc_probabilities
general_hessian = cc.calc_hessian

def split_param_vec(beta, *args, **kwargs):
    """
    beta:       1D numpy array. All elements should by ints, floats, or
                longs. Should have 1 element for each utility 
                coefficient being estimated (i.e. num_features).
    ====================
    Returns:    tuple. None, None, beta. This function is merely for 
                compatibility with the other choice model files. It is also
                needed for the em-algorithm optimizer to work.
    """    
    return None, None, beta


def _mnl_utility_transform(systematic_utilities, *args, **kwargs):
    """
    systematic_utilities:   1D numpy array of the systematic utilities for each
                            each available alternative for each observation
    ====================
    Returns:   systematic_utilities[:, None]
    """    
    return systematic_utilities[:, np.newaxis]
    

def _mnl_transform_deriv_v(systematic_utilities, *args, **kwargs):
    """
    systematic_utilities:   1D numpy array of the systematic utilities for each
                            each available alternative for each observation
    ====================
    Returns:    np.diag(np.ones(systematic_utilities.shape[0])). Returns the 
                derivative of the transformation vector with respect to the 
                vector of systematic utilities.
    """        
    return diags(np.ones(systematic_utilities.shape[0]), 0, format='csr')
    
    
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


def _calc_neg_log_likelihood_and_neg_gradient(beta, design, 
                                              alt_IDs, alt_to_obs,
                                              alt_to_shapes,
                                              choice_vector,
                                              chosen_row_to_obs,
                                              utility_transform,
                                              block_matrix_idxs,
                                              ridge,
                                              dh_dv_mnl,
                                              *args):
    """
    beta:               1D numpy array. All elements should by ints, floats, or
                        longs. Should have 1 element for each utility 
                        coefficient being estimated (i.e. num_features).
                        
    design:             2D numpy array with one row per observation per 
                        available alternative. There should be one column per
                        utility coefficient being estimated. All elements 
                        should be ints, floats, or longs.
                        
    alt_IDs:            1D numpy array. All elements should be ints. There 
                        should be one row per obervation per available 
                        alternative for the given observation. Elements denote
                        the alternative corresponding to the given row of the
                        design matrix.
                        
    alt_to_obs:         2D numpy array with one row per observation per 
                        available alternative and one column per observation.
                        This matrix maps the rows of the design matrix to the 
                        unique observations (on the columns).
                        
    alt_to_shapes:      2D numpy array with one row per observation per 
                        available alternative and one column per possible
                        alternative. This matrix maps the rows of the design
                        matrix to the possible alternatives for this dataset.
                        
    choice_vector:      1D numpy array. All elements should be either ones or
                        zeros. There should be one row per observation per 
                        available alternative for the given observation. 
                        Elements denote the alternative which is chosen by the
                        given observation with a 1 and a zero otherwise.
                        
    chosen_row_to_obs:  2D numpy array with one row per observation per 
                        available alternative and one column per observation. 
                        This matrix indicates, for each observation (on the
                        columns), which rows of the design matrix were the 
                        realized outcome. Should have one and only one '1' in
                        each column. No row should have more than one '1' 
                        though it is okay if a row is all zeros.
                        
    utility_transform:  callable. Should accept a 1D array of systematic 
                        utility values, a 1D array of alternative IDs, and
                        miscellaneous args and kwargs. Should return a 1D array 
                        whose elements contain the appropriately transformed 
                        systematic utility values, based on the current model 
                        being evaluated.
                        
    block_matrix_idxs:  list. list of arrays. There will be one array per 
                        column in alt_to_obs. The array will note which rows 
                        correspond to which observations.
    
    ridge:              int, float, long, or None. Determines whether or not 
                        ridge regression is performed. If a float is passed, 
                        then that float determines the ridge penalty for the 
                        optimization. Default = None.
                        
    dh_dv_mnl:          callable. Must accept a 1D array of systematic 
                        utility values, a 1D array of alternative IDs, 
                        (shape parameters if there are any) and 
                        miscellaneous args and kwargs. Should return a 
                        2D array whose elements contain the 
                        derivative of the tranformed utility vector 
                        with respect to the vector of systematic 
                        utilities. The dimensions of the returned 
                        vector should be (design.shape[0], 
                        design.shape[0]).
    ====================
    Returns:            tuple. The first element is a float. The second element
                        is a 1D numpy array of shape (design.shape[1],). The
                        first element is the negative log-likelihood of this
                        model evaluated at the passed values of beta. The 
                        second element is the gradient of the negative log-
                        likelihood with respect to the vector of utility 
                        coefficients.
    """
    neg_log_likelihood = -1 * general_log_likelihood(beta,
                                                     design, 
                                                     alt_IDs,
                                                     alt_to_obs,
                                                     alt_to_shapes,
                                                     choice_vector,
                                                     utility_transform,
                                                     ridge=ridge)
                                                            
    neg_beta_gradient_vec = -1 * general_gradient(beta,
                                                  design,
                                                  alt_IDs,
                                                  alt_to_obs,
                                                  alt_to_shapes,
                                                  choice_vector,
                                                  utility_transform,
                                                  _mnl_transform_deriv_c,
                                                  dh_dv_mnl,
                                                  _mnl_transform_deriv_alpha,
                                                  None,
                                                  None,
                                                  ridge)
    
    return neg_log_likelihood, neg_beta_gradient_vec

    
def _calc_neg_hessian(beta,
                      design, 
                      alt_IDs,
                      alt_to_obs,
                      alt_to_shapes,
                      choice_vector,
                      chosen_row_to_obs,
                      utility_transform,
                      block_matrix_idxs,
                      ridge,
                      dh_dv_mnl,
                      *args):
    """
    beta:               1D numpy array. All elements should by ints, floats, or
                        longs. Should have 1 element for each utility 
                        coefficient being estimated (i.e. num_features).
                        
    design:             2D numpy array with one row per observation per 
                        available alternative. There should be one column per
                        utility coefficient being estimated. All elements 
                        should be ints, floats, or longs.
                        
    alt_IDs:            1D numpy array. All elements should be ints. There 
                        should be one row per obervation per available 
                        alternative for the given observation. Elements denote
                        the alternative corresponding to the given row of the
                        design matrix.
                        
    alt_to_obs:         2D numpy array with one row per observation per 
                        available alternative and one column per observation.
                        This matrix maps the rows of the design matrix to the 
                        unique observations (on the columns).
                        
    alt_to_shapes:      2D numpy array with one row per observation per 
                        available alternative and one column per possible
                        alternative. This matrix maps the rows of the design
                        matrix to the possible alternatives for this dataset.
                        
    choice_vector:      1D numpy array. All elements should be either ones or
                        zeros. There should be one row per observation per 
                        available alternative for the given observation. 
                        Elements denote the alternative which is chosen by the
                        given observation with a 1 and a zero otherwise.
                        
    chosen_row_to_obs:  2D numpy array with one row per observation per 
                        available alternative and one column per observation. 
                        This matrix indicates, for each observation (on the
                        columns), which rows of the design matrix were the 
                        realized outcome. Should have one and only one '1' in
                        each column. No row should have more than one '1' 
                        though it is okay if a row is all zeros.
                        
    utility_transform:  callable. Should accept a 1D array of systematic 
                        utility values, a 1D array of alternative IDs, and
                        miscellaneous args and kwargs. Should return a 1D array 
                        whose elements contain the appropriately transformed 
                        systematic utility values, based on the current model 
                        being evaluated.
                        
    block_matrix_idxs:  list. list of arrays. There will be one array per 
                        column in alt_to_obs. The array will note which rows 
                        correspond to which observations.
    
    ridge:              int, float, long, or None. Determines whether or not 
                        ridge regression is performed. If a float is passed, 
                        then that float determines the ridge penalty for the 
                        optimization. Default = None.
                        
    dh_dv_mnl:          callable. Must accept a 1D array of systematic 
                        utility values, a 1D array of alternative IDs, 
                        (shape parameters if there are any) and 
                        miscellaneous args and kwargs. Should return a 
                        2D array whose elements contain the 
                        derivative of the tranformed utility vector 
                        with respect to the vector of systematic 
                        utilities. The dimensions of the returned 
                        vector should be (design.shape[0], 
                        design.shape[0]).
    ====================
    Returns:            2D, square numpy array. The number of rows (and 
                        columns) will be desig.shape[1], i.e. the number of
                        utility coefficients in the model.
    """
    
    return -1 * general_hessian(beta,
                                design, 
                                alt_IDs,
                                alt_to_obs,
                                alt_to_shapes,
                                utility_transform,
                                _mnl_transform_deriv_c,
                                dh_dv_mnl,
                                _mnl_transform_deriv_alpha,
                                block_matrix_idxs,
                                None,
                                None,
                                ridge)

    
def _estimate(init_values, design_matrix, alt_id_vector, 
              choice_vector, alt_to_obs, alt_to_shapes,
              chosen_row_to_obs, print_results=True, 
              method='newton-cg', loss_tol=1e-06,
              gradient_tol=1e-06, maxiter=1000,
              m_method="cg", m_step_maxiter=1200,
              ridge=None, **kwargs):
    """
    init_vals:          1D numpy array of the initial values to start the 
                        optimizatin process with. There should be one value
                        for each utility coefficient and shape parameter being
                        estimated.
                        
    design_matrix:      2D numpy array with one row per observation per 
                        available alternative. There should be one column per
                        utility coefficient being estimated. All elements 
                        should be ints, floats, or longs.
                        
    alt_id_vector:      1D numpy array. All elements should be ints. There 
                        should be one row per observation per available 
                        alternative for the given observation. Elements denote 
                        the alternative corresponding to the given row of the 
                        design matrix.
                        
    choice_vector:      1D numpy array. All elements should be either ones or
                        zeros. There should be one row per observation per 
                        available alternative for the given observation. 
                        Elements denote the alternative which is chosen by the
                        given observation with a 1 and a zero otherwise.
    
    alt_to_obs:         2D numpy array with one row per observation per 
                        available alternative and one column per observation. 
                        This matrix maps the rows of the design matrix to the 
                        unique observations (on the columns).
                        
    alt_to_shapes:      2D numpy array with one row per observation per 
                        available alternative and one column per possible
                        alternative. This matrix maps the rows of the design
                        matrix to the possible alternatives for this dataset.
                        
    chosen_row_to_obs:  2D numpy array with one row per observation per 
                        available alternative and one column per observation. 
                        This matrix indicates, for each observation (on the
                        columns), which rows of the design matrix were the 
                        realized outcome. Should have one and only one '1' in
                        each column. No row should have more than one '1' 
                        though it is okay if a row is all zeros.
    
    print_res:          OPTIONAL. Bool. Determines whether the timing and
                        initial and final log likelihood results will be 
                        printed as they they are determined. Default = True.
                        
    method:             OPTIONAL. String. Should be a valid string which can
                        be passed to scipy.optimize.minimize. Determines the
                        optimization algorithm which is used for this problem.
                        Default = 'newton-cg'.
                    
    loss_tol:           OPTIONAL. Float. Determines the tolerance on the 
                        difference in objective function values from one 
                        iteration to the next which is needed to determine
                        convergence. Default = 1e-06.
                    
    gradient_tol:       OPTIONAL. Float. Determines the tolerance on the 
                        difference in gradient values from one 
                        iteration to the next which is needed to determine
                        convergence. Default = 1e-06.
                        
    maxiter:            OPTIONAL. Int. Determines the maximum number of 
                        iterations used by the optimizer. Default = 1000.
                        
    m_method:           OPTIONAL. String. Should be a valid string 
                        which can be passed to scipy.optimize.minimize.
                        Determines the optimization algorithm which is 
                        used for the M-step of the algorithm problem.
                        Default='cg'.
                        
    m_step_maxiter:     OPTIONAL. Int. Determines the maximum number of
                        iterations used by the optimizer within the 
                        M-step of the EM-algorithm.
                    
    ridge:              OPTIONAl. int, float, long, or None. Determines whether 
                        or not ridge regression is performed. If a scalar is 
                        passed, then that scalar determines the ridge penalty 
                        for the optimization. Default = None.
    ==================
    Returns:            tuple. Result dictionary, final log-likelihood, and
                        fitted probabilities of the chosen alternative.
    """
    
    # Make sure we have the correct dimensions for the initial parameter values
    try:    
        assert init_values.shape[0] == design_matrix.shape[1]
    except AssertionError as e:
        print("The initial values are of the wrong dimension")
        print("They should be of dimension {}".format(design_matrix.shape[1]))
        raise e
        
    # Make sure the ridge regression parameter is None or a real scalar
    try:
        assert ridge is None or isinstance(ridge, (int, float, long))
    except AssertionError as e:
        print("ridge should be None or an int, float, or long.")
        print("The passed value of ridge had type: {}".format(type(ridge)))
        raise e
           
    if print_results:
        # Print the log-likelihood at zero
        log_likelihood_at_zero = general_log_likelihood(
                                              np.zeros(design_matrix.shape[1]),
                                                                 design_matrix, 
                                                                 alt_id_vector,
                                                                    alt_to_obs,
                                                                 alt_to_shapes,
                                                                 choice_vector,
                                                        _mnl_utility_transform,
                                                                   ridge=ridge)
                                                                   
        print("Log-likelihood at zero: {:,.4f}".format(log_likelihood_at_zero))
        
        # Print the log-likelihood at the starting values
        initial_log_likelihood = general_log_likelihood(init_values,
                                                         design_matrix, 
                                                         alt_id_vector,
                                                         alt_to_obs,
                                                         alt_to_shapes,
                                                         choice_vector,
                                                        _mnl_utility_transform,
                                                         ridge=ridge)
        print("Initial Log-likelihood: {:,.4f}".format(initial_log_likelihood))
        sys.stdout.flush()
        
    # Get the block matrix indices for the hessian matrix. Do it outside the
    # iterative minimization process in order to minimize unnecessary 
    # computations
    block_matrix_indices = cc.create_matrix_block_indices(alt_to_obs)
    
    # Pre-calculate the derivative of the transformation vector with respect
    # to the vector of systematic utilities
    dh_dv = diags(np.ones(design_matrix.shape[0]), 0, format='csr')
    
    # Create a function to calculate dh_dv which will return the
    # pre-calculated result when called
    calc_dh_dv = lambda x, *args: dh_dv
    
    # Start timing the estimation process
    start_time = time.time()
    
    if method == "em":
        results = choice_model_em.naive_em_algorithm(init_values,
                                                     design_matrix,
                                                     alt_id_vector,
                                                     alt_to_obs,
                                                     alt_to_shapes,
                                                     choice_vector,
                                                     split_param_vec,
                                                     _mnl_utility_transform,
                                                     _mnl_transform_deriv_c,
                                                     calc_dh_dv,
                                                    _mnl_transform_deriv_alpha,
                                                     initial_log_likelihood,
                                                     maxiter=maxiter,
                                                 m_step_maxiter=m_step_maxiter,
                                                     ll_tol=loss_tol,
                                                     gradient_tol=gradient_tol,
                                                     m_method=m_method)
    else:
        results = minimize(_calc_neg_log_likelihood_and_neg_gradient,
                           init_values,
                           args = (design_matrix,
                                   alt_id_vector,                               
                                   alt_to_obs,
                                   alt_to_shapes,
                                   choice_vector,
                                   chosen_row_to_obs,
                                   _mnl_utility_transform,
                                   block_matrix_indices,
                                   ridge,
                                   calc_dh_dv), 
                           method = method,
                           jac = True, 
                           hess = _calc_neg_hessian,
                           tol = loss_tol,
                           options = {'gtol': gradient_tol,
                                      "maxiter": maxiter})
    
    # Calculate the final log-likelihood. Note the '-1' is because we minimized
    # the negative log-likelihood but we want the actual log-likelihood
    final_log_likelihood = -1 * results["fun"]
                   
    # Stop timing the estimation process and report the timing results
    end_time = time.time()
    if print_results:
        elapsed_sec = (end_time - start_time)
        elapsed_min = elapsed_sec /60.0
        if elapsed_min > 1.0:
            print("Estimation Time: {:.2f} minutes.".format(elapsed_min))
        else:
            print("Estimation Time: {:.2f} seconds.".format(elapsed_sec))
        print("Final log-likelihood: {:,.4f}".format(final_log_likelihood))
        sys.stdout.flush()
        
    # Calculate the predicted probabilities
    probability_results = general_calc_probabilities(results.x,
                                                     design_matrix,
                                                     alt_id_vector,
                                                     alt_to_obs,
                                                     alt_to_shapes,
                                                     _mnl_utility_transform,
                                        chosen_row_to_obs=chosen_row_to_obs,
                                                    return_long_probs=True)
    prob_of_chosen_alternatives, long_probs = probability_results
    
    # Calculate the model residuals
    residuals = choice_vector - long_probs
    
    # Calculate the observation specific chi-squared components
    chi_squared_terms = np.square(residuals) / long_probs
    individual_chi_squareds = alt_to_obs.T.dot(chi_squared_terms)
    
    # Store supplementary objects in the estimation results dict
    results["utility_coefs"] = results.x
    results["shape_params"] = None
    results["intercept_params"] = None
    
    # Store the log-likelihood at zero
    results["log_likelihood_null"] = log_likelihood_at_zero
    
    # Calculate and store the rho-squared and rho-bar-squared
    results["rho_squared"] = 1.0 - (final_log_likelihood /
                                    log_likelihood_at_zero)
    results["rho_bar_squared"] = 1.0 - ((final_log_likelihood - 
                                         results.x.shape[0]) / 
                                        log_likelihood_at_zero)
    
    # Calculate and store the final gradient
    results["final_gradient"] = general_gradient(results.x, 
                                                 design_matrix,
                                                 alt_id_vector, 
                                                 alt_to_obs,
                                                 alt_to_shapes,
                                                 choice_vector,
                                                 _mnl_utility_transform,
                                                 _mnl_transform_deriv_c,
                                                 calc_dh_dv,
                                                 _mnl_transform_deriv_alpha,
                                                 None,
                                                 None,
                                                 ridge)
    # Calculate and store the final hessian
    results["final_hessian"] = general_hessian(results.x,
                                               design_matrix, 
                                               alt_id_vector,
                                               alt_to_obs,
                                               alt_to_shapes,
                                               _mnl_utility_transform,
                                               _mnl_transform_deriv_c,
                                               calc_dh_dv,
                                               _mnl_transform_deriv_alpha,
                                               block_matrix_indices,
                                               None,
                                               None,
                                               ridge)
                                               
    # Calculate and store the final fischer information matrix
    results["fisher_info"] = cc.calc_fischer_info_matrix(results.x,
                                                         design_matrix,
                                                         alt_id_vector,
                                                         alt_to_obs,
                                                         alt_to_shapes,
                                                         choice_vector,
                                                        _mnl_utility_transform,
                                                        _mnl_transform_deriv_c,
                                                         calc_dh_dv,
                                                    _mnl_transform_deriv_alpha,
                                                         None,
                                                         None,
                                                         ridge)
                                                             
    return (results, final_log_likelihood, 
            prob_of_chosen_alternatives,
            long_probs, residuals,
            individual_chi_squareds)


class MNL(base_mcm.MNDC_Model):
    def __init__(self, data, alt_id_col, 
                 obs_id_col, choice_col,
                 specification,
                 names=None,
                 *args, **kwargs):
        """
        data:           string or pandas dataframe. If string, data should be 
                        an absolute or relative path to a CSV file containing 
                        the long format data for this choice model. Note long 
                        format is has one row per available alternative for 
                        each observation. If pandas dataframe, the dataframe 
                        should be the long format data for the choice model.
                        
        alt_id_col:     string. Should denote the column in data which contains
                        the alternative identifiers for each row.
                        
        obs_id_col:     string. Should denote the column in data which contains
                        the observation identifiers for each row.
                        
        choice_col:     string. Should denote the column in data which contains
                        the ones and zeros that denote whether or not the given
                        row corresponds to the chosen alternative for the given
                        individual.
                        
        specification:  OrderedDict. Keys are a proper subset of the columns
                        in long_form_df. Values are either a list or a single
                        string, "all_diff" or "all_same". If a list, the
                        elements should be:
                        1) single objects that are within the alternative ID 
                           column of long_form_df
                        2) lists of objects that are within the alternative
                           ID column of long_form_df.
                        For each single object in the list, a unique column
                        will be created (i.e. there will be a unique 
                        coefficient for that variable in the corresponding 
                        utility equation of the corresponding alternative). For
                        lists within the specification_dict values, a single 
                        column will be created for all the alternatives within 
                        iterable (i.e. there will be one common coefficient for
                        the variables in the iterable).
                        
        names:          OPTIONAL. OrderedDict. Should have the same keys as
                        specification_dict. For each key:
                        1) if the corresponding value in specification_dict is
                           "all_same", then there should be a single string as
                           the value in names.
                        2) if the corresponding value in specification_dict is
                           "all_diff", then there should be a list of strings 
                           as the value in names. There should be one string in
                           the value in names for each possible alternative.
                        3) if the corresponding value in specification_dict is
                           a list, then there should be a list of strings as
                           the value in names. There should be one string the
                           value in names per item in the value in 
                           specification_dict.
                        Default == None.                
        ====================
        Returns:        None.   
        """
        ##########
        # Print a helpful message for users who have included shape parameters
        # or shape names unneccessarily
        ##########
        msg_1 = "The Multinomial Logit Model has no shape parameters. "
        msg_2 = "shape_names and shape_ref_pos will be ignored if passed."
        shape_ignore_msg = msg_1 + msg_2
        
        for keyword in ["shape_names", "shape_ref_pos"]:
            if keyword in kwargs and kwargs[keyword] is not None:
                print(shape_ignore_msg)
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
                                  model_type="Multinomial Logit Model")
                                  
        # Store the utility transform function
        self.utility_transform = _mnl_utility_transform
                                  
        return None


    def fit_mle(self, init_vals, 
                print_res=True, method="BFGS",
                loss_tol=1e-06, gradient_tol=1e-06,
                maxiter=1000, m_method="cg",
                m_step_maxiter=1200,
                ridge=None, **kwargs):
        """
        init_vals:      1D numpy array of the initial values to start the 
                        optimizatin process with. There should be one value
                        for each utility coefficient and shape parameter being
                        estimated.
                        
        print_res:      OPTIONAL. Bool. Determines whether the timing and
                        initial and final log likelihood results will be 
                        printed as they they are determined.
                        
        method:         OPTIONAL. String. Should be a valid string which can
                        be passed to scipy.optimize.minimize. Determines the
                        optimization algorithm which is used for this problem.
                        
        loss_tol:       OPTIONAL. Float. Determines the tolerance on the 
                        difference in objective function values from one 
                        iteration to the next which is needed to determine
                        convergence. Default = 1e-06.
                        
        gradient_tol:   OPTIONAL. Float. Determines the tolerance on the 
                        difference in gradient values from one 
                        iteration to the next which is needed to determine
                        convergence. Default = 1e-06.
                        
        m_method:       OPTIONAL. String. Should be a valid string 
                        which can be passed to scipy.optimize.minimize.
                        Determines the optimization algorithm which is 
                        used for the M-step of the algorithm problem.
                        Default='cg'.
                        
        m_step_maxiter: OPTIONAL. Int. Determines the maximum number of
                        iterations used by the optimizer within the 
                        M-step of the EM-algorithm.
                        
        ridge:          OPTIONAl. int, float, long, or None. Determines whether 
                        or not ridge regression is performed. If a float is 
                        passed, then that float determines the ridge penalty 
                        for the optimization. Default = None.
        ====================
        Returns:        None. Results of the estimation process are saved to
                        the model instance.
        """
        # Check integrity of passed arguments
        kwargs_to_be_ignored = ["init_shapes", "init_intercepts", "init_index"]
        if any([x in kwargs for x in kwargs_to_be_ignored]):
            msg = "MNL model does not use of any of the following kwargs:\n{}"
            msg_2 = "Remove such kwargs and pass a single init_vals argument"
            raise ValueError(msg.format(kwargs_to_be_ignored) + msg_2)
            
        # Store the optimization method
        self.optimization_method = method
        
        # Store the ridge parameter
        self.ridge_param = ridge
        
        if ridge is not None:
            msg = "NOTE: An L2-penalized regression is being performed. The "
            msg_2 = "reported standard errors and robust standard errors "
            msg_3 = "***WILL BE INCORRECT***."
            
            print("=" * 30)
            print(msg + msg_2 + msg_3)
            print("=" * 30)
            print("\n")
        
        # Construct the mappings from alternatives to observations and from
        # chosen alternatives to observations
        alt_to_obs, alt_to_shapes, chosen_row_to_obs =\
                                                    self.get_mappings_for_fit()
        
        # Get the estimation results
        estimation_res = _estimate(init_vals,
                                   self.design,
                                   self.alt_IDs, 
                                   self.choices,
                                   alt_to_obs,
                                   alt_to_shapes,
                                   chosen_row_to_obs,                                   
                                   print_results=print_res,
                                   method=method,
                                   loss_tol=loss_tol,
                                   gradient_tol=gradient_tol,
                                   maxiter=maxiter,
                                   m_method=m_method,
                                   m_step_maxiter=m_step_maxiter,
                                   ridge=ridge)
        
        # Store the estimation results
        self.store_fit_results(estimation_res)
                                 
        return None

