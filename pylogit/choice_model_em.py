# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:56:19 2016

@name:      EM Algorithm for Logit Type Conditional Choice Models
@author:    Timothy Brathwaite
@summary:   Contains generic functions necessary for estimating logit-type
            conditional choice models using the EM Algorithm.
"""

import time, sys

import numpy as np
from scipy.optimize import minimize
from scipy.misc import logsumexp
from scipy.sparse import diags, hstack

from choice_calcs import calc_log_likelihood, calc_gradient

# Define the boundary values which are not to be exceeded ducing computation
min_exponent_val = -700
max_exponent_val = 700

max_comp_value = 1e300
min_comp_value = 1e-300


def calc_expected_weight(psi_vec):
    """
    psi_vec:    1D numpy array with shape (num_obs,).
                Each element in the array should be a
                float.
    ====================
    Returns:    1D numpy array with shape (num_obs,). 
                Each element will be a float denoting the
                expected value of the weights for each
                observation, assuming one trial per
                observation, and given the value of psi
                for each observation.
    """
    return np.tanh(psi_vec / 2.0) / (2 * psi_vec)
    
    
def calc_psi(h_vec,
             chosen_row_indices,
             rejected_row_indices,
             rejected_row_to_obs,
             rejected_row_to_alts,
             wide_unavailable_indices):
    """
    h_vec:                      1D numpy array with 1 element for each 
                                available alternative per observation.
                                Each element contains h_{i, j} where
                                h_{i, j} = h(x_{ij}, theta).  
                                h_vec.shape[0] should equal
                                (rejected_row_indices.shape[0] +
                                 chosen_row_indices.shape[0], )
                                 
    chosen_row_indices:         1D numpy array with 1 element per 
                                observation. Each element in the array 
                                should denote the index position in h_vec
                                that correpsonds to the chosen alternative
                                for the given observation.
                                
    rejected_row_indices:       1D numpy array with 1 element per rejected
                                alternative per observation. Each element
                                in the array should denote the index 
                                position in h_vec that correpsonds to the
                                given rejected alternative for the given
                                observation.
                                
    rejected_row_to_obs:        2D scipy compressed sparse row matrix.
                                There should be one row per rejected 
                                alternative per observation and one column
                                per observation. The entries should be
                                only zeros and ones. The matrix should
                                indicate the observation corresponding to
                                each row.
                                
    rejected_row_to_alts:       2D scipy compressed sparse row matrix.
                                There should be one row per rejected 
                                alternative per observation and one column
                                per alternative in the overal dataset. The
                                entries should be only zeros and ones. The
                                matrix should indicate the alternative 
                                corresponding to each row.
                                
    wide_unavailable_indices:   tuple of length 2. Each element of the
                                tuple should be an array comprised of the
                                indices of a 2D index that should be 
                                selected to determine which entries 
                                correspond alternatives that are 
                                unavailable to a given observation. Should
                                be the output of using np.where() on the
                                mapping matrix obs_to_alternatives.
    ====================
    Returns:                    1D numpy array of shape (num_obs,).
                                Each element contains the value of psi,
                                h_{i, chosen} - logsumexp(h_{i, rejected}).
                                where h_{i, j} = h(x_{ij}, \theta) and
                                h_{i, rejected} is a vector of such values
                                for all of the rejected alternatives for
                                observation i.
    """
    ##########
    # Select the entries in h_vec that correspond to 
    # chosen and rejected alternatives
    ##########
    rejected_h = h_vec[rejected_row_indices]
    chosen_h = h_vec[chosen_row_indices]
    
    ##########
    # Create the wide format version of rejected_h
    ##########
    # Note that rejected_row_to_alts is a sparse matrix
    # Element-wise multiplication is therefore performed
    # using rejected_row_to_obs.multiply.
    # Also, rejected_row_to_alts is a sparse matrix,
    # so matrix multiplication is performed using *
    wide_rejected_h = (rejected_row_to_obs.multiply(rejected_h[:, None]).T *
                       rejected_row_to_alts)
    
    ##########
    # Fill the 'unavailable' entries of wide_rejected_h with -inf
    ##########
    wide_rejected_h[wide_unavailable_indices] = -np.inf
    
    ##########
    # Calculate and return psi
    ##########
    return chosen_h - logsumexp(wide_rejected_h, axis=1)
    
    
def calc_complete_data_log_likelihood(w_vec, psi_vec, base_diag):
    """
    w_vec:      1D numpy array. All elements should be ints,
                floats, or longs. Each element should be the
                expectation of the hidden weight variable,
                given the observed data and most recent 
                parameter updates. The shape of the array
                should be (num_observations, ).
                
    psi_vec:    1D numpy array. All elements should be ints,
                floats, or longs. Each element should be 
                numerator in the binary representation of
                the multinomial choice model. The shape of 
                the array should be (num_observations, ).
                
    base_diag:  2D compressed sparse row matrix. The shape 
                should be (num_observations, num_observations).
                All elements should be ints, floats, or longs.
                The matrix should be a diagonal matrix.
    ====================
    Returns:    scalar. An int, float, or long. The returned
                value will be the complete data log-likelihood
                at the current values of theta, the vector of
                parameters being estimated.
    """
    # Place the updated values from w_vec, i.e. the expectaion of
    # the hidden variables given all other information so far, into
    # the diagonal matrix for W^t
    base_diag.data = w_vec
    
    # Compute the various terms in the complete data log-likelihood
    term_1 = psi_vec.dot(base_diag.dot(psi_vec))
    term_2 = psi_vec.sum()
    
    return 0.5 * (term_2 - term_1)
    
    
def calc_psi_gradient_theta(
                        design,                        
                        sys_utilities,
                        alt_IDs,
                        alt_to_shapes,
                        h_vec,
                        chosen_row_indices,
                        rejected_row_indices,
                        rejected_row_to_obs,
                        transform_first_deriv_c,
                        transform_first_deriv_v,
                        transform_deriv_alpha,
                        intercept_params,
                        shape_params):
    """
    design:                     bla bla.
    
    sys_utilities:              1D numpy array. Should == 
                                design_matrix.dot(beta).
    
    alt_IDs:                    1D numpy array. All elements should be ints. 
                                There should be one row per obervation per 
                                available alternative for the given 
                                observation. Elements denote the alternative 
                                corresponding to the given row of the design 
                                matrix.
                        
    alt_to_shapes:              2D numpy array with one row per observation per 
                                available alternative and one column per 
                                possible alternative. This matrix maps the rows
                                of the design matrix to the possible 
                                alternatives for this dataset.
    
    h_vec:                      1D numpy array with 1 element for each 
                                available alternative per observation.
                                Each element contains h_{i, j} where
                                h_{i, j} = h(x_{ij}, theta).  
                                h_vec.shape[0] should equal
                                (rejected_row_indices.shape[0] +
                                 chosen_row_indices.shape[0], ). Note that 
                                h_vec should have already been protected 
                                against underflow and overflow.
                                 
    chosen_row_indices:         1D numpy array with 1 element per 
                                observation. Each element in the array 
                                should denote the index position in h_vec
                                that correpsonds to the chosen alternative
                                for the given observation.
                                
    rejected_row_indices:       1D numpy array with 1 element per rejected
                                alternative per observation. Each element
                                in the array should denote the index 
                                position in h_vec that correpsonds to the
                                given rejected alternative for the given
                                observation.
                                
    rejected_row_to_obs:        2D scipy compressed sparse row matrix.
                                There should be one row per rejected 
                                alternative per observation and one column
                                per observation. The entries should be
                                only zeros and ones. The matrix should
                                indicate the observation corresponding to
                                each row.
                                
    transform_first_deriv_c:    callable. Must accept a 1D array of systematic 
                                utility values, a 1D array of alternative IDs,
                                the alt_to_shapes array, (shape parameters if
                                there are any) and miscellaneous args and 
                                kwargs. Should return a 2D array whose elements 
                                contain the derivative of the tranformed 
                                utility vector with respect to the vector of 
                                shape parameters. The dimensions of the 
                                returned vector should be (design.shape[0], 
                                num_alternatives). If there are no shape 
                                parameters, the callable should return None.
                        
    transform_first_deriv_v:    callable. Must accept a 1D array of systematic 
                                utility values, a 1D array of alternative IDs, 
                                (shape parameters if there are any) and 
                                miscellaneous args and kwargs. Should return a 
                                2D array whose elements contain the 
                                derivative of the tranformed utility vector 
                                with respect to the vector of systematic 
                                utilities. The dimensions of the returned 
                                vector should be (design.shape[0], 
                                design.shape[0]).
                                
    transform_deriv_alpha:      callable. Must accept a 1D array of systematic
                                utility values, a 1D array of alternative IDs,
                                the alt_to_shapes array, (intercept parameters
                                if there are any) and miscellaneous args and
                                kwargs. Should return a 2D array whose elements
                                contain the derivative of the tranformed 
                                utility vector with respect to the vector of 
                                shape parameters. The dimensions of the 
                                returned vector should be (design.shape[0],
                                num_alternatives - 1). If there are no 
                                intercept parameters, the callable should '
                                return None.
                                
    intercept_params:           None or 1D numpy array. If an array, each 
                                element should be an int, float, or long. For 
                                identifiability, there should be J- 1 elements 
                                where J is the total number of observed 
                                alternatives for this dataset. Default == None.
                                
    shape_params:               None or 1D numpy array. If an array, each 
                                element should be an int, float, or long. There 
                                should be one value per shape parameter of the 
                                model being used. Default == None.
    ====================
    Returns:                    2D numpy array. The derivative of the psi 
                                vector with respect to theta. All elements will 
                                be ints, floats, or longs. The shape of the 
                                returned array will be 
                                (chosen_row_indices.shape[0], theta.shape[0]).
    """
    ##########
    # Calculate dh_dtheta
    ##########
    # Differentiate the transformed utilities with respect to the shape params
    # Note that dh_dc should be a sparse matrix
    dh_dc = transform_first_deriv_c(sys_utilities, alt_IDs, 
                                    alt_to_shapes, shape_params)
    # Differentiate the transformed utilities by the intercept params
    # Note that dh_d_alpha should be a sparse matrix
    dh_d_alpha = transform_deriv_alpha(sys_utilities, alt_IDs,
                                       alt_to_shapes, intercept_params)
    # Differentiate the transformed utilities with respect to the systematic 
    # utilities. Note that dh_dv should be a sparse matrix
    dh_dv = transform_first_deriv_v(sys_utilities, alt_IDs, 
                                    alt_to_shapes, shape_params)
    # Differentiate the transformed utilities with respect to the utility 
    # coefficients. Note that dh_db should be a dense **matrix**, not a dense
    # 2D array. This is because the dot product of a 2D scipy sparse array and
    # a 2D dense numpy array yields a 2D dense numpy matrix
    dh_db = dh_dv.dot(design)
    
    # Store the necesary sparse matrices in a list
    necessary_sparses = [x for x in [dh_dc, dh_d_alpha] if x is not None]
    if len(necessary_sparses) > 1:
        non_beta_derivs = hstack(necessary_sparses, format='csr').toarray()
    elif len(necessary_sparses) == 1:
        non_beta_derivs = necessary_sparses[0].toarray()
    else:
        non_beta_derivs = None
    
    # Create the final dh_d_theta array
    if non_beta_derivs:
        dh_d_theta = np.concatenate((non_beta_derivs,
                                     dh_db), axis=1)
    else:
        dh_d_theta = dh_db
    # Make sure dh_d_theta is a numpy array
    dh_d_theta = np.asarray(dh_d_theta)
    
    ##########
    # Partition the rows of dh_d_theta into chosen and rejected rows
    ##########
    chosen_dh_d_theta = dh_d_theta[chosen_row_indices, :]
    rejected_dh_d_theta = dh_d_theta[rejected_row_indices, :]
    
    ##########
    # Calculate, for each rejected alternative for each person, the 
    # probability of being chosen, given that the actual chosen 
    # alternative is excluded from consideration.
    ##########
    # Exponentiate the transformed utilities                                      
    long_rejected_exponentials = np.exp(h_vec[rejected_row_indices])
    # Calcluate the sum of the exponents of the rejected transformations
    # of the index, i.e. the denominator in the probability of the rejected
    # alternatives being chosen, excluding the actually chosen choice from
    # consideration.
    individual_denominators = rejected_row_to_obs.transpose().dot(
                                                 long_rejected_exponentials)
    long_rejected_denom = rejected_row_to_obs.dot(individual_denominators)
    # Calculate the rejected probabilities, i.e. the probability of being
    # the second best alternative.
    long_rejected_probs = (long_rejected_exponentials /
                           long_rejected_denom)
    
    ##########
    # Make sure long_rejected_probs is a column vector
    ##########
    reject_probs_shape = long_rejected_probs.shape
    if len(reject_probs_shape) == 2 and reject_probs_shape[1] == 1:
        pass
    else:
        assert len(reject_probs_shape) == 1
        long_rejected_probs = long_rejected_probs[:, None]
        
    ##########
    # Element-wise multiply long_rejected_probs by rejected_dh_d_theta
    ##########
    # Weight each of the derivatives of h(x_i,j') by the probability
    # of being the second best alternative behind the chosen alternative
    product_1 = rejected_dh_d_theta * long_rejected_probs
    # For each observation, sum the weighted d_hi_d_theta across all
    # rejected alternatives
    product_2 = rejected_row_to_obs.transpose().dot(product_1)
    
    assert isinstance(product_2, np.ndarray)
    
    # Return the desired derivative of psi with respect to theta
    return chosen_dh_d_theta - product_2
    
    
def calc_gradient_complete_data_log_likelihood(w_vec, psi_vec, d_psi_d_theta):
    """
    w_vec:          1D numpy array. All elements should be ints,
                    floats, or longs. Each element should be the
                    expectation of the hidden weight variable,
                    given the observed data and most recent 
                    parameter updates. The shape of the array
                    should be (num_observations, ).
                
    psi_vec:        1D numpy array. All elements should be ints,
                    floats, or longs. Each element should be 
                    numerator in the binary representation of
                    the multinomial choice model. The shape of 
                    the array should be (num_observations, ).
                
    d_psi_d_theta:  2D numpy array. The shape should be
                    (num_observations, theta.shape[0]). All
                    elements should be ints, floats, or longs.
                    This array should represent the derivative
                    of psi_vec with respect to theta.
    ====================
    Returns:        1D numpy array. All elements will be of type
                    int, float, or long. The returned array will
                    be the gradient of the complete data 
                    log-likelihood at the current value of theta.
    """
    term_1 = (-1 * psi_vec * w_vec + 0.5)[None, :]
    return term_1.dot(d_psi_d_theta).ravel()
    
    
def calc_hessian_complete_data_log_likelihood(w_vec, d_psi_d_theta):
    """
    w_vec:          1D numpy array. All elements should be ints,
                    floats, or longs. Each element should be the
                    expectation of the hidden weight variable,
                    given the observed data and most recent 
                    parameter updates. The shape of the array
                    should be (num_observations, ).
                
    d_psi_d_theta:  2D numpy array. The shape should be
                    (num_observations, theta.shape[0]). All
                    elements should be ints, floats, or longs.
                    This array should represent the derivative
                    of psi_vec with respect to theta.
    ====================
    Returns:        2D numpy array. All elements will be of type
                    int, float, or long. The returned array will
                    be the hessian of the complete data 
                    log-likelihood at the current value of theta.
    """
    return -1 * d_psi_d_theta.T.dot(w_vec[:, None] * d_psi_d_theta)
    
    
def calc_neg_complete_data_log_likelihood_and_gradient(theta,
                                                       w_vec,
                                                       design,
                                                       alt_IDs,
                                                       alt_to_shapes,
                                                       splitting_func,
                                                       utility_transform,
                                                       chosen_row_indices,
                                                       rejected_row_indices,
                                                       rejected_row_to_obs,
                                                       rejected_row_to_alts,
                                                       wide_unavailable_idxs,
                                                       transform_first_deriv_c,
                                                       transform_first_deriv_v,
                                                       transform_deriv_alpha,
                                                       base_diag):
    """
    theta:                      1D numpy array of the initial values to 
                                start the optimizatin process with. There
                                should be one value for each utility 
                                coefficient, intercept parameter, and shape
                                parameter being estimated.
    
    w_vec:                      1D numpy array. All elements should be ints,
                                floats, or longs. Each element should be the
                                expectation of the hidden weight variable,
                                given the observed data and most recent 
                                parameter updates. The shape of the array
                                should be (num_observations, ).
                                
    design:                     2D numpy array with one row per observation  
                                per available alternative. There should be
                                one column per utility coefficient being 
                                estimated. All elements should be ints, 
                                floats, or longs.
                                
    alt_IDs:                    1D numpy array. All elements should be ints. 
                                There should be one row per obervation per 
                                available alternative for the given 
                                observation. Elements denote the alternative 
                                corresponding to the given row of the design 
                                matrix.
                                
    alt_to_shapes:              2D numpy array with one row per observation 
                                per available alternative and one column per
                                possible alternative. This matrix maps the 
                                rows of the design matrix to the possible 
                                alternatives for this dataset.
                                
    utility_transform:          callable. Should accept a 1D array of 
                                systematic utility values, a 1D array of 
                                alternative IDs, and miscellaneous args and 
                                kwargs. Should return a 1D array whose elements 
                                contain the appropriately transformed 
                                systematic utility values, based on the current 
                                model being evaluated.
                                
    splitting_func:             callable. Should accept in the following order:
                                a 1D array of parameter values being estimated,
                                a 2D compressed sparse row matrix that maps the
                                rows of the design matrix to the alternatives 
                                in this dataset, and the design matrix. It 
                                should return a tuple with three elements: the 
                                array of shape parameters or None, the array of 
                                'outside' intercepts or None, and the array of 
                                index coefficients.
                                
    chosen_row_indices:         1D numpy array with 1 element per 
                                observation. Each element in the array 
                                should denote the index position in h_vec
                                that correpsonds to the chosen alternative
                                for the given observation.
                                
    rejected_row_indices:       1D numpy array with 1 element per rejected
                                alternative per observation. Each element
                                in the array should denote the index 
                                position in h_vec that correpsonds to the
                                given rejected alternative for the given
                                observation.
                                
    rejected_row_to_obs:        2D scipy compressed sparse row matrix.
                                There should be one row per rejected 
                                alternative per observation and one column
                                per observation. The entries should be
                                only zeros and ones. The matrix should
                                indicate the observation corresponding to
                                each row.
                                
    rejected_row_to_alts:       2D scipy compressed sparse row matrix.
                                There should be one row per rejected 
                                alternative per observation and one column
                                per alternative in the overal dataset. The
                                entries should be only zeros and ones. The
                                matrix should indicate the alternative 
                                corresponding to each row.
                                
    wide_unavailable_idxs:      tuple of length 2. Each element of the
                                tuple should be an array comprised of the
                                indices of a 2D index that should be 
                                selected to determine which entries 
                                correspond alternatives that are 
                                unavailable to a given observation. Should
                                be the output of using np.where() on the
                                mapping matrix obs_to_alternatives.
    
    transform_first_deriv_c:    callable. Must accept a 1D array of systematic 
                                utility values, a 1D array of alternative IDs,
                                the alt_to_shapes array, (shape parameters if
                                there are any) and miscellaneous args and 
                                kwargs. Should return a 2D array whose elements 
                                contain the derivative of the tranformed 
                                utility vector with respect to the vector of 
                                shape parameters. The dimensions of the 
                                returned vector should be (design.shape[0], 
                                num_alternatives). If there are no shape 
                                parameters, the callable should return None.
                                
    transform_first_deriv_v:    callable. Must accept a 1D array of systematic 
                                utility values, a 1D array of alternative IDs, 
                                (shape parameters if there are any) and 
                                miscellaneous args and kwargs. Should return a 
                                2D array whose elements contain the 
                                derivative of the tranformed utility vector 
                                with respect to the vector of systematic 
                                utilities. The dimensions of the returned 
                                vector should be (design.shape[0], 
                                design.shape[0]).
                                
    transform_deriv_alpha:      callable. Must accept a 1D array of systematic
                                utility values, a 1D array of alternative IDs,
                                the alt_to_shapes array, (intercept parameters
                                if there are any) and miscellaneous args and
                                kwargs. Should return a 2D array whose elements
                                contain the derivative of the tranformed 
                                utility vector with respect to the vector of 
                                shape parameters. The dimensions of the 
                                returned vector should be (design.shape[0],
                                num_alternatives - 1). If there are no 
                                intercept parameters, the callable should '
                                return None.
                                
    base_diag:                  2D compressed sparse row diagonal matrix. 
                                Should have shape (num_obs, num_obs). The data 
                                inside this matrix will continually be 
                                overwritten to calculate the complete data
                                log-likelihood.
    ==============================
    Returns:                    tuple. The first element contains the negative
                                log-likelihood. The second element contains the
                                negative gradient of the complete data log-
                                likelihood with respect to theta, the array of
                                parameters being estimated.
    """
    # Separate the shape, intercept, and beta parameters
    shape_vec, intercept_vec, coefficient_vec = splitting_func(theta, 
                                                               alt_to_shapes,
                                                               design)
    
    # Calculate the systematic utility for each alternative for each individual
    sys_utilities = design.dot(coefficient_vec)
    
    # Calculate the probability from the transformed utilities
    # The transformed utilities will be of shape (num_rows, 1)
    transformed_utilities = utility_transform(sys_utilities,
                                              alt_IDs,
                                              alt_to_shapes,
                                              shape_vec,
                                              intercept_vec).ravel()
    
    # The following commands are to guard against numeric under/over-flow
    transformed_utilities[transformed_utilities < min_exponent_val] =\
                                                               min_exponent_val
    transformed_utilities[transformed_utilities > max_exponent_val] =\
                                                               max_exponent_val
    # Calculate the Psi vector
    psi_vec = calc_psi(transformed_utilities,
                       chosen_row_indices,
                       rejected_row_indices,
                       rejected_row_to_obs,
                       rejected_row_to_alts,
                       wide_unavailable_idxs)
    
    # Calculate the complete data log_likelihood based on
    # w_vec, psi_vec, and base_diag
    complete_log_likelihood = calc_complete_data_log_likelihood(w_vec,
                                                                psi_vec,
                                                                base_diag)
    
    # Calculate d_psi_d_theta
    d_psi_d_theta = calc_psi_gradient_theta(design,                        
                                            sys_utilities,
                                            alt_IDs,
                                            alt_to_shapes,
                                            transformed_utilities,
                                            chosen_row_indices,
                                            rejected_row_indices,
                                            rejected_row_to_obs,
                                            transform_first_deriv_c,
                                            transform_first_deriv_v,
                                            transform_deriv_alpha,
                                            intercept_vec,
                                            shape_vec)
    
    # Calculate the gradient of the complete data log-likelihood
    # given w_vec, psi_vec, and d_psi_d_theta
    complete_gradient = calc_gradient_complete_data_log_likelihood(w_vec,
                                                                   psi_vec,
                                                             d_psi_d_theta)
    
    return -1 * complete_log_likelihood, -1 * complete_gradient
    
    
def calc_neg_hessian_complete_data_log_likelihood(theta,
                                                  w_vec,
                                                  design,
                                                  alt_IDs,
                                                  alt_to_shapes,
                                                  splitting_func,
                                                  utility_transform,
                                                  chosen_row_indices,
                                                  rejected_row_indices,
                                                  rejected_row_to_obs,
                                                  rejected_row_to_alts,
                                                  wide_unavailable_idxs,
                                                  transform_first_deriv_c,
                                                  transform_first_deriv_v,
                                                  transform_deriv_alpha,
                                                  base_diag):
    """
    theta:                      1D numpy array. All elements should be ints,
                                floats, or longs. There should be one value 
                                for each utility coefficient, intercept 
                                parameter, and shape parameter being estimated.
    
    w_vec:                      1D numpy array. All elements should be ints,
                                floats, or longs. Each element should be the
                                expectation of the hidden weight variable,
                                given the observed data and most recent 
                                parameter updates. The shape of the array
                                should be (num_observations, ).
                                
    design:                     2D numpy array with one row per observation  
                                per available alternative. There should be
                                one column per utility coefficient being 
                                estimated. All elements should be ints, 
                                floats, or longs.
                                
    alt_IDs:                    1D numpy array. All elements should be ints. 
                                There should be one row per obervation per 
                                available alternative for the given 
                                observation. Elements denote the alternative 
                                corresponding to the given row of the design 
                                matrix.
                                
    alt_to_shapes:              2D numpy array with one row per observation 
                                per available alternative and one column per
                                possible alternative. This matrix maps the 
                                rows of the design matrix to the possible 
                                alternatives for this dataset.
                                
    utility_transform:          callable. Should accept a 1D array of 
                                systematic utility values, a 1D array of 
                                alternative IDs, and miscellaneous args and 
                                kwargs. Should return a 1D array whose elements 
                                contain the appropriately transformed 
                                systematic utility values, based on the current 
                                model being evaluated.
                                
    splitting_func:             callable. Should accept in the following order:
                                a 1D array of parameter values being estimated,
                                a 2D compressed sparse row matrix that maps the
                                rows of the design matrix to the alternatives 
                                in this dataset, and the design matrix. It 
                                should return a tuple with three elements: the 
                                array of shape parameters or None, the array of 
                                'outside' intercepts or None, and the array of 
                                index coefficients.
                                
    chosen_row_indices:         1D numpy array with 1 element per 
                                observation. Each element in the array 
                                should denote the index position in h_vec
                                that correpsonds to the chosen alternative
                                for the given observation.
                                
    rejected_row_indices:       1D numpy array with 1 element per rejected
                                alternative per observation. Each element
                                in the array should denote the index 
                                position in h_vec that correpsonds to the
                                given rejected alternative for the given
                                observation.
                                
    rejected_row_to_obs:        2D scipy compressed sparse row matrix.
                                There should be one row per rejected 
                                alternative per observation and one column
                                per observation. The entries should be
                                only zeros and ones. The matrix should
                                indicate the observation corresponding to
                                each row.
                                
    rejected_row_to_alts:       2D scipy compressed sparse row matrix.
                                There should be one row per rejected 
                                alternative per observation and one column
                                per alternative in the overal dataset. The
                                entries should be only zeros and ones. The
                                matrix should indicate the alternative 
                                corresponding to each row.
                                
    wide_unavailable_idxs:      tuple of length 2. Each element of the
                                tuple should be an array comprised of the
                                indices of a 2D index that should be 
                                selected to determine which entries 
                                correspond alternatives that are 
                                unavailable to a given observation. Should
                                be the output of using np.where() on the
                                mapping matrix obs_to_alternatives.
    
    transform_first_deriv_c:    callable. Must accept a 1D array of systematic 
                                utility values, a 1D array of alternative IDs,
                                the alt_to_shapes array, (shape parameters if
                                there are any) and miscellaneous args and 
                                kwargs. Should return a 2D array whose elements 
                                contain the derivative of the tranformed 
                                utility vector with respect to the vector of 
                                shape parameters. The dimensions of the 
                                returned vector should be (design.shape[0], 
                                num_alternatives). If there are no shape 
                                parameters, the callable should return None.
                                
    transform_first_deriv_v:    callable. Must accept a 1D array of systematic 
                                utility values, a 1D array of alternative IDs, 
                                (shape parameters if there are any) and 
                                miscellaneous args and kwargs. Should return a 
                                2D array whose elements contain the 
                                derivative of the tranformed utility vector 
                                with respect to the vector of systematic 
                                utilities. The dimensions of the returned 
                                vector should be (design.shape[0], 
                                design.shape[0]).
                                
    transform_deriv_alpha:      callable. Must accept a 1D array of systematic
                                utility values, a 1D array of alternative IDs,
                                the alt_to_shapes array, (intercept parameters
                                if there are any) and miscellaneous args and
                                kwargs. Should return a 2D array whose elements
                                contain the derivative of the tranformed 
                                utility vector with respect to the vector of 
                                shape parameters. The dimensions of the 
                                returned vector should be (design.shape[0],
                                num_alternatives - 1). If there are no 
                                intercept parameters, the callable should '
                                return None.
                                
    base_diag:                  2D compressed sparse row diagonal matrix. Should
                                have shape (num_obs, num_obs). The data inside
                                this matrix will continually be overwritten to
                                calculate the complete data log-likelihood.
    ==============================
    Returns:                    2D numpy array. The negative hessian of the
                                complete data log-likelihood.
    """
    # Separate the shape, intercept, and beta parameters
    shape_vec, intercept_vec, coefficient_vec = splitting_func(theta, 
                                                               alt_to_shapes,
                                                               design)
    
    # Calculate the systematic utility for each alternative for each individual
    sys_utilities = design.dot(coefficient_vec)
    
    # Calculate the probability from the transformed utilities
    # The transformed utilities will be of shape (num_rows, 1)
    transformed_utilities = utility_transform(sys_utilities,
                                              alt_IDs,
                                              alt_to_shapes,
                                              shape_vec,
                                              intercept_vec).ravel()
    
    # The following commands are to guard against numeric under/over-flow
    transformed_utilities[transformed_utilities < min_exponent_val] =\
                                                               min_exponent_val
    transformed_utilities[transformed_utilities > max_exponent_val] =\
                                                               max_exponent_val
    
    # Calculate d_psi_d_theta
    d_psi_d_theta = calc_psi_gradient_theta(design,                        
                                            sys_utilities,
                                            alt_IDs,
                                            alt_to_shapes,
                                            transformed_utilities,
                                            chosen_row_indices,
                                            rejected_row_indices,
                                            rejected_row_to_obs,
                                            transform_first_deriv_c,
                                            transform_first_deriv_v,
                                            transform_deriv_alpha,
                                            intercept_vec,
                                            shape_vec)
    
    # Calculate the gradient of the complete data log-likelihood
    # given w_vec, psi_vec, and d_psi_d_theta
    complete_hessian = calc_hessian_complete_data_log_likelihood(w_vec,
                                                                 d_psi_d_theta)
    
    return -1 * complete_hessian
    
    
def naive_em_algorithm(theta,
                       design,
                       alt_IDs,
                       alt_to_obs,
                       alt_to_shapes,
                       choice_vector,
                       splitting_func,
                       utility_transform,
                       transform_first_deriv_c,
                       transform_first_deriv_v,
                       transform_deriv_alpha,
                       init_log_likelihood,
                       maxiter=1200,
                       m_step_maxiter=1200,
                       ll_tol=1e-6,
                       gradient_tol=1e-6,
                       m_method="cg",
                       **kwargs):
    """
    theta:                      1D numpy array of the initial values to 
                                start the optimizatin process with. There
                                should be one value for each utility 
                                coefficient, intercept parameter, and shape
                                parameter being estimated.
    
    design:                     2D numpy array with one row per observation  
                                per available alternative. There should be
                                one column per utility coefficient being 
                                estimated. All elements should be ints, 
                                floats, or longs.
                                
    alt_IDs:                    1D numpy array. All elements should be ints. 
                                There should be one row per obervation per 
                                available alternative for the given 
                                observation. Elements denote the alternative 
                                corresponding to the given row of the design 
                                matrix.
                                
    alt_to_obs:                 2D numpy array with one row per observation 
                                per available alternative and one column 
                                per observation. This matrix maps the rows of 
                                the design matrix to the unique observations 
                                (on the columns).
                                
    alt_to_shapes:              2D numpy array with one row per observation 
                                per available alternative and one column per
                                possible alternative. This matrix maps the 
                                rows of the design matrix to the possible 
                                alternatives for this dataset.
                                
    choice_vector:              1D numpy array. All elements should be either 
                                ones or zeros. There should be one row per 
                                observation per available alternative for the 
                                given observation. Elements denote the 
                                alternative which is chosen by the given 
                                observation with a 1 and a zero otherwise.
                                
    utility_transform:          callable. Should accept a 1D array of 
                                systematic utility values, a 1D array of 
                                alternative IDs, and miscellaneous args and 
                                kwargs. Should return a 1D array whose elements 
                                contain the appropriately transformed 
                                systematic utility values, based on the current 
                                model being evaluated.
                                
    splitting_func:             callable. Should accept in the following order:
                                a 1D array of parameter values being estimated,
                                a 2D compressed sparse row matrix that maps the
                                rows of the design matrix to the alternatives 
                                in this dataset, and the design matrix. It 
                                should return a tuple with three elements: the 
                                array  of shape parameters or None, the array 
                                of 'outside' intercepts or None, and the array 
                                of index coefficients.
                                
    transform_first_deriv_c:    callable. Must accept a 1D array of systematic 
                                utility values, a 1D array of alternative IDs,
                                the alt_to_shapes array, (shape parameters if
                                there are any) and miscellaneous args and 
                                kwargs. Should return a 2D array whose elements 
                                contain the derivative of the tranformed 
                                utility vector with respect to the vector of 
                                shape parameters. The dimensions of the 
                                returned vector should be (design.shape[0], 
                                num_alternatives). If there are no shape 
                                parameters, the callable should return None.
                                
    transform_first_deriv_v:    callable. Must accept a 1D array of systematic 
                                utility values, a 1D array of alternative IDs, 
                                (shape parameters if there are any) and 
                                miscellaneous args and kwargs. Should return a 
                                2D array whose elements contain the 
                                derivative of the tranformed utility vector 
                                with respect to the vector of systematic 
                                utilities. The dimensions of the returned 
                                vector should be (design.shape[0], 
                                design.shape[0]).
                                
    transform_deriv_alpha:      callable. Must accept a 1D array of systematic
                                utility values, a 1D array of alternative IDs,
                                the alt_to_shapes array, (intercept parameters
                                if there are any) and miscellaneous args and
                                kwargs. Should return a 2D array whose elements
                                contain the derivative of the tranformed 
                                utility vector with respect to the vector of 
                                shape parameters. The dimensions of the 
                                returned vector should be (design.shape[0],
                                num_alternatives - 1). If there are no 
                                intercept parameters, the callable should '
                                return None.
                                
    init_log_likelihood:        int, float, or long. The log-likelihood of the
                                choice model at the initial value of theta.
    
    maxiter:                    OPTIONAL. Int. Determines the maximum number of 
                                iterations of the EM-algorithm. Default = 1200.
                                
    m_step_maxiter:             OPTIONAL. Int. Determines the maximum number of
                                iterations used by the optimizer within the 
                                M-step of the EM-algorithm.
                                
    ll_tol:                     OPTIONAL. float. Determines the convergence 
                                tolerance for the log-likelihood values. If the
                                difference between two successive 
                                log-likelihoods is smaller than this value, the 
                                estimator will be deemed to have converged. 
                                Default = 1e-6.
                                
    gradient_tol:               OPTIONAL. float. Determines the convergence 
                                tolerance for the gradient. The estimator will
                                be deemed to have converged when the partial
                                derivative of the log-likelihood with respect
                                to each value of theta is less than
                                gradient_tol. Default = 1e-6.
                                
    m_method:                   OPTIONAL. String. Should be a valid string 
                                which can be passed to scipy.optimize.minimize.
                                Determines the optimization algorithm which is 
                                used for the M-step of the algorithm problem.
                                Default='cg'.
    ==============================
    Returns:                    dict. Should have the same keys as the 
                                scipy.optimize.minimize results dictionary.
                                
    NOTE:                       This function doesn't yet do ridge regression.
    """
    ###########
    # Create needed model objects
    ###########
    # Determine what rows correspond to chosen and rejected alternatives
    chosen_row_indices = np.where(choice_vector == 1)[0]
    rejected_row_indices = np.where(choice_vector == 0)[0]
    
    # Select the rows of the mapping matrices that correspond to the
    # rejected alternatives
    rejected_row_to_obs = alt_to_obs[rejected_row_indices, :]
    rejected_row_to_alts = alt_to_shapes[rejected_row_indices, :]
    # Get the indices of the rows and columns that correspond to
    # alternatives that are unavailable to particular individuals
    # in wide format. All alternatives for the dataset are on the
    # columns and the observations correspond to the rows.
    wide_unavailable_idxs = np.where(alt_to_obs.transpose()
                                               .dot(alt_to_shapes)
                                               .toarray() == 0)
    
    # Create a sparse diagonal matrix of the correct shape for use
    # in computing the complete data log-likelihood
    base_diag = diags(np.ones(alt_to_obs.shape[1]), 0, format='csr')
    
    ###########
    # Get the initial values and containers needed for the
    # iterative estimation process of the EM algorithm
    ###########
    # Calculate and store the initial log-likelihood in a list
    # Note the "init_log_likelihood - 10" is used to meet the
    # while loops condition. It is not reported or used thereafter
    log_likelihoods = [init_log_likelihood - 10, init_log_likelihood]
    
    # Initialize the iteration counter
    iteration = 0
    
    # Initialize the gradient of the log-likelihood
    gradient = np.ones(theta.shape[0])
    
    # Separate the shape, intercept, and beta parameters
    shape_vec, intercept_vec, coefficient_vec = splitting_func(theta, 
                                                               alt_to_shapes,
                                                               design)
    
    # Start the timer
    start_time = time.time()
    
    # Use a while loop with three conditions: a log-likelihood condition, 
    # a gradient condition, and a max iteration condition. Note the 
    # conditions should be with respect to the actual log-likelihood, 
    # not the complete data log-likeliood.
    
    # Note I removed this condition.
    #(log_likelihoods[-1] - log_likelihoods[-2] > ll_tol) and
    while ((0 <= iteration < maxiter) and
            log_likelihoods[-1] !=  log_likelihoods[-2] and
           (np.abs(gradient) > gradient_tol).any()):
        # Periodically, give users an update
        if iteration % 100 == 0 and iteration != 0:
            msg = "Currently on iteration {:,.0f}"
            msg_1 = "Current log-likelihood is: {:,.2f}"
            print msg.format(iteration)
            print msg_1.format(log_likelihoods[-1])
            print"\n"
            sys.stdout.flush()
        
        ###########
        # Perform the E-step:
        ###########
        # Calculate the systematic utility for each alternative 
        # for each individual
        sys_utilities = design.dot(coefficient_vec)

        # Calculate the probability from the transformed utilities
        # The transformed utilities will be of shape (num_rows, 1)
        transformed_utilities = utility_transform(sys_utilities,
                                                  alt_IDs,
                                                  alt_to_shapes,
                                                  shape_vec,
                                                  intercept_vec).ravel()

        # The following commands are to guard against numeric under/over-flow
        transformed_utilities[transformed_utilities < min_exponent_val] =\
                                                               min_exponent_val
        transformed_utilities[transformed_utilities > max_exponent_val] =\
                                                               max_exponent_val

        # Calculate the initial psi vector
        psi_vec = calc_psi(transformed_utilities,
                           chosen_row_indices,
                           rejected_row_indices,
                           rejected_row_to_obs,
                           rejected_row_to_alts,
                           wide_unavailable_idxs)
        
        # Calculate the weight vector
        weights = calc_expected_weight(psi_vec)
        
        ###########
        # Perform the M-step:
        ###########
        # Get the maximization results
        max_res = minimize(calc_neg_complete_data_log_likelihood_and_gradient,
                           theta,
                           args = (weights,
                                   design,
                                   alt_IDs,
                                   alt_to_shapes,
                                   splitting_func,
                                   utility_transform,
                                   chosen_row_indices,
                                   rejected_row_indices,
                                   rejected_row_to_obs,
                                   rejected_row_to_alts,
                                   wide_unavailable_idxs,
                                   transform_first_deriv_c,
                                   transform_first_deriv_v,
                                   transform_deriv_alpha,
                                   base_diag), 
                       method = m_method,
                       jac = True, 
                       hess = calc_neg_hessian_complete_data_log_likelihood,
                       tol = ll_tol,
                       options = {'gtol': gradient_tol,
                                  "maxiter": m_step_maxiter})
        
        # Get the updates to the parameter estimates
        theta = max_res.x
        
        # Separate the shape, intercept, and beta parameters
        shape_vec, intercept_vec, coefficient_vec = splitting_func(theta, 
                                                                 alt_to_shapes,
                                                                   design)
        
        # Calculate the true objective functions log-likelihood 
        # at the new theta
        new_ll = calc_log_likelihood(coefficient_vec,
                                     design, 
                                     alt_IDs,
                                     alt_to_obs,
                                     alt_to_shapes,
                                     choice_vector,
                                     utility_transform,
                                     intercept_params=intercept_vec,
                                     shape_params=shape_vec)
        
        # Store the new log-likelihood value
        log_likelihoods.append(new_ll)
        
        # Calculate the gradient of the true objective function
        # Note the None is because ridge regression is not yet supported
        gradient = calc_gradient(coefficient_vec, 
                                 design,
                                 alt_IDs, 
                                 alt_to_obs,
                                 alt_to_shapes,
                                 choice_vector,
                                 utility_transform,
                                 transform_first_deriv_c,
                                 transform_first_deriv_v,
                                 transform_deriv_alpha,
                                 intercept_vec,
                                 shape_vec,
                                 None)
        
        # Keep track of what iteration we're on
        iteration += 1
        
    # Once the while loop is exited, store the final estimated parameters,
    # the final gradient, and the log-likelihood-progress, and create the
    # results dictionary
    max_res["fun"] = -1 * new_ll
    max_res["jac"] = -1 * gradient
    # Note that the first log-likelihood in the list wasn't real and was
    # simply for initialization purposes so the while loop would work
    max_res["log_likelihood_progress"] = log_likelihoods[1:]
    
    # Store the 'reason' for convergence
    if iteration >= maxiter:
        msg = "Maximum number of iterations reached in EM Algorithm"
    elif (np.abs(gradient) < gradient_tol).all():
        msg = "All entries of the gradient were less than the tolerance."
    elif log_likelihoods[-1] - log_likelihoods[-2] < ll_tol:
        msg = "Change in log-likelihood was below the tolerance level."
        print "Terminated estimation on iteration {:,.0f}".format(iteration)
    else:
        msg = "The EM Algorithm stopped for an unknown reason."
        
    max_res["estimation_reason"] = msg
    
    # Print out a message regarding the EM algorithm estimation time.
    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    elapsed_min = elapsed_sec /60.0
    if elapsed_min > 1.0:
        msg = "EM-Algorithm Estimation Time: {:.2f} minutes."
        print msg.format(elapsed_min)
    else:
        msg = "EM-Algorithm Estimation Time: {:.2f} seconds."
        print msg.format(elapsed_sec)
    print ""
    
    # Return the results dictionary
    return max_res