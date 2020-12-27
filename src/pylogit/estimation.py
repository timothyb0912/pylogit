# -*- coding: utf-8 -*-
"""
This module provides a general "estimate" function and EstimationObj class for
pylogit's logit-type models.
"""
from __future__ import absolute_import

import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import choice_calcs as cc
from .choice_tools import ensure_ridge_is_scalar_or_none
from .choice_tools import ensure_contiguity_in_observation_rows


def ensure_positivity_and_length_of_weights(weights, data):
    assert isinstance(data, pd.DataFrame)
    if weights is None:
        return None
    elif not isinstance(weights, np.ndarray) or weights.ndim != 1:
        msg = '`weights` MUST be a 1D ndarray.'
        raise ValueError(msg)
    elif weights.size != data.shape[0]:
        msg = '`weights` must have the same number of rows as `data`.'
        raise ValueError(msg)
    elif (weights < 0).any():
        msg = '`weights` MUST be >= 0.'
        raise ValueError(msg)
    return None


class EstimationObj(object):
    """
    Generic class for storing pointers to data and methods needed in the
    estimation process. Defines the basic Estimation Object and the various
    methods and attributes that such an object should have. Will be subclassed
    in order to serve the needs of the various types of estimated models.

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
    constrained_pos : list or None, optional.
        Denotes the positions of the array of estimated parameters that are
        not to change from their initial values. If a list is passed, the
        elements are to be integers where no such integer is greater than
        `init_values.size.` Default == None.
    weights : 1D ndarray or None, optional.
        Allows for the calculation of weighted log-likelihoods. The weights can
        represent various things. In stratified samples, the weights may be
        the proportion of the observations in a given strata for a sample in
        relation to the proportion of observations in that strata in the
        population. In latent class models, the weights may be the probability
        of being a particular class.
    """
    def __init__(self,
                 model_obj,
                 mapping_dict,
                 ridge,
                 zero_vector,
                 split_params,
                 constrained_pos=None,
                 weights=None):
        # Store pointers to needed objects
        self.alt_id_vector = model_obj.alt_IDs
        self.choice_vector = model_obj.choices
        self.obs_id_vector = model_obj.data[model_obj.obs_id_col].values
        self.design = model_obj.design
        self.intercept_ref_pos = model_obj.intercept_ref_position
        self.shape_ref_pos = model_obj.shape_ref_position

        # Explicitly store pointers to the mapping matrices
        self.rows_to_obs = mapping_dict["rows_to_obs"]
        self.rows_to_alts = mapping_dict["rows_to_alts"]
        self.chosen_row_to_obs = mapping_dict["chosen_row_to_obs"]
        self.rows_to_nests = mapping_dict["rows_to_nests"]
        self.rows_to_mixers = mapping_dict["rows_to_mixers"]

        # Perform necessary checking of ridge parameter here!
        ensure_ridge_is_scalar_or_none(ridge)
        # Ensure the dataset has contiguity in rows with the same obs_id
        ensure_contiguity_in_observation_rows(self.obs_id_vector)
        # Ensure the weights are appropriate for model estimation
        ensure_positivity_and_length_of_weights(weights, model_obj.data)

        # Store the ridge parameter
        self.ridge = ridge

        # Store the constrained parameters
        self.constrained_pos = constrained_pos

        # Store reference to what 'zero vector' is for this model / dataset
        self.zero_vector = zero_vector

        # Store the weights that were passed to the constructor
        self.weights =\
            np.ones(self.design.shape[0]) if weights is None else weights

        # Store the function that separates the various portions of the
        # parameters being estimated (shape parameters, outside intercepts,
        # utility coefficients)
        self.split_params = split_params

        # Store the function that calculates the transformation of the index
        self.utility_transform = model_obj.utility_transform

        # Note the following attributes should be set to actual callables that
        # calculate the necessary derivatives in the classes that inherit from
        # EstimationObj
        self.calc_dh_dv = lambda *args: None
        self.calc_dh_d_alpha = lambda *args: None
        self.calc_dh_d_shape = lambda *args: None

        return None

    def convenience_split_params(self, params, return_all_types=False):
        """
        Splits parameter vector into shape, intercept, and index parameters.

        Parameters
        ----------
        params : 1D ndarray.
            The array of parameters being estimated or used in calculations.
        return_all_types : bool, optional.
            Determines whether or not a tuple of 4 elements will be returned
            (with one element for the nest, shape, intercept, and index
            parameters for this model). If False, a tuple of 3 elements will
            be returned with one element for the shape, intercept, and index
            parameters.

        Returns
        -------
        tuple. Will have 4 or 3 elements based on `return_all_types`.
        """
        return self.split_params(params,
                                 self.rows_to_alts,
                                 self.design,
                                 return_all_types=return_all_types)

    def convenience_calc_probs(self, params):
        """
        Calculates the probabilities of the chosen alternative, and the long
        format probabilities for this model and dataset.
        """
        msg = "Method should be defined by descendant classes"
        raise NotImplementedError(msg)

    def convenience_calc_log_likelihood(self, params):
        """
        Calculates the log-likelihood for this model and dataset.
        """
        msg = "Method should be defined by descendant classes"
        raise NotImplementedError(msg)

    def convenience_calc_gradient(self, params):
        """
        Calculates the gradient of the log-likelihood for this model / dataset.
        """
        msg = "Method should be defined by descendant classes"
        raise NotImplementedError(msg)

    def convenience_calc_hessian(self, params):
        """
        Calculates the hessian of the log-likelihood for this model / dataset.
        """
        msg = "Method should be defined by descendant classes"
        raise NotImplementedError(msg)

    def convenience_calc_fisher_approx(self, params):
        """
        Calculates the BHHH approximation of the Fisher Information Matrix for
        this model / dataset.
        """
        msg = "Method should be defined by descendant classes"
        raise NotImplementedError(msg)

    def calc_neg_log_likelihood_and_neg_gradient(self, params):
        """
        Calculates and returns the negative of the log-likelihood and the
        negative of the gradient. This function is used as the objective
        function in scipy.optimize.minimize.
        """
        neg_log_likelihood = -1 * self.convenience_calc_log_likelihood(params)
        neg_gradient = -1 * self.convenience_calc_gradient(params)

        if self.constrained_pos is not None:
            neg_gradient[self.constrained_pos] = 0

        return neg_log_likelihood, neg_gradient

    def calc_neg_hessian(self, params):
        """
        Calculate and return the negative of the hessian for this model and
        dataset.
        """
        return -1 * self.convenience_calc_hessian(params)


class LogitTypeEstimator(EstimationObj):
    """
    Generic class for storing pointers to data and methods needed in the
    estimation process.

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

    Attributes
    ----------

    Methods
    -------
    """
    def __init__(self,
                 model_obj,
                 mapping_dict,
                 ridge,
                 zero_vector,
                 split_params,
                 constrained_pos=None,
                 weights=None):

        kwargs = {"constrained_pos": constrained_pos,
                  "weights": weights}
        super(LogitTypeEstimator, self).__init__(model_obj,
                                                 mapping_dict,
                                                 ridge,
                                                 zero_vector,
                                                 split_params,
                                                 **kwargs)

        return None

    def convenience_calc_probs(self, params):
        """
        Calculates the probabilities of the chosen alternative, and the long
        format probabilities for this model and dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        prob_args = [betas,
                     self.design,
                     self.alt_id_vector,
                     self.rows_to_obs,
                     self.rows_to_alts,
                     self.utility_transform]

        prob_kwargs = {"intercept_params": intercepts,
                       "shape_params": shapes,
                       "chosen_row_to_obs": self.chosen_row_to_obs,
                       "return_long_probs": True}
        prob_results = cc.calc_probabilities(*prob_args, **prob_kwargs)

        return prob_results

    def convenience_calc_log_likelihood(self, params):
        """
        Calculates the log-likelihood for this model and dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        args = [betas,
                self.design,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.choice_vector,
                self.utility_transform]

        kwargs = {"intercept_params": intercepts,
                  "shape_params": shapes,
                  "ridge": self.ridge,
                  "weights": self.weights}
        log_likelihood = cc.calc_log_likelihood(*args, **kwargs)

        return log_likelihood

    def convenience_calc_gradient(self, params):
        """
        Calculates the gradient of the log-likelihood for this model / dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        args = [betas,
                self.design,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.choice_vector,
                self.utility_transform,
                self.calc_dh_d_shape,
                self.calc_dh_dv,
                self.calc_dh_d_alpha,
                intercepts,
                shapes,
                self.ridge,
                self.weights]

        return cc.calc_gradient(*args)

    def convenience_calc_hessian(self, params):
        """
        Calculates the hessian of the log-likelihood for this model / dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        args = [betas,
                self.design,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.utility_transform,
                self.calc_dh_d_shape,
                self.calc_dh_dv,
                self.calc_dh_d_alpha,
                intercepts,
                shapes,
                self.ridge,
                self.weights]

        return cc.calc_hessian(*args)

    def convenience_calc_fisher_approx(self, params):
        """
        Calculates the BHHH approximation of the Fisher Information Matrix for
        this model / dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        args = [betas,
                self.design,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.choice_vector,
                self.utility_transform,
                self.calc_dh_d_shape,
                self.calc_dh_dv,
                self.calc_dh_d_alpha,
                intercepts,
                shapes,
                self.ridge,
                self.weights]

        return cc.calc_fisher_info_matrix(*args)


def calc_individual_chi_squares(residuals,
                                long_probabilities,
                                rows_to_obs):
    """
    Calculates individual chi-squared values for each choice situation in the
    dataset.

    Parameters
    ----------
    residuals : 1D ndarray.
        The choice vector minus the predicted probability of each alternative
        for each observation.
    long_probabilities : 1D ndarray.
        The probability of each alternative being chosen in each choice
        situation.
    rows_to_obs : 2D scipy sparse array.
        Should map each row of the long format dataferame to the unique
        observations in the dataset.

    Returns
    -------
    ind_chi_squareds : 1D ndarray.
        Will have as many elements as there are columns in `rows_to_obs`. Each
        element will contain the pearson chi-squared value for the given choice
        situation.
    """
    chi_squared_terms = np.square(residuals) / long_probabilities
    return rows_to_obs.T.dot(chi_squared_terms)


def calc_rho_and_rho_bar_squared(final_log_likelihood,
                                 null_log_likelihood,
                                 num_est_parameters):
    """
    Calculates McFadden's rho-squared and rho-bar squared for the given model.

    Parameters
    ----------
    final_log_likelihood : float.
        The final log-likelihood of the model whose rho-squared and rho-bar
        squared are being calculated for.
    null_log_likelihood : float.
        The log-likelihood of the model in question, when all parameters are
        zero or their 'base' values.
    num_est_parameters : int.
        The number of parameters estimated in this model.

    Returns
    -------
    `(rho_squared, rho_bar_squared)` : tuple of floats.
        The rho-squared and rho-bar-squared for the model.
    """
    rho_squared = 1.0 - final_log_likelihood / null_log_likelihood
    rho_bar_squared = 1.0 - ((final_log_likelihood - num_est_parameters) /
                             null_log_likelihood)

    return rho_squared, rho_bar_squared


def calc_and_store_post_estimation_results(results_dict,
                                           estimator):
    """
    Calculates and stores post-estimation results that require the use of the
    systematic utility transformation functions or the various derivative
    functions. Note that this function is only valid for logit-type models.

    Parameters
    ----------
    results_dict : dict.
        This dictionary should be the dictionary returned from
        scipy.optimize.minimize. In particular, it should have the following
        keys: `["fun", "x", "log_likelihood_null"]`.
    estimator : an instance of the EstimationObj class.
        Should contain the following attributes or methods:

          - convenience_split_params
          - convenience_calc_probs
          - convenience_calc_gradient
          - convenience_calc_hessian
          - convenience_calc_fisher_approx
          - choice_vector
          - rows_to_obs

    Returns
    -------
    results_dict : dict.
        The following keys will have been entered into `results_dict`:

          - final_log_likelihood
          - utility_coefs
          - intercept_params
          - shape_params
          - nest_params
          - chosen_probs
          - long_probs
          - residuals
          - ind_chi_squareds
          - rho_squared
          - rho_bar_squared
          - final_gradient
          - final_hessian
          - fisher_info
    """
    # Store the final log-likelihood
    final_log_likelihood = -1 * results_dict["fun"]
    results_dict["final_log_likelihood"] = final_log_likelihood

    # Get the final array of estimated parameters
    final_params = results_dict["x"]

    # Add the estimated parameters to the results dictionary
    split_res = estimator.convenience_split_params(final_params,
                                                   return_all_types=True)
    results_dict["nest_params"] = split_res[0]
    results_dict["shape_params"] = split_res[1]
    results_dict["intercept_params"] = split_res[2]
    results_dict["utility_coefs"] = split_res[3]

    # Get the probability of the chosen alternative and long_form probabilities
    chosen_probs, long_probs = estimator.convenience_calc_probs(final_params)
    results_dict["chosen_probs"] = chosen_probs
    results_dict["long_probs"] = long_probs

    #####
    # Calculate the residuals and individual chi-square values
    #####
    # Calculate the residual vector
    if len(long_probs.shape) == 1:
        residuals = estimator.choice_vector - long_probs
    else:
        residuals = estimator.choice_vector[:, None] - long_probs
    results_dict["residuals"] = residuals

    # Calculate the observation specific chi-squared components
    args = [residuals, long_probs, estimator.rows_to_obs]
    results_dict["ind_chi_squareds"] = calc_individual_chi_squares(*args)

    # Calculate and store the rho-squared and rho-bar-squared
    log_likelihood_null = results_dict["log_likelihood_null"]
    rho_results = calc_rho_and_rho_bar_squared(final_log_likelihood,
                                               log_likelihood_null,
                                               final_params.shape[0])
    results_dict["rho_squared"] = rho_results[0]
    results_dict["rho_bar_squared"] = rho_results[1]

    #####
    # Calculate the gradient, hessian, and BHHH approximation to the fisher
    # info matrix
    #####
    results_dict["final_gradient"] =\
        estimator.convenience_calc_gradient(final_params)
    results_dict["final_hessian"] =\
        estimator.convenience_calc_hessian(final_params)
    results_dict["fisher_info"] =\
        estimator.convenience_calc_fisher_approx(final_params)

    # Store the constrained positions that was used in this estimation process
    results_dict["constrained_pos"] = estimator.constrained_pos

    return results_dict


def estimate(init_values,
             estimator,
             method,
             loss_tol,
             gradient_tol,
             maxiter,
             print_results,
             use_hessian=True,
             just_point=False,
             **kwargs):
    """
    Estimate the given choice model that is defined by `estimator`.

    Parameters
    ----------
    init_vals : 1D ndarray.
        Should contain the initial values to start the optimization process
        with.
    estimator : an instance of the EstimationObj class.
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
    print_res : bool, optional.
        Determines whether the timing and initial and final log likelihood
        results will be printed as they they are determined.
        Default `== True`.
    use_hessian : bool, optional.
        Determines whether the `calc_neg_hessian` method of the `estimator`
        object will be used as the hessian function during the estimation. This
        kwarg is used since some models (such as the Mixed Logit and Nested
        Logit) use a rather crude (i.e. the BHHH) approximation to the Fisher
        Information Matrix, and users may prefer to not use this approximation
        for the hessian during estimation.
    just_point : bool, optional.
        Determines whether or not calculations that are non-critical for
        obtaining the maximum likelihood point estimate will be performed.
        Default == False.

    Return
    ------
    results : dict.
        The dictionary of estimation results that is returned by
        scipy.optimize.minimize. It will also have (at minimum) the following
        keys:
          - "log-likelihood_null"
          - "final_log_likelihood"
          - "utility_coefs"
          - "intercept_params"
          - "shape_params"
          - "nest_params"
          - "chosen_probs"
          - "long_probs"
          - "residuals"
          - "ind_chi_squareds"
          - "rho_squared"
          - "rho_bar_squared"
          - "final_gradient"
          - "final_hessian"
          - "fisher_info"
    """
    if not just_point:
        # Perform preliminary calculations
        log_likelihood_at_zero =\
            estimator.convenience_calc_log_likelihood(estimator.zero_vector)

        initial_log_likelihood =\
            estimator.convenience_calc_log_likelihood(init_values)

        if print_results:
            # Print the log-likelihood at zero
            null_msg = "Log-likelihood at zero: {:,.4f}"
            print(null_msg.format(log_likelihood_at_zero))

            # Print the log-likelihood at the starting values
            init_msg = "Initial Log-likelihood: {:,.4f}"
            print(init_msg.format(initial_log_likelihood))
            sys.stdout.flush()

    # Get the hessian fucntion for this estimation process
    hess_func = estimator.calc_neg_hessian if use_hessian else None

    # Estimate the actual parameters of the model
    start_time = time.time()

    results = minimize(estimator.calc_neg_log_likelihood_and_neg_gradient,
                       init_values,
                       method=method,
                       jac=True,
                       hess=hess_func,
                       tol=loss_tol,
                       options={'gtol': gradient_tol,
                                "maxiter": maxiter},
                       **kwargs)

    if not just_point:
        if print_results:
            # Stop timing the estimation process and report the timing results
            end_time = time.time()
            elapsed_sec = (end_time - start_time)
            elapsed_min = elapsed_sec / 60.0
            if elapsed_min > 1.0:
                msg = "Estimation Time for Point Estimation: {:.2f} minutes."
                print(msg.format(elapsed_min))
            else:
                msg = "Estimation Time for Point Estimation: {:.2f} seconds."
                print(msg.format(elapsed_sec))
            print("Final log-likelihood: {:,.4f}".format(-1 * results["fun"]))
            sys.stdout.flush()

        # Store the log-likelihood at zero
        results["log_likelihood_null"] = log_likelihood_at_zero

        # Calculate and store the post-estimation results
        results = calc_and_store_post_estimation_results(results, estimator)

    return results
