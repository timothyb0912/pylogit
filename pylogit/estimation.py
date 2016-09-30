"""
This module provides a general "estimate" function and EstimationObj class for
pylogit's logit-type models.
"""
import sys
import time
import numpy as np
from scipy.optimize import minimize

import choice_calcs as cc
from choice_calcs import create_matrix_block_indices
from choice_tools import ensure_ridge_is_scalar_or_none


class EstimationObj(object):
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
                 split_params):
        # Store pointers to needed objects
        self.alt_id_vector = model_obj.alt_IDs
        self.choice_vector = model_obj.choices
        self.design = model_obj.design
        self.intercept_ref_pos = model_obj.intercept_ref_position
        self.shape_ref_pos = model_obj.shape_ref_position

        # Explicitly store pointers to the mapping matrices
        self.rows_to_obs = mapping_dict["rows_to_obs"]
        self.rows_to_alts = mapping_dict["rows_to_alts"]
        self.chosen_row_to_obs = mapping_dict["chosen_row_to_obs"]

        # Perform necessary checking of ridge parameter here!
        ensure_ridge_is_scalar_or_none(ridge)

        # Store the ridge parameter
        self.ridge = ridge

        # Store the constrained parameters
        # Commented out because this feature is not yet supported in logit-type
        # models.
        # self.constrained_pos = constrained_pos

        # Store reference to what 'zero vector' is for this model / dataset
        self.zero_vector = zero_vector

        # Store the function that separates the various portions of the
        # parameters being estimated (shape parameters, outside intercepts,
        # utility coefficients)
        self.split_params = split_params

        # Store the function that calculates the transformation of the index
        self.utility_transform = model_obj.utility_transform

        # Get the block matrix indices for the hessian matrix.
        self.block_matrix_idxs = create_matrix_block_indices(self.rows_to_obs)

        # Note the following attributes should be set to actual callables that
        # calculate the necessary derivatives in the classes that inherit from
        # EstimationObj
        self.calc_dh_dv = lambda *args: None
        self.calc_dh_d_alpha = lambda *args: None
        self.calc_dh_d_shape = lambda *args: None

        return None

    def convenience_split_params(self, params):
        """
        Splits parameter vector into shape, intercept, and index parameters.
        """
        return self.split_params(params, self.rows_to_alts, self.design)

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
                  "ridge": self.ridge}
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
                self.ridge]

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
                self.block_matrix_idxs,
                intercepts,
                shapes,
                self.ridge]

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
                self.ridge]

        return cc.calc_fisher_info_matrix(*args)

    def calc_neg_log_likelihood_and_neg_gradient(self, params):
        """
        Calculates and returns the negative of the log-likelihood and the
        negative of the gradient. This function is used as the objective
        function in scipy.optimize.minimize.
        """
        neg_log_likelihood = -1 * self.convenience_calc_log_likelihood(params)
        neg_gradient = -1 * self.convenience_calc_gradient(params)

        return neg_log_likelihood, neg_gradient

    def calc_neg_hessian(self, params):
        """
        Calculate and return the negative of the hessian for this model and
        dataset.
        """
        return -1 * self.convenience_calc_hessian(params)


def calc_individual_chi_squares(residuals,
                                long_probabilities,
                                rows_to_obs):
    """
    Calculates individual chi-squared values for each choice situation in the
    dataset.

    Parameters
    ----------
    residuals
    long_probabilities
    rows_to_obs

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
    estimator

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
    split_res = estimator.convenience_split_params(final_params)
    results_dict["nest_params"] = None
    results_dict["shape_params"] = split_res[0]
    results_dict["intercept_params"] = split_res[1]
    results_dict["utility_coefs"] = split_res[2]

    # Get the probability of the chosen alternative and long_form probabilities
    chosen_probs, long_probs = estimator.convenience_calc_probs(final_params)
    results_dict["chosen_probs"] = chosen_probs
    results_dict["long_probs"] = long_probs

    #####
    # Calculate the residuals and individual chi-square values
    #####
    # Calculate the residual vector
    residuals = estimator.choice_vector - long_probs
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

    return results_dict


def estimate(init_values,
             estimator,
             method,
             loss_tol,
             gradient_tol,
             maxiter,
             print_results,
             **kwargs):
    # Perform preliminary calculations
    log_likelihood_at_zero =\
        estimator.convenience_calc_log_likelihood(estimator.zero_vector)

    initial_log_likelihood =\
        estimator.convenience_calc_log_likelihood(init_values)

    if print_results:
        # Print the log-likelihood at zero
        print("Log-likelihood at zero: {:,.4f}".format(log_likelihood_at_zero))

        # Print the log-likelihood at the starting values
        print("Initial Log-likelihood: {:,.4f}".format(initial_log_likelihood))
        sys.stdout.flush()

    # Estimate the actual parameters of the model
    start_time = time.time()

    results = minimize(estimator.calc_neg_log_likelihood_and_neg_gradient,
                       init_values,
                       method=method,
                       jac=True,
                       hess=estimator.calc_neg_hessian,
                       tol=loss_tol,
                       options={'gtol': gradient_tol,
                                "maxiter": maxiter},
                       **kwargs)

    # Stop timing the estimation process and report the timing results
    end_time = time.time()
    if print_results:
        elapsed_sec = (end_time - start_time)
        elapsed_min = elapsed_sec / 60.0
        if elapsed_min > 1.0:
            print("Estimation Time: {:.2f} minutes.".format(elapsed_min))
        else:
            print("Estimation Time: {:.2f} seconds.".format(elapsed_sec))
        print("Final log-likelihood: {:,.4f}".format(-1 * results["fun"]))
        sys.stdout.flush()

    # Store the log-likelihood at zero
    results["log_likelihood_null"] = log_likelihood_at_zero

    # Calculate and store the post-estimation results
    results = calc_and_store_post_estimation_results(results, estimator)

    return results
