"""
This module provides a general "estimate" function for the pylogit's logit-type
discrete choice models.

I'm thinking it may be best to have an estimation_obj class that is composed
of attributes from the general model class but creates the various convenience
functions needed for model estimation...
"""
import numpy as np

import choice_calcs as cc
from choice_calcs import create_matrix_block_indices


class EstimationObj(object):
    """
    Generic class for storing pointers to data and methods needed in the
    estimation process.

    Parameters
    ----------
    alt_id_vector
    choice_vector
    mapping_res
    intercept_ref_pos
    shape_ref_pos
    zero_vector
    ridge
    constrained_pos

    Attributes
    ----------

    Methods
    -------
    """
    def __init__(self,
                 model_obj,
                 mapping_dict,
                 ridge,
                 constrained_pos,
                 zero_vector,
                 split_params):
        # Store pointers to needed objects
        self.alt_id_vector = model_obj.alt_IDs
        self.choice_vector = model_obj.choices
        self.design = model_obj.design
        self.intercept_ref_pos = model_obj.intercept_ref_position
        self.shape_ref_pos = model_obj.shape_ref_position

        # Store pointers to the mapping matrices
        for key in mapping_dict:
            setattr(self, key, mapping_dict[key])

        # Store the ridge parameter
        self.ridge = ridge

        # Store the constrained parameters
        self.constrained_pos = constrained_pos

        # Store reference to what 'zero vector' for this model / dataset
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

    def convenience_calc_probs(self, params):
        """
        Calculates the probabilities of the chosen alternative, and the long
        format probabilities for this model and dataset.
        """
        shapes, intercepts, betas = self.split_params(params,
                                                      self.rows_to_alts,
                                                      self.design)

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
        shapes, intercepts, betas = self.split_params(params,
                                                      self.rows_to_alts,
                                                      self.design)

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
        shapes, intercepts, betas = self.split_params(params,
                                                      self.rows_to_alts,
                                                      self.design)

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
        shapes, intercepts, betas = self.split_params(params,
                                                      self.rows_to_alts,
                                                      self.design)

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
                self.block_matrix_idxs,
                intercepts,
                shapes,
                self.ridge]

        return cc.calc_hessian(*args)



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
                                           convenient_split_param_vec,
                                           convenient_calc_probs,
                                           convenient_calc_gradient,
                                           convenient_calc_hessian,
                                           convenient_calc_fisher,
                                           rows_to_obs,
                                           choice_vector):
    """
    Calculates and stores post-estimation results that require the use of the
    systematic utility transformation functions or the various derivative
    functions.

    Parameters
    ----------
    results_dict : dict.
        This dictionary should be the dictionary returned from
        scipy.optimize.minimize. In particular, it should have the following
        keys: `["fun", "x", "log_likelihood_null"]`.
    convenient_split_param_vec : callable.
        Should accept the final array of estimated parameters. Should return
        a tuple of `(nest_params, shape_params, intercept_params,
        utility_coefs)`.
    convenient_calc_probs : callable.
        Should accept the final array of estimated parameters. Should return
        a tuple containing the probability of the chosen alternative and the
        probability of each row of the current long-format dataframe.
    convenient_calc_gradient : callable.
        Should accept the final array of estimated parameters. Should return
        the gradient of the log-likelihood, given the current dataset.
    convenient_calc_hessian : callable.
        Should accept the final array of estimated parameters. Should return
        the hessian of the log-likelihood, given the current dataset.
    convenient_calc_fisher : callable.
        Should accept the final array of estimated parameters. Should return
        the BHHH approximation to the Fisher Information Matrix.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.

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
    split_res = convenient_split_param_vec(final_params)
    results_dict["nest_params"] = split_res[0]
    results_dict["shape_params"] = split_res[1]
    results_dict["intercept_params"] = split_res[2]
    results_dict["utility_coefs"] = split_res[3]

    # Get the probability of the chosen alternative and long_form probabilities
    prob_of_chosen_alts, long_probs = convenient_calc_probs(final_params)
    results_dict["chosen_probs"] = prob_of_chosen_alts
    results_dict["long_probs"] = long_probs

    #####
    # Calculate the residuals and individual chi-square values
    #####
    # Calculate the residual vector
    residuals = choice_vector - long_probs
    results_dict["residuals"] = residuals

    # Calculate the observation specific chi-squared components
    results_dict["ind_chi_squareds"] = calc_individual_chi_squares(residuals,
                                                                   long_probs,
                                                                   rows_to_obs)

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
    results_dict["final_gradient"] = convenient_calc_gradient(final_params)
    results_dict["final_hessian"] = convenient_calc_hessian(final_params)
    results_dict["fisher_info"] = convenient_calc_fisher(final_params)

    return results_dict

def estimate():
    # Check validity of the provided arguments

    # Estimate the actual parameters of the model

    # Calculate and store the post-estimation results
    calc_and_store_post_estimation_results(results,
                                           convenient_split_param_vec,
                                           convenient_calc_probs,
                                           convenient_calc_gradient,
                                           convenient_calc_hessian,
                                           convenient_calc_fisher,
                                           rows_to_obs,
                                           choice_vector)

    return results