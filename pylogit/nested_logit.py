# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:26:33 2016

@module:    nested_logit.py
@name:      Nested Logit Model
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating nested logit models
            (with the help of the "base_multinomial_cm.py" file).
"""
import time
import sys
import warnings
import numpy as np
from scipy.optimize import minimize

import nested_choice_calcs as nc
import base_multinomial_cm_v2 as base_mcm
from choice_tools import ensure_ridge_is_scalar_or_none
from display_names import model_type_to_display_name

# Alias necessary functions from the base multinomial choice model module
general_log_likelihood = nc.calc_nested_log_likelihood
general_gradient = nc.calc_nested_gradient
general_calc_probabilities = nc.calc_nested_probs
bhhh_approx = nc.calc_bhhh_hessian_approximation

# Create a warning string that will be issued if ridge regression is performed.
_msg_3 = "NOTE: An L2-penalized regression is being performed. The "
_msg_4 = "reported standard errors and robust standard errors "
_msg_5 = "***WILL BE INCORRECT***."
_ridge_warning_msg = _msg_3 + _msg_4 + _msg_5


# Create a function that will identify degenerate nests
def identify_degenerate_nests(nest_spec):
    """
    Identify the nests within nest_spec that are degenerate, i.e. those nests
    with only a single alternative within the nest.

    Parameters
    ----------
    nest_spec : OrderedDict.
        Keys are strings that define the name of the nests. Values are lists
        of alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!

    Returns
    -------
    list.
        Will contain the positions in the list of keys from `nest_spec` that
        are degenerate.
    """
    degenerate_positions = []
    for pos, key in enumerate(nest_spec):
        if len(nest_spec[key]) == 1:
            degenerate_positions.append(pos)
    return degenerate_positions


# Define a functionto split the combined parameter array into nest coefficients
# and index coefficients.
def split_params(all_params, rows_to_nests):
    """
    Parameters
    ----------
    all_params : 1D ndarray.
        Should contain all of the parameters being estimated (i.e. all the
        nest coefficients and all of the index coefficients). All elements
        should be ints, floats, or longs.
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).

    Returns
    -------
    orig_nest_coefs : 1D ndarray.
        The nest coefficients being used for estimation. Note that these values
        are the logit of the inverse of the scale parameters for each lower
        level nest.
    index_coefs : 1D ndarray.
        The coefficients of the index being used for this nested logit model.
    """
    # Split the array of all coefficients
    num_nests = rows_to_nests.shape[1]
    orig_nest_coefs = all_params[:num_nests]
    index_coefs = all_params[num_nests:]
    return orig_nest_coefs, index_coefs


# Define function for calculating the log likelihood
def convenient_nested_log_likelihood(all_coefs,
                                     design,
                                     rows_to_obs,
                                     rows_to_nests,
                                     choice_vec,
                                     ridge=None):
    """
    Parameters
    ----------
    all_coefs : 1D ndarray.
        Should contain all of the parameters being estimated (i.e. all the
        nest coefficients and all of the index coefficients). All elements
        should be ints, floats, or longs.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated.
        All elements should be ints, floats, or longs.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).
    choice_vec : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a `1` and a zero otherwise.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If a scalar is
        passed, then that scalar determines the ridge penalty for the
        optimization. Default `== None`.

    Returns
    -------
    float. The log-likelihood of the nested logit model.
    """
    # Split the array of all coefficients
    orig_nest_coefs, index_coefs = split_params(all_coefs, rows_to_nests)
    # Get the
    nest_coefs = nc.naturalize_nest_coefs(orig_nest_coefs)

    return general_log_likelihood(nest_coefs,
                                  index_coefs,
                                  design,
                                  rows_to_obs,
                                  rows_to_nests,
                                  choice_vec,
                                  ridge=ridge)


# Create a function for calculating the gradient within a
# scipy.optimize.minimize estimation routine
def convenient_nested_gradient(all_coefs,
                               design,
                               choice_vec,
                               rows_to_obs,
                               rows_to_nests,
                               ridge=None):
    """
    Parameters
    ----------
    all_coefs : 1D ndarray.
        Should contain all of the parameters being estimated (i.e. all the
        nest coefficients and all of the index coefficients). All elements
        should be ints, floats, or longs.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated.
        All elements should be ints, floats, or longs.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If a scalar is
        passed, then that scalar determines the ridge penalty for the
        optimization. Default `== None`.

    Returns
    -------
    1D ndarray. The gradient of the nested logit's log-likelihood with respect
    to `all_coefs`.
    """
    # Split the array of all coefficients
    orig_nest_coefs, index_coefs = split_params(all_coefs, rows_to_nests)

    return general_gradient(orig_nest_coefs,
                            index_coefs,
                            design,
                            choice_vec,
                            rows_to_obs,
                            rows_to_nests,
                            ridge=ridge)


# Create an objective function for a
# scipy.optimize.minimize estimation routine
def calc_neg_nested_log_likelihood_and_gradient(all_coefs,
                                                design,
                                                choice_vec,
                                                rows_to_obs,
                                                rows_to_nests,
                                                constrained_pos,
                                                ridge):
    """
    Parameters
    ----------
    all_coefs : 1D ndarray.
        Should contain all of the parameters being estimated (i.e. all the
        nest coefficients and all of the index coefficients). All elements
        should be ints, floats, or longs.
    design : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated.
        All elements should be ints, floats, or longs.
    choice_vec : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a `1` and a zero otherwise.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).
    constrained_pos : list, or None.
        Denotes the positions of the array of estimated parameters that are not
        to change from their initial values. If a list is passed, the elements
        are to be integers where no such integer is is greater than
        `init_values.size`.
    ridge : int, float, long, or None.
        Determines whether or not ridge regression is performed. If a scalar is
        passed, then that scalar determines the ridge penalty for the
        optimization.

    Returns
    -------
    neg_log_likelihood : float.
        The negative of the log-likelihood for the nested logit model given
        the passed arguments.
    neg_gradient : 1D ndarray.
        The negative of the gradient of the log-likelilhood of the nested logit
        model given the passed arguments.
    """
    log_likelihood = convenient_nested_log_likelihood(all_coefs,
                                                      design,
                                                      rows_to_obs,
                                                      rows_to_nests,
                                                      choice_vec,
                                                      ridge=ridge)

    gradient = convenient_nested_gradient(all_coefs,
                                          design,
                                          choice_vec,
                                          rows_to_obs,
                                          rows_to_nests,
                                          ridge=ridge)
    if constrained_pos is not None:
        gradient[constrained_pos] = 0

    return -1 * log_likelihood, -1 * gradient


def _estimate(init_values,
              design_matrix,
              choice_vector,
              rows_to_obs,
              rows_to_nests,
              chosen_row_to_obs,
              constrained_pos,
              print_results=True,
              method='BFGS',
              loss_tol=1e-06,
              gradient_tol=1e-06,
              maxiter=1000,
              ridge=False,
              **kwargs):
    """
    Parameters
    ----------
    init_values : 1D ndarray.
        Should contain the initial values to start the optimizatin process
        with. There should be one value for each nest parameter and utility
        coefficient in the model.
    design_matrix : 2D ndarray.
        There should be one row per observation per available alternative.
        There should be one column per utility coefficient being estimated.
        All elements should be ints, floats, or longs.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a `1` and a zero otherwise.
    rows_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix maps the rows of the design
        matrix to the unique observations (on the columns).
    rows_to_nests : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per nest. This matrix maps the rows of the design matrix to
        the unique nests (on the columns).
    chosen_row_to_obs : 2D scipy sparse array.
        There should be one row per observation per available alternative and
        one column per observation. This matrix indicates, for each observation
        (on the columns), which rows of the design matrix were the realized
        outcome.
    constrained_pos : list, or None, optional.
        Denotes the positions of the array of estimated parameters that are not
        to change from their initial values. If a list is passed, the elements
        are to be integers where no such integer is is greater than
        `init_values.size`.
    print_res : bool, optional.
        Determines whether the timing and initial and final log likelihood
        results will be printed as they they are determined. Default = True.
    method : string, optional.
        Should be a valid string which can be passed to
        `scipy.optimize.minimize`. Determines the optimization algorithm which
        is used for this problem. Default == 'BFGS'.
    loss_tol : float, optional.
        Determines the tolerance on the difference in objective function values
        from one iteration to the next that is needed to determine convergence.
        Default `== 1e-06`.
    gradient_tol : float, optional.
        Determines the tolerance on the difference in gradient values from one
        iteration to the next which is needed to determine convergence.
        Default `== 1e-06`.
    maxiter : int, optional.
        Determines the maximum number of iterations used by the optimizer.
        Default `== 1000`.
    ridge : int, float, long, or None, optional.
        Determines whether or not ridge regression is performed. If a scalar is
        passed, then that scalar determines the ridge penalty for the
        optimization. Default `== None`.

    Returns
    -------
    results : dict.
        Contains the estimation results dictionary from scipy.optimize.minimize
        as well as various calculated quantities. Custom added keys will be:
        `["utility_coefs", "intercept_params", "shape_params", "nest_params",
          "log_likelihood_null", "rho_squared", "rho_bar_squared",
          "final_gradient", "final_hessian", "fisher_info", "constrained_pos",
          "final_log_likelihood", "chosen_probs", "long_probs, "residuals",
          "ind_chi_squareds"]`
    """
    ##########
    # Make sure we have the correct dimensions for the initial parameter values
    ##########
    # Figure out how many shape parameters we should have and how many index
    # coefficients we should have
    num_nests = rows_to_nests.shape[1]
    num_index_coefs = design_matrix.shape[1]

    assumed_param_dimensions = num_index_coefs + num_nests
    if init_values.shape[0] != assumed_param_dimensions:
        msg = "The initial values are of the wrong dimension"
        msg_1 = "It should be of dimension {}".format(assumed_param_dimensions)
        msg_2 = "But instead it has dimension {}".format(init_values.shape[0])
        raise ValueError(msg + msg_1 + msg_2)

    ##########
    # Check other function arguments for 'correctness'
    ##########
    # Make sure the ridge regression parameter is None or a real scalar
    ensure_ridge_is_scalar_or_none(ridge)

    ##########
    # Begin the model estimation process.
    ##########
    # Isolate the initial shape, intercept, and beta parameters.
    init_nest_params, init_betas = split_params(init_values, rows_to_nests)

    # Get the log-likelihood at zero and the initial log likelihood
    # Note, we use intercept_params=None since this will cause the function
    # to think there are no intercepts being added to the transformation
    # vector, which is the same as adding zero to the transformation vector
    basic_nest_params = np.ones(init_nest_params.shape[0])
    zero_beta_vals = np.zeros(init_betas.shape[0])
    log_likelihood_at_zero = general_log_likelihood(basic_nest_params,
                                                    zero_beta_vals,
                                                    design_matrix,
                                                    rows_to_obs,
                                                    rows_to_nests,
                                                    choice_vector,
                                                    ridge=ridge)

    # Note the nested logit model has no shape parameters so there is no such
    # parameter included in the log-likelihood calculations below or above
    natural_init_nest_params = nc.naturalize_nest_coefs(init_nest_params)
    initial_log_likelihood = general_log_likelihood(natural_init_nest_params,
                                                    init_betas,
                                                    design_matrix,
                                                    rows_to_obs,
                                                    rows_to_nests,
                                                    choice_vector,
                                                    ridge=ridge)
    ##########
    # Print initial model conditions
    # i.e., log-likelihoods at 'zero' and at initial values
    ##########
    if print_results:
        # Print the log-likelihood at zero
        print("Log-likelihood at zero: {:,.4f}".format(log_likelihood_at_zero))

        # Print the log-likelihood at the starting values
        print("Initial Log-likelihood: {:,.4f}".format(initial_log_likelihood))
        sys.stdout.flush()

    ##########
    # Perform the minimization to estimate the nested logit model
    ##########
    # Start timing the estimation process
    start_time = time.time()

    results = minimize(calc_neg_nested_log_likelihood_and_gradient,
                       init_values,
                       args=(design_matrix,
                             choice_vector,
                             rows_to_obs,
                             rows_to_nests,
                             constrained_pos,
                             ridge),
                       method=method,
                       jac=True,
                       tol=loss_tol,
                       options={'gtol': gradient_tol,
                                "maxiter": maxiter},
                       **kwargs)

    #########
    # Store the raw and processed outputs of the estimation outputs
    #########
    # Calculate the final log-likelihood. Note the '-1' is because we minimized
    # the negative log-likelihood but we want the actual log-likelihood
    final_log_likelihood = -1 * results["fun"]

    # Stop timing the estimation process and report the timing results
    end_time = time.time()
    if print_results:
        elapsed_sec = (end_time - start_time)
        elapsed_min = elapsed_sec / 60.0
        if elapsed_min > 1.0:
            print("Estimation Time: {:.2f} minutes.".format(elapsed_min))
        else:
            print("Estimation Time: {:.2f} seconds.".format(elapsed_sec))
        print("Final log-likelihood: {:,.4f}".format(final_log_likelihood))
        sys.stdout.flush()

    # Isolate the final shape, intercept, and beta parameters
    split_res = split_params(results.x, rows_to_nests)
    final_nest_params, final_utility_coefs = split_res
    natural_final_nest_params = nc.naturalize_nest_coefs(final_nest_params)

    # Store the separate values of the nest, shape, intercept, and beta
    # parameters in the estimation results dict
    results["utility_coefs"] = final_utility_coefs
    results["intercept_params"] = None
    results["shape_params"] = None
    results["nest_params"] = final_nest_params

    # Calculate the predicted probabilities
    c2obs = chosen_row_to_obs
    desired_res = "long_and_chosen_probs"
    probability_results = general_calc_probabilities(natural_final_nest_params,
                                                     final_utility_coefs,
                                                     design_matrix,
                                                     rows_to_obs,
                                                     rows_to_nests,
                                                     chosen_row_to_obs=c2obs,
                                                     return_type=desired_res)

    prob_of_chosen_alternatives, long_probs = probability_results

    # Calculate the residual vector
    residuals = choice_vector - long_probs

    # Calculate the observation specific chi-squared components
    chi_squared_terms = np.square(residuals) / long_probs
    individual_chi_squareds = rows_to_obs.T.dot(chi_squared_terms)

    # Store the log-likelihood at zero
    results["log_likelihood_null"] = log_likelihood_at_zero

    # Calculate and store the rho-squared and rho-bar-squared
    results["rho_squared"] = 1.0 - (final_log_likelihood /
                                    log_likelihood_at_zero)
    results["rho_bar_squared"] = 1.0 - ((final_log_likelihood -
                                         results.x.shape[0]) /
                                        log_likelihood_at_zero)

    # Calculate and store the final gradient
    results["final_gradient"] = general_gradient(final_nest_params,
                                                 final_utility_coefs,
                                                 design_matrix,
                                                 choice_vector,
                                                 rows_to_obs,
                                                 rows_to_nests,
                                                 ridge=ridge)

    # Calculate and store the final hessian
#    results["final_hessian"] = general_hessian(final_utility_coefs,
#                                               design_matrix,
#                                               alt_id_vector,
#                                               rows_to_obs,
#                                               rows_to_alts,
#                                               easy_utility_transform,
#                                               _cloglog_transform_deriv_c,
#                                               easy_calc_dh_dv,
#                                               calc_dh_d_alpha,
#                                               block_matrix_indices,
#                                               final_intercept_params,
#                                               final_shape_params,
#                                               ridge)
    # Use the BHHH approximation for now, since the analytic hessian seems
    # hard to compute.
    results["final_hessian"] = bhhh_approx(final_nest_params,
                                           final_utility_coefs,
                                           design_matrix,
                                           choice_vector,
                                           rows_to_obs,
                                           rows_to_nests,
                                           ridge=ridge)
    # Account for the 'constrained' parameters
    for idx_val in constrained_pos:
        results["final_hessian"][idx_val, :] = 0
        results["final_hessian"][:, idx_val] = 0
        results["final_hessian"][idx_val, idx_val] = -1

    # Calculate and store the final fisher information matrix
    # Use a placeholder value for now, especially since what I was using as
    # the fisher information matrix I'm now calling the bhhh_approximation
    # and I'm directly using it to approximate the hessian, as opposed to
    # simply using this to compute the sandwhich estimator
    results["fisher_info"] = np.diag(-1 * np.ones(results.x.shape[0]))
#    results["fisher_info"] = cc.calc_fischer_info_matrix(
#                                                          final_utility_coefs,
#                                                                design_matrix,
#                                                                alt_id_vector,
#                                                                  rows_to_obs,
#                                                                 rows_to_alts,
#                                                                choice_vector,
#                                                       easy_utility_transform,
#                                                   _cloglog_transform_deriv_c,
#                                                              easy_calc_dh_dv,
#                                                              calc_dh_d_alpha,
#                                                       final_intercept_params,
#                                                           final_shape_params,
#                                                                        ridge)

    # Record which parameters were constrained during estimation
    results["constrained_pos"] = constrained_pos

    # Add all miscellaneous objects that we need to store to the results dict
    results["final_log_likelihood"] = final_log_likelihood
    results["chosen_probs"] = prob_of_chosen_alternatives
    results["long_probs"] = long_probs
    results["residuals"] = residuals
    results["ind_chi_squareds"] = individual_chi_squareds

    return results


# Define the nested logit model class
class NestedLogit(base_mcm.MNDC_Model):
    """
    Parameters
    ----------
    data : string or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV file
        containing the long format data for this choice model. Note long format
        is has one row per available alternative for each observation. If
        pandas dataframe, the dataframe should be the long format data for the
        choice model.
    alt_id_col : string.
        Should denote the column in `data` that contains the identifiers of
        each row's corresponding alternative.
    obs_id_col : string.
        Should denote the column in `data` that contains the identifiers of
        each row's corresponding observation.
    choice_col : string.
        Should denote the column in data which contains the ones and zeros that
        denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in `long_form_df`. Values are
        either a list or a single string, `'all_diff' or 'all_same'`. If a
        list, the elements should be:

            - single objects that are within the alternative ID column of
              `long_form_df`
            - lists of objects that are within the alternative ID column of
              `long_form_df`.

        For each single object in the list, a unique column will be created
        (i.e. there will be a unique coefficient for that variable in the
        corresponding utility equation of the corresponding alternative). For
        lists within the specification_dict values, a single column will be
        created for all the alternatives within iterable (i.e. there will be
        one common coefficient for the variables in the iterable).
    names : OrderedDict, optional.
        Should have the same keys as `specification`. For each key:

            - if the corresponding value in `specification` is `'all_same'`,
              then there should be a single string as the value in names.
            - if the corresponding value in `specification` is `'all_diff'`,
              then there should be a list of strings as the value in names.
              There should be one string in the value in names for each
              possible alternative.
            - if the corresponding value in `specification` is a list, then
              there should be a list of strings as the value in names. There
              should be one string the value in names per item in the value in
              `specification`.

        Default == None.
    nest_spec : OrderedDict, optional.
        This kwarg MUST be passed for nested logit models. It is optional only
        for api compatibility. Keys are strings that define the name of the
        nests. Values are lists of alternative ids, denoting which alternatives
        belong to which nests. Each alternative id must only be associated with
        a single nest! Default `== None`.
    model_type : str, optional.
        Denotes the model type of the choice model being instantiated.
        Default == "".
    """
    def __init__(self,
                 data,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 names=None,
                 nest_spec=None,
                 **kwargs):
        ##########
        # Print a helpful message if nest_spec has not been included
        ##########
        condition_1 = nest_spec is None and "nest_spec" not in kwargs
        condition_2 = "nest_spec" in kwargs and kwargs["nest_spec"] is None
        missing_nest_spec = condition_1 or condition_2
        if missing_nest_spec:
            msg = "The Nested Logit Model REQUIRES a nest specification dict."
            raise ValueError(msg)

        ##########
        # Carry out the common instantiation process for all choice models
        ##########
        model_name = model_type_to_display_name["Nested Logit"]
        super(NestedLogit, self).__init__(data,
                                          alt_id_col,
                                          obs_id_col,
                                          choice_col,
                                          specification,
                                          names=names,
                                          nest_spec=nest_spec,
                                          model_type=model_name)

        ##########
        # Store the utility transform function
        ##########
        self.utility_transform = lambda x, *args, **kwargs: x[:, None]

        return None

    def fit_mle(self,
                init_vals,
                constrained_pos=None,
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
            Should containn the initial values to start the optimization
            process with. There should be one value for each nest parameter
            and utility coefficient. Nest parameters not being estimated
            should still be included. Handle these parameters using the
            `constrained_pos` kwarg.
        constrained_pos : list, or None, optional.
            Denotes the positions of the array of estimated parameters that are
            not to change from their initial values. If a list is passed, the
            elements are to be integers where no such integer is greater than
            `init_values.size`.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
        method : str, optional.
            Should be a valid string which can be passed to
            `scipy.optimize.minimize`. Determines the optimization algorithm
            which is used for this problem.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next which is needed to determine
            convergence. Default `== 1e-06`.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
            Default `== 1e-06`.
        ridge : int, float, long, or None, optional.
            Determines whether ridge regression is performed. If a scalar is
            passed, then that scalar determines the ridge penalty for the
            optimization. Default `== None`.

        Returns
        -------
        None. Estimation results are saved to the model instance.
        """
        # Check integrity of passed arguments
        kwargs_to_be_ignored = ["init_shapes", "init_intercepts", "init_coefs"]
        if any([x in kwargs for x in kwargs_to_be_ignored]):
            msg = "Nested Logit model does not use the following kwargs:\n{}"
            msg_2 = "Remove such kwargs and pass a single init_vals argument"
            raise ValueError(msg.format(kwargs_to_be_ignored) + msg_2)

        # Store the optimization method
        self.optimization_method = method

        # Store the ridge parameter
        self.ridge_param = ridge

        if ridge is not None:
            warnings.warn(_ridge_warning_msg)

        # Construct the mappings from alternatives to observations and from
        # chosen alternatives to observations
        mapping_res = self.get_mappings_for_fit()
        rows_to_obs = mapping_res["rows_to_obs"]
        rows_to_nests = mapping_res["rows_to_nests"]
        chosen_row_to_obs = mapping_res["chosen_row_to_obs"]

        # Determine the degenerate nests whose nesting parameters are to be
        # constrained to one. Note the following functions assume that the nest
        # parameters are placed before the index coefficients.
        fixed_params = identify_degenerate_nests(self.nest_spec)

        # Include the user specified parameters that are to be constrained to
        # their initial values
        if constrained_pos is not None:
            fixed_params.extend(constrained_pos)
        final_constrained_pos = sorted(list(set(fixed_params)))

        # Get the estimation results
        estimation_res = _estimate(init_vals,
                                   self.design,
                                   self.choices,
                                   rows_to_obs,
                                   rows_to_nests,
                                   chosen_row_to_obs,
                                   final_constrained_pos,
                                   print_results=print_res,
                                   method=method,
                                   loss_tol=loss_tol,
                                   gradient_tol=gradient_tol,
                                   maxiter=maxiter,
                                   ridge=ridge,
                                   **kwargs)

        # Store the estimation results
        self.store_fit_results(estimation_res)

        return None
