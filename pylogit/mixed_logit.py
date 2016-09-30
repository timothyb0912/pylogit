# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 18:15:50 2016

@name:      Mixed MultiNomial Logit
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating mixed multinomial logit
            models (with the help of the "base_multinomial_cm.py" file).
            Version 1 only works for MNL kernels and only for mixing of index
            coefficients.

General References
------------------
Train, K., 2009. Discrete Choice Models With Simulation. 2 ed., Cambridge
    University Press, New York, NY, USA.
"""

import time
import sys
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

import base_multinomial_cm_v2 as base_mcm
import choice_calcs as cc
import mixed_logit_calcs as mlc
from choice_tools import get_dataframe_from_data
from choice_tools import create_design_matrix
from choice_tools import create_long_form_mappings

# Alias necessary functions for model estimation
general_calc_probabilities = cc.calc_probabilities
general_sequence_probs = mlc.calc_choice_sequence_probs
general_log_likelihood = mlc.calc_mixed_log_likelihood
general_gradient = mlc.calc_mixed_logit_gradient
general_bhhh = mlc.calc_bhhh_hessian_approximation_mixed_logit

_msg_1 = "The Mixed MNL Model has no shape parameters. "
_msg_2 = "shape_names and shape_ref_pos will be ignored if passed."
_shape_ignore_msg = _msg_1 + _msg_2


def mnl_utility_transform(sys_utility_array, *args, **kwargs):
    """
    Parameters
    ----------
    sys_utility_array : ndarray.
        Should have 1D or 2D. Should have been created by the dot product of a
        design matrix and an array of index coefficients.

    Returns
    -------
        systematic_utilities : 2D ndarray.
            The input systematic utilities. If `sys_utility_array` is 2D, then
            `sys_utility_array` is returned. Else, returns
            `sys_utility_array[:, None]`.
    """
    # Return a 2D array of systematic utility values
    if len(sys_utility_array.shape) == 1:
        systematic_utilities = sys_utility_array[:, np.newaxis]
    else:
        systematic_utilities = sys_utility_array

    return systematic_utilities


def _estimate(init_values,
              design_3d,
              alt_id_vector,
              choice_vector,
              mapping_matrices,
              constrained_pos=None,
              print_results=True,
              method='newton-cg',
              loss_tol=1e-06,
              gradient_tol=1e-06,
              maxiter=1000,
              ridge=None,
              **kwargs):
    """
    Parameters
    ----------
    init_vals : 1D ndarray.
        1D numpy array of the initial values to start the optimizatin process
        with. There should be one value for each index coefficient being
        estimated.
    design_3d : 2D ndarray.
        2D numpy array with one row per observation per available alternative.
        There should be one column per index coefficient being estimated. All
        elements should be ints, floats, or longs.
    alt_id_vector : 1D ndarray.
        All elements should be ints. There should be one row per observation
        per available alternative for the given observation. Elements denote
        the alternative corresponding to the given row of the design matrix.
    choice_vector : 1D ndarray.
        All elements should be either ones or zeros. There should be one row
        per observation per available alternative for the given observation.
        Elements denote the alternative which is chosen by the given
        observation with a 1 and a zero otherwise.
    constrained_pos : list or None, optional.
        Denotes the positions of the array of estimated parameters that are not
        to change from their initial values. If a list is passed, the elements
        are to be integers where no such integer is is greater than
        init_values.size.
    print_res : bool, optional.
        Determines whether the timing and initial and final log likelihood
        results will be printed as they they are determined. Default = True.
    method : str, optional.
        Should be a valid string which can be passed to
        scipy.optimize.minimize. Determines the optimization algorithm which is
        used for this problem. Default = 'newton-cg'.
    loss_tol : float, optional.
        Determines the tolerance on the difference in objective function values
        from one iteration to the next which is needed to determine
        convergence. Default = 1e-06.
    gradient_tol : float, optional.
        Determines the tolerance on the difference in gradient values from one
        iteration to the next which is needed to determine convergence.
        Default = 1e-06.
    maxiter : int, optional.
        Determines the maximum number of iterations used by the optimizer.
        Default = 1000.
    ridge : int, float, long, or None. OPTIONAl.
        Determines whether or not ridge regression is performed. If a scalar is
        passed, then that scalar determines the ridge penalty for the
        optimization. Default = None.

    Returns
    -------
    results : dict.
        Result dictionary returned by `scipy.optimize.minimize`. In addition to
        the generic key-value pairs that are returned, `results` will have the
        folowing keys:
        - "final_log_likelihood"
        - "long_probs"
        - "residuals"
        - "ind_chi_squareds"
        - "simulated_sequence_probs"
        - "expanded_sequence_probs"
        - "utility_coefs"
        - "shape_params"
        - "intercept_params"
        - "nest_params"
        - "log_likelihood_null"
        - "rho_squared"
        - "rho_bar_squared"
        - "final_gradient"
        - "final_hessian"
        - "fisher_info"
        - "constrained_pos"

    """
    # Make sure we have the correct dimensions for the initial parameter values
    try:
        assert init_values.shape[0] == design_3d.shape[2]
    except AssertionError as e:
        print("The initial values are of the wrong dimension")
        print("They should be of dimension {}".format(design_3d.shape[2]))
        raise e

    # Make sure the ridge regression parameter is None or a real scalar
    try:
        assert ridge is None or isinstance(ridge, (int, float, long))
    except AssertionError as e:
        print("ridge should be None or an int, float, or long.")
        print("The passed value of ridge had type: {}".format(type(ridge)))
        raise e

    # Get the required mapping matrices for estimation and prediction
    rows_to_obs = mapping_matrices["rows_to_obs"]
    rows_to_alts = mapping_matrices["rows_to_alts"]
    rows_to_mixers = mapping_matrices["rows_to_mixers"]
    chosen_row_to_obs = mapping_matrices["chosen_row_to_obs"]

    # Calculate the null-log-likelihood
    log_like_args = (np.zeros(design_3d.shape[2]),
                     design_3d,
                     alt_id_vector,
                     rows_to_obs,
                     rows_to_alts,
                     rows_to_mixers,
                     choice_vector,
                     mnl_utility_transform)
    log_likelihood_at_zero = general_log_likelihood(*log_like_args,
                                                    ridge=ridge)

    # Calculate the initial log-likelihood
    log_like_args = (init_values,
                     design_3d,
                     alt_id_vector,
                     rows_to_obs,
                     rows_to_alts,
                     rows_to_mixers,
                     choice_vector,
                     mnl_utility_transform)
    initial_log_likelihood = general_log_likelihood(*log_like_args,
                                                    ridge=ridge)
    if print_results:
        # Print the log-likelihood at zero
        msg = "Log-likelihood at zero: {:,.4f}"
        print(msg.format(log_likelihood_at_zero))

        # Print the log-likelihood at the starting values
        print("Initial Log-likelihood: {:,.4f}".format(initial_log_likelihood))
        sys.stdout.flush()

    # Start timing the estimation process
    start_time = time.time()

    results = minimize(mlc.calc_neg_log_likelihood_and_neg_gradient,
                       init_values,
                       args=(design_3d,
                             alt_id_vector,
                             rows_to_obs,
                             rows_to_alts,
                             rows_to_mixers,
                             choice_vector,
                             mnl_utility_transform,
                             constrained_pos,
                             ridge),
                       method=method,
                       jac=True,
                       hess=general_bhhh,
                       tol=loss_tol,
                       options={'gtol': gradient_tol,
                                "maxiter": maxiter})

    # Calculate the final log-likelihood. Note the '-1' is because we minimized
    # the negative log-likelihood but we want the actual log-likelihood
    final_log_likelihood = -1 * results["fun"]
    results["final_log_likelihood"] = final_log_likelihood

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

    # Calculate the predicted probabilities
    prob_args = (results.x,
                 design_3d,
                 alt_id_vector,
                 rows_to_obs,
                 rows_to_alts,
                 mnl_utility_transform)
    prob_kwargs = {"chosen_row_to_obs": chosen_row_to_obs,
                   "return_long_probs": True}
    probability_results = general_calc_probabilities(*prob_args, **prob_kwargs)
    prob_of_chosen_alternatives, long_probs = probability_results
    results["long_probs"] = long_probs
    results["chosen_probs"] = prob_of_chosen_alternatives

    # Calculate the model residuals
    residuals = choice_vector[:, None] - long_probs
    results["residuals"] = residuals

    # Calculate the observation specific chi-squared components
    chi_squared_terms = np.square(residuals) / long_probs
    results["ind_chi_squareds"] = rows_to_obs.T.dot(chi_squared_terms)

    # Get the probability of each sequence of choices, given the draws
    prob_res = mlc.calc_choice_sequence_probs(long_probs,
                                              choice_vector,
                                              rows_to_mixers,
                                              return_type='all')
    results["simulated_sequence_probs"] = prob_res[0]
    results["expanded_sequence_probs"] = prob_res[1]

    # Store supplementary objects in the estimation results dict
    results["utility_coefs"] = results.x
    results["shape_params"] = None
    results["intercept_params"] = None
    results["nest_params"] = None

    # Store the log-likelihood at zero
    results["log_likelihood_null"] = log_likelihood_at_zero

    # Calculate and store the rho-squared and rho-bar-squared
    results["rho_squared"] = 1.0 - (final_log_likelihood /
                                    log_likelihood_at_zero)
    results["rho_bar_squared"] = 1.0 - ((final_log_likelihood -
                                         results.x.shape[0]) /
                                        log_likelihood_at_zero)

    # Calculate and store the final gradient
    grad_args = (results.x,
                 design_3d,
                 alt_id_vector,
                 rows_to_obs,
                 rows_to_alts,
                 rows_to_mixers,
                 choice_vector,
                 mnl_utility_transform)
    results["final_gradient"] = general_gradient(*grad_args, ridge=ridge)
    # Calculate and store the final hessian
    # Note this is somewhat of a hack since we're using the BHHH approximation
    # to the hessian as the actual hessian, instead of simply using it to
    # approximate the Huber-White 'robust' asymptotic covariance matrix
    results["final_hessian"] = general_bhhh(*grad_args, ridge=ridge)

    # Account for the 'constrained' parameters when presenting hessian results
    if constrained_pos is not None:
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

    # Record which parameters were constrained during estimation
    results["constrained_pos"] = constrained_pos

    return results


class MixedLogit(base_mcm.MNDC_Model):
    """
    Parameters
    ----------
    data : string or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV
        file containing the long format data for this choice model. Note
        long format has one row per available alternative for each
        observation. If pandas dataframe, the dataframe should be the long
        format data for the choice model.
    alt_id_col : str.
        Should denote the column in data which contains the alternative
        identifiers for each row.
    obs_id_col : str.
        Should denote the column in data which contains the observation
        identifiers for each row.
    choice_col : str.
        Should denote the column in data which contains the ones and zeros
        that denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in long_form_df. Values are
        either a list or a single string, `all_diff` or `all_same`. If a
        list, the elements should be one of the following:

        - single objects that are within the alternative ID
          column of long_form_df
        - lists of objects that are within the alternative
          ID column of long_form_df.

        For each single object in the list, a unique column will be created
        (i.e. there will be a unique coefficient for that variable in the
        corresponding utility equation of the corresponding alternative).
        For lists within the specification_dict values, a single column
        will be created for all the alternatives within iterable (i.e.
        there will be one common coefficient for the variables in the
        iterable).
    names : OrderedDict, optional.
        Should have the same keys as `specification_dict`. For each key:

        - if the corresponding value in specification_dict is "all_same",
          then there should be a single string as the value in names.
        - if the corresponding value in specification_dict is "all_diff",
          then there should be a list of strings as the value in names.
          There should be one string in the value in names for each
        - if the corresponding value in specification_dict is a list, then
          there should be a list of strings as the value in names. There
          should be one string in the value in names per item in the value
          in specification_dict. Default == None.
    mixing_id_col : str, or None, optional.
        Should be a column heading in `data`. Should denote the column in
        `data` which contains the identifiers of the units of observation over
        which the coefficients of the model are thought to be randomly
        distributed. If `model_type == "Mixed Logit"`, then `mixing_id_col`
        must be passed. Default == None.
    mixing_vars : list, or None, optional.
        All elements of the list should be strings. Each string should be
        present in the values of `names.values()` and they're associated
        variables should only be index variables (i.e. part of the design
        matrix). If `model_type == "Mixed Logit"`, then `mixing_vars` must be
        passed. Default == None.

    Methods
    -------
    panel_predict(new_data, num_draws, return_long_probs, choice_col, seed)
        Predicts the probability of each individual in `new_data` making each
        possible choice in each choice situation they are faced with. This
        method differs from the `predict()` function by using 'individualized
        coefficient distributions' that are conditioned on each person's past
        choices and choice situations (if there are any).
    """
    def __init__(self,
                 data,
                 alt_id_col,
                 obs_id_col,
                 choice_col,
                 specification,
                 names=None,
                 mixing_id_col=None,
                 mixing_vars=None,
                 *args, **kwargs):
        ##########
        # Print a helpful message for users who have included shape parameters
        # or shape names unneccessarily
        ##########
        for keyword in ["shape_names", "shape_ref_pos"]:
            if keyword in kwargs and kwargs[keyword] is not None:
                warnings.warn(_shape_ignore_msg)
                break

        if "intercept_ref_pos" in kwargs:
            if kwargs["intercept_ref_pos"] is not None:
                msg = "All Mixed Logit intercepts should be in the index."
                raise ValueError(msg)

        # Carry out the common instantiation process for all choice models
        super(MixedLogit, self).__init__(data,
                                         alt_id_col,
                                         obs_id_col,
                                         choice_col,
                                         specification,
                                         names=names,
                                         model_type="Mixed Logit Model",
                                         mixing_id_col=mixing_id_col,
                                         mixing_vars=mixing_vars)

        # Store the utility transform function
        self.utility_transform = mnl_utility_transform

        return None

    def fit_mle(self,
                init_vals,
                num_draws,
                seed=None,
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
            Should contain the initial values to start the optimization process
            with. There should be one value for each utility coefficient and
            shape parameter being estimated.
        num_draws : int.
            Should be greater than zero. Denotes the number of draws that we
            are making from each normal distribution.
        seed : int or None, optional.
            If an int is passed, it should be greater than zero. Denotes the
            value to be used in seeding the random generator used to generate
            the draws from the normal distribution. Default == None.
        constrained_pos : list or None, optional.
            Denotes the positions of the array of estimated parameters that are
            not to change from their initial values. If a list is passed, the
            elements are to be integers where no such integer is greater than
            `init_values.size.` Default == None.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
        method : str, optional.
            Should be a valid string which can be passed to
            scipy.optimize.minimize. Determines the optimization algorithm
            that is used for this problem.
        loss_tol : float, optional.
            Determines the tolerance on the difference in objective function
            values from one iteration to the next which is needed to determine
            convergence. Default = 1e-06.
        gradient_tol : float, optional.
            Determines the tolerance on the difference in gradient values from
            one iteration to the next which is needed to determine convergence.
            Default = 1e-06.
        maxiter : int, optional.
            Denotes the maximum number of iterations of the algorithm specified
            by `method` that will be used to estimate the parameters of the
            given model. Default == 1000.
        ridge : int, float, long, or None, optional.
            Determines whether or not ridge regression is performed. If a float
            is passed, then that float determines the ridge penalty for the
            optimization. Default = None.

        Returns
        -------
        None. Estimation results are saved to the model instance.
        """
        # Check integrity of passed arguments
        kwargs_to_be_ignored = ["init_shapes", "init_intercepts", "init_coefs"]
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
        mapping_res = self.get_mappings_for_fit()
        rows_to_mixers = mapping_res["rows_to_mixers"]

        # Get the draws for each random coefficient
        num_mixing_units = rows_to_mixers.shape[1]
        draw_list = mlc.get_normal_draws(num_mixing_units,
                                         num_draws,
                                         len(self.mixing_pos),
                                         seed=seed)

        # Create the 3D design matrix
        self.design_3d = mlc.create_expanded_design_for_mixing(self.design,
                                                               draw_list,
                                                               self.mixing_pos,
                                                               rows_to_mixers)

        # Get the estimation results
        estimation_res = _estimate(init_vals,
                                   self.design_3d,
                                   self.alt_IDs,
                                   self.choices,
                                   mapping_res,
                                   constrained_pos=constrained_pos,
                                   print_results=print_res,
                                   method=method,
                                   loss_tol=loss_tol,
                                   gradient_tol=gradient_tol,
                                   maxiter=maxiter,
                                   ridge=ridge)

        # Store the estimation results
        self.store_fit_results(estimation_res)

        return None

    def __filter_past_mappings(self,
                               past_mappings,
                               long_inclusion_array):
        """
        Parameters
        ----------
        past_mappings : dict.
            All elements should be None or compressed sparse row matrices from
            scipy.sparse. The following keys should be in past_mappings:

            - "rows_to_obs",
            - "rows_to_alts",
            - "chosen_rows_to_obs",
            - "rows_to_nests",
            - "rows_to_mixers"

            The values that are not None should be 'mapping' matrices that
            denote which rows of the past long-format design matrix belong to
            which unique object such as unique observations, unique
            alternatives, unique nests, unique 'mixing' units etc.
        long_inclusion_array : 1D ndarray.
            Should denote, via a `1`, the rows of the past mapping matrices
            that are to be included in the filtered mapping matrices.

        Returns
        -------
        new_mappings : dict.
            The returned dictionary will be the same as `past_mappings` except
            that all the mapping matrices will have been filtered according to
            `long_inclusion_array`.
        """
        new_mappings = {}
        for key in past_mappings:
            if past_mappings[key] is None:
                new_mappings[key] = None
            else:
                mask_array = long_inclusion_array[:, None]
                orig_map = past_mappings[key]
                # Initialize the resultant array that is desired
                new_map = orig_map.multiply(np.tile(mask_array,
                                                    (1, orig_map.shape[1]))).A
                # Perform the desired filtering
                current_filter = (new_map.sum(axis=1) != 0)
                if current_filter.shape[0] > 0:
                    current_filter = current_filter.ravel()
                    new_map = new_map[current_filter, :]
                # Do the second filtering
                current_filter = (new_map.sum(axis=0) != 0)
                if current_filter.shape[0] > 0:
                    current_filter = current_filter.ravel()
                    new_map = new_map[:, current_filter]

                new_mappings[key] = csr_matrix(new_map)

        return new_mappings

    def panel_predict(self,
                      data,
                      num_draws,
                      return_long_probs=True,
                      choice_col=None,
                      seed=None):
        """
        Parameters
        ----------
        data : string or pandas dataframe.
            If string, data should be an absolute or relative path to a CSV
            file containing the long format data to be predicted with this
            choice model. Note long format has one row per available
            alternative for each observation. If pandas dataframe, the
            dataframe should be in long format.
        num_draws : int.
            Should be greater than zero. Denotes the number of draws being
            made from each mixing distribution for the random coefficients.
        return_long_probs : bool, optional.
            Indicates whether or not the long format probabilites (a 1D numpy
            array with oneelement per observation per available alternative)
            should be returned. Default == True.
        choice_col : str or None, optonal.
            Denotes the column in long_form which contains a one if the
            alternative pertaining to the given row was the observed outcome
            for the observation pertaining to the given row and a zero
            otherwise. If passed, then an array of probabilities of just the
            chosen alternative for each observation will be returned.
            Default == None.
        seed : int or None, optional.
            If an int is passed, it should be greater than zero. Denotes the
            value to be used in seeding the random generator used to generate
            the draws from the mixing distributions of each random coefficient.
            Default == None.

        Returns
        -------
        numpy array or tuple of two numpy arrays.
            - If `choice_col` is passed AND `return_long_probs` is True, then
              the tuple `(chosen_probs, pred_probs_long)` is returned.
            - If `return_long_probs` is True and `choice_col` is None, then
              only `pred_probs_long` is returned.
            - If `choice_col` is passed and `return_long_probs` is False then
              `chosen_probs` is returned.

            `chosen_probs` is a 1D numpy array of shape (num_observations,).
            Each element is the probability of the corresponding observation
            being associated with its realized outcome.

            `pred_probs_long` is a 1D numpy array with one element per
            observation per available alternative for that observation. Each
            element is the probability of the corresponding observation being
            associated with that row's corresponding alternative.

        Notes
        -----
        It is NOT valid to have `chosen_row_to_obs == None` and
        `return_long_probs == False`.
        """
        # Get the dataframe of observations we'll be predicting on
        dataframe = get_dataframe_from_data(data)

        # Determine the conditions under which we will add an intercept column
        # to our long format dataframe.
        condition_1 = "intercept" in self.specification
        condition_2 = "intercept" not in dataframe.columns

        if condition_1 and condition_2:
            dataframe["intercept"] = 1.0

        # Make sure the necessary columns are in the long format dataframe
        for column in [self.alt_id_col,
                       self.obs_id_col,
                       self.mixing_id_col]:
            if column is not None:
                try:
                    assert column in dataframe.columns
                except AssertionError as e:
                    print("{} not in data.columns".format(column))
                    raise e

        # Get the new column of alternative IDs and get the new design matrix
        new_alt_IDs = dataframe[self.alt_id_col].values

        new_design_res = create_design_matrix(dataframe,
                                              self.specification,
                                              self.alt_id_col,
                                              names=self.name_spec)
        new_design_2d = new_design_res[0]

        # Get the new mappings between the alternatives and observations
        mapping_res = create_long_form_mappings(dataframe,
                                                self.obs_id_col,
                                                self.alt_id_col,
                                                choice_col=choice_col,
                                                nest_spec=self.nest_spec,
                                                mix_id_col=self.mixing_id_col)

        new_rows_to_obs = mapping_res["rows_to_obs"]
        new_rows_to_alts = mapping_res["rows_to_alts"]
        new_chosen_to_obs = mapping_res["chosen_row_to_obs"]
        new_rows_to_mixers = mapping_res["rows_to_mixers"]

        # Determine the coefficients being used for prediction.
        # Note that I am making an implicit assumption (for now) that the
        # kernel probabilities are coming from a logit-type model.
        new_index_coefs = self.coefs.values
        new_intercepts = (self.intercepts.values if self.intercepts
                          is not None else None)
        new_shape_params = (self.shapes.values if self.shapes
                            is not None else None)

        # Get the draws for each random coefficient
        num_mixing_units = new_rows_to_mixers.shape[1]
        draw_list = mlc.get_normal_draws(num_mixing_units,
                                         num_draws,
                                         len(self.mixing_pos),
                                         seed=seed)

        # Calculate the 3D design matrix for the prediction.
        design_args = (new_design_2d,
                       draw_list,
                       self.mixing_pos,
                       new_rows_to_mixers)
        new_design_3d = mlc.create_expanded_design_for_mixing(*design_args)
        # Calculate the desired probabilities for the mixed logit model.
        prob_args = (new_index_coefs,
                     new_design_3d,
                     new_alt_IDs,
                     new_rows_to_obs,
                     new_rows_to_alts,
                     mnl_utility_transform)
        prob_kwargs = {"intercept_params": new_intercepts,
                       "shape_params": new_shape_params,
                       "return_long_probs": True}
        # Note that I am making an implicit assumption (for now) that the
        # kernel probabilities are coming from a logit-type model.
        new_kernel_probs = general_calc_probabilities(*prob_args,
                                                      **prob_kwargs)

        # Initialize and calculate the weights needed for prediction with
        # "individualized" coefficient distributions. Should have shape
        # (new_row_to_mixer.shape[1], num_draws)
        weights_per_ind_per_draw = (1.0 / num_draws *
                                    np.ones((new_rows_to_mixers.shape[1],
                                             num_draws)))

        ##########
        # Create an array denoting the observation ids that are present in both
        # the dataset to be predicted and the dataset used for model estimation
        ##########
        # Get the old mixing ids
        old_mixing_id_long = self.data[self.mixing_id_col].values
        # Get the new mixing ids
        new_mixing_id_long = dataframe[self.mixing_id_col].values
        # Get the unique individual ids from the original and preserve order
        orig_unique_id_idx_old = np.sort(np.unique(old_mixing_id_long,
                                                   return_index=True)[1])
        orig_unique_id_idx_new = np.sort(np.unique(new_mixing_id_long,
                                                   return_index=True)[1])
        # Get the unique ids, in their original order of appearance
        orig_order_unique_ids_old = old_mixing_id_long[orig_unique_id_idx_old]
        orig_order_unique_ids_new = new_mixing_id_long[orig_unique_id_idx_new]

        # Figure out which long format rows have ids are common to both
        # datasets
        old_repeat_mixing_id_idx = np.in1d(old_mixing_id_long,
                                           orig_order_unique_ids_new)
        # Figure out which unique ids are in both datasets
        old_unique_mix_id_repeats = np.in1d(orig_order_unique_ids_old,
                                            orig_order_unique_ids_new)
        new_unique_mix_id_repeats = np.in1d(orig_order_unique_ids_new,
                                            orig_order_unique_ids_old)

        # Get the 2d design matrix used to estimate the model, and filter it
        # to only those individuals for whom we are predicting new choice
        # situations.
        past_design_2d = self.design[old_repeat_mixing_id_idx, :]

        ##########
        # Appropriately filter the old mapping matrix that maps rows of the
        # long format design matrix to unique mixing units.
        ##########
        orig_mappings = self.get_mappings_for_fit()
        past_mappings = self.__filter_past_mappings(orig_mappings,
                                                    old_repeat_mixing_id_idx)

        # Create the 3D design matrix for those choice situations, using the
        # draws that were just taken from the mixing distributions of interest.
        past_draw_list = [x[new_unique_mix_id_repeats, :] for x in draw_list]
        design_args = (past_design_2d,
                       past_draw_list,
                       self.mixing_pos,
                       past_mappings["rows_to_mixers"])
        past_design_3d = mlc.create_expanded_design_for_mixing(*design_args)

        # Get the kernel probabilities of each of the alternatives for each
        # each of the previoius choice situations, given the current draws of
        # of the random coefficients
        prob_args = (new_index_coefs,
                     past_design_3d,
                     self.alt_IDs[old_repeat_mixing_id_idx],
                     past_mappings["rows_to_obs"],
                     past_mappings["rows_to_alts"],
                     mnl_utility_transform)
        prob_kwargs = {"return_long_probs": True}
        past_kernel_probs = mlc.general_calc_probabilities(*prob_args,
                                                           **prob_kwargs)

        ##########
        # Calculate the old sequence probabilities of all the individual's
        # for whom we have recorded observations and for whom we are predicting
        # future choice situations
        ##########
        past_choices = self.choices[old_repeat_mixing_id_idx]
        sequence_args = (past_kernel_probs,
                         past_choices,
                         past_mappings["rows_to_mixers"])
        seq_kwargs = {"return_type": 'all'}
        old_sequence_results = mlc.calc_choice_sequence_probs(*sequence_args,
                                                              **seq_kwargs)
        # Note sequence_probs_per_draw should have shape
        past_sequence_probs_per_draw = old_sequence_results[1]
        # Calculate the weights for each individual who has repeat observations
        # in the previously observed dataset
        past_weights = (past_sequence_probs_per_draw /
                        past_sequence_probs_per_draw.sum(axis=1)[:, None])
        # Rearrange the past weights to match the current ordering of the
        # unique observations
        rel_new_ids = orig_order_unique_ids_new[new_unique_mix_id_repeats]
        num_rel_new_id = rel_new_ids.shape[0]
        new_unique_mix_id_repeats_2d = rel_new_ids.reshape((num_rel_new_id, 1))
        rel_old_ids = orig_order_unique_ids_old[old_unique_mix_id_repeats]
        num_rel_old_id = rel_old_ids.shape[0]
        old_unique_mix_id_repeats_2d = rel_old_ids.reshape((1, num_rel_old_id))
        new_to_old_repeat_ids = csr_matrix(new_unique_mix_id_repeats_2d ==
                                           old_unique_mix_id_repeats_2d)
        past_weights = new_to_old_repeat_ids.dot(past_weights)

        # Map these weights to earlier initialized weights
        weights_per_ind_per_draw[new_unique_mix_id_repeats, :] = past_weights

        # Create a 'long' format version of the weights array. This version
        # should have the same number of rows as the new kernel probs but the
        # same number of columns as the weights array (aka the number of draws)
        weights_per_draw = new_rows_to_mixers.dot(weights_per_ind_per_draw)

        # Calculate the predicted probabilities of each alternative for each
        # choice situation being predicted
        pred_probs_long = (weights_per_draw * new_kernel_probs).sum(axis=1)
        # Note I am assuming pred_probs_long should be 1D (as should be the
        # case if we are predicting with one set of betas and one 2D data
        # object)
        pred_probs_long = pred_probs_long.ravel()

        # Format the returned objects according to the user's desires.
        if new_chosen_to_obs is None:
            chosen_probs = None
        else:
            # chosen_probs will be of shape (num_observations,)
            chosen_probs = new_chosen_to_obs.transpose().dot(pred_probs_long)
            if len(chosen_probs.shape) > 1 and chosen_probs.shape[1] > 1:
                pass
            else:
                chosen_probs = chosen_probs.ravel()

        # Return the long form and chosen probabilities if desired
        if return_long_probs and chosen_probs is not None:
            return chosen_probs, pred_probs_long
        # If working with predictions, return just the long form probabilities
        elif return_long_probs and chosen_probs is None:
            return pred_probs_long
        # If estimating the model and storing fitted probabilities or
        # testing the model on data for which we know the chosen alternative,
        # just return the chosen probabilities.
        elif chosen_probs is not None:
            return chosen_probs
        else:
            msg = "chosen_row_to_obs is None AND return_long_probs == False"
            raise Exception(msg)

        return None
