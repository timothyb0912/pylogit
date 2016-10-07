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

import warnings
import numpy as np
from scipy.sparse import csr_matrix

import base_multinomial_cm_v2 as base_mcm
import choice_calcs as cc
import mixed_logit_calcs as mlc
from choice_tools import get_dataframe_from_data
from choice_tools import create_design_matrix
from choice_tools import create_long_form_mappings
from display_names import model_type_to_display_name
from estimation import EstimationObj
from estimation import estimate

# Alias necessary functions for model estimation
general_calc_probabilities = cc.calc_probabilities
general_sequence_probs = mlc.calc_choice_sequence_probs
general_log_likelihood = mlc.calc_mixed_log_likelihood
general_gradient = mlc.calc_mixed_logit_gradient
general_bhhh = mlc.calc_bhhh_hessian_approximation_mixed_logit

_msg_1 = "The Mixed MNL Model has no shape parameters. "
_msg_2 = "shape_names and shape_ref_pos will be ignored if passed."
_shape_ignore_msg = _msg_1 + _msg_2


def split_param_vec(beta, return_all_types=False, *args, **kwargs):
    """
    Parameters
    ----------
    beta : 1D numpy array.
        All elements should by ints, floats, or longs. Should have 1 element
        for each utility coefficient being estimated (i.e. num_features).
    return_all_types : bool, optional.
        Determines whether or not a tuple of 4 elements will be returned (with
        one element for the nest, shape, intercept, and index parameters for
        this model). If False, a tuple of 3 elements will be returned, as
        described below.

    Returns
    -------
    tuple.
        `(None, None, beta)`. This function is merely for compatibility with
        the other choice model files.

    Note
    ----
    If `return_all_types == True` then the function will return a tuple of four
    objects. In order, these objects will either be None or the arrays
    representing the arrays corresponding to the nest, shape, intercept, and
    index parameters.
    """
    if return_all_types:
        return None, None, None, beta
    else:
        return None, None, beta


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


def check_length_of_init_values(design_3d, init_values):
    """
    Ensures that the initial values are of the correct length, given the design
    matrix that they will be dot-producted with. Raises a ValueError if that is
    not the case, and provides a useful error message to users.

    Parameters
    ----------
    init_values : 1D ndarray.
        1D numpy array of the initial values to start the optimizatin process
        with. There should be one value for each index coefficient being
        estimated.
    design_3d : 2D ndarray.
        2D numpy array with one row per observation per available alternative.
        There should be one column per index coefficient being estimated. All
        elements should be ints, floats, or longs.

    Returns
    -------
    None.
    """
    if init_values.shape[0] != design_3d.shape[2]:
        msg_1 = "The initial values are of the wrong dimension. "
        msg_2 = "They should be of dimension {}".format(design_3d.shape[2])
        raise ValueError(msg_1 + msg_2)

    return None


def add_mixl_specific_results_to_estimation_res(estimator, results_dict):
    """
    Stores particular items in the results dictionary that are unique to mixed
    logit-type models. In particular, this function calculates and adds
    `sequence_probs` and `expanded_sequence_probs` to the results dictionary.
    The `constrained_pos` object is also stored to the results_dict.

    Parameters
    ----------
    estimator : an instance of the MixedEstimator class.
        Should contain a `choice_vector` attribute that is a 1D ndarray
        representing the choices made for this model's dataset. Should also
        contain a `rows_to_mixers` attribute that maps each row of the long
        format data to a unit of observation that the mixing is being performed
        over.
    results_dict : dict.
        This dictionary should be the dictionary returned from
        scipy.optimize.minimize. In particular, it should have the following
        `long_probs` key.

    Returns
    -------
    results_dict.
    """
    # Get the probability of each sequence of choices, given the draws
    prob_res = mlc.calc_choice_sequence_probs(results_dict["long_probs"],
                                              estimator.choice_vector,
                                              estimator.rows_to_mixers,
                                              return_type='all')
    # Add the various items to the results_dict.
    results_dict["simulated_sequence_probs"] = prob_res[0]
    results_dict["expanded_sequence_probs"] = prob_res[1]

    return results_dict


class MixedEstimator(EstimationObj):
    """
    Estimation object for the Mixed Logit Model.

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
          - design_3d
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
    """
    def __init__(self,
                 model_obj,
                 mapping_dict,
                 ridge,
                 zero_vector,
                 split_params,
                 constrained_pos=None):
        super(MixedEstimator, self).__init__(model_obj,
                                             mapping_dict,
                                             ridge,
                                             zero_vector,
                                             split_params,
                                             constrained_pos=constrained_pos)

        # Add the 3d design matrix to the object
        self.design_3d = model_obj.design_3d

        return None

    def check_length_of_initial_values(self, init_values):
        """
        Ensures that the initial values are of the correct length.
        """
        return check_length_of_init_values(self.design_3d, init_values)

    def convenience_calc_probs(self, params):
        """
        Calculates the probabilities of the chosen alternative, and the long
        format probabilities for this model and dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        prob_args = (betas,
                     self.design_3d,
                     self.alt_id_vector,
                     self.rows_to_obs,
                     self.rows_to_alts,
                     self.utility_transform)
        prob_kwargs = {"chosen_row_to_obs": self.chosen_row_to_obs,
                       "return_long_probs": True}
        probability_results = general_calc_probabilities(*prob_args,
                                                         **prob_kwargs)

        return probability_results

    def convenience_calc_log_likelihood(self, params):
        """
        Calculates the log-likelihood for this model and dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        args = [betas,
                self.design_3d,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.rows_to_mixers,
                self.choice_vector,
                self.utility_transform]

        kwargs = {"ridge": self.ridge}
        log_likelihood = general_log_likelihood(*args, **kwargs)

        return log_likelihood

    def convenience_calc_gradient(self, params):
        """
        Calculates the gradient of the log-likelihood for this model / dataset.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        args = [betas,
                self.design_3d,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.rows_to_mixers,
                self.choice_vector,
                self.utility_transform]

        return general_gradient(*args, ridge=self.ridge)

    def convenience_calc_hessian(self, params):
        """
        Calculates the hessian of the log-likelihood for this model / dataset.
        Note that this function name is INCORRECT with regard to the actual
        actions performed. The Mixed Logit model uses the BHHH approximation
        to the Fisher Information Matrix in place of the actual hessian.
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        args = [betas,
                self.design_3d,
                self.alt_id_vector,
                self.rows_to_obs,
                self.rows_to_alts,
                self.rows_to_mixers,
                self.choice_vector,
                self.utility_transform]

        approx_hess = general_bhhh(*args, ridge=self.ridge)

        # Account for the contrained position when presenting the results of
        # the approximate hessian.
        if self.constrained_pos is not None:
            for idx_val in self.constrained_pos:
                approx_hess[idx_val, :] = 0
                approx_hess[:, idx_val] = 0
                approx_hess[idx_val, idx_val] = -1

        return approx_hess

    def convenience_calc_fisher_approx(self, params):
        """
        Calculates the BHHH approximation of the Fisher Information Matrix for
        this model / dataset. Note that this function name is INCORRECT with
        regard to the actual actions performed. The Mixed Logit model uses a
        placeholder for the BHHH approximation of the Fisher Information Matrix
        because the BHHH approximation is already being used to approximate the
        hessian.

        This placeholder allows calculation of a value for the 'robust'
        standard errors, even though such a value is not useful since it is not
        correct...
        """
        shapes, intercepts, betas = self.convenience_split_params(params)

        placeholder_bhhh = np.diag(-1 * np.ones(betas.shape[0]))

        return placeholder_bhhh


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
                msg = "All Mixed Logit intercepts should be in the index. "
                msg_2 = "intercept_ref_pos should be None."
                raise ValueError(msg + msg_2)

        # Carry out the common instantiation process for all choice models
        model_name = model_type_to_display_name["Mixed Logit"]
        super(MixedLogit, self).__init__(data,
                                         alt_id_col,
                                         obs_id_col,
                                         choice_col,
                                         specification,
                                         names=names,
                                         model_type=model_name,
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

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
        mixl_estimator = MixedEstimator(self,
                                        mapping_res,
                                        ridge,
                                        zero_vector,
                                        split_param_vec,
                                        constrained_pos=constrained_pos)

        # Perform one final check on the length of the initial values
        mixl_estimator.check_length_of_initial_values(init_vals)

        # Get the estimation results
        estimation_res = estimate(init_vals,
                                  mixl_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res,
                                  use_hessian=False)

        # Store the mixed logit specific estimation results
        args = [mixl_estimator, estimation_res]
        estimation_res = add_mixl_specific_results_to_estimation_res(*args)

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
