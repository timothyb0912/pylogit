# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:26:33 2016

@module:    nested_logit.py
@name:      Nested Logit Model
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating nested logit models
            (with the help of the "base_multinomial_cm.py" file).
"""
from __future__ import absolute_import

import warnings
import numpy as np

from . import nested_choice_calcs as nc
from . import base_multinomial_cm_v2 as base_mcm
from .display_names import model_type_to_display_name
from .estimation import EstimationObj
from .estimation import estimate

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
def split_param_vec(all_params, rows_to_nests, return_all_types=False):
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
    return_all_types : bool, optional.
        Determines whether or not a tuple of 4 elements will be returned (with
        one element for the nest, shape, intercept, and index parameters for
        this model). If False, a tuple of 2 elements will be returned, as
        described below. The tuple will contain the nest parameters and the
        index coefficients.

    Returns
    -------
    orig_nest_coefs : 1D ndarray.
        The nest coefficients being used for estimation. Note that these values
        are the logit of the inverse of the scale parameters for each lower
        level nest.
    index_coefs : 1D ndarray.
        The coefficients of the index being used for this nested logit model.

    Note
    ----
    If `return_all_types == True` then the function will return a tuple of four
    objects. In order, these objects will either be None or the arrays
    representing the arrays corresponding to the nest, shape, intercept, and
    index parameters.
    """
    # Split the array of all coefficients
    num_nests = rows_to_nests.shape[1]
    orig_nest_coefs = all_params[:num_nests]
    index_coefs = all_params[num_nests:]

    if return_all_types:
        return orig_nest_coefs, None, None, index_coefs
    else:
        return orig_nest_coefs, index_coefs


class NestedEstimator(EstimationObj):
    """
    Estimation object for the 2-level Nested Logit Model.

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
            zero.
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
        super(NestedEstimator, self).__init__(model_obj,
                                              mapping_dict,
                                              ridge,
                                              zero_vector,
                                              split_params,
                                              constrained_pos=constrained_pos)

        return None

    def check_length_of_initial_values(self, init_values):
        """
        Ensures that the initial values are of the correct length.
        """
        # Figure out how many shape parameters we should have and how many
        # index coefficients we should have
        num_nests = self.rows_to_nests.shape[1]
        num_index_coefs = self.design.shape[1]

        assumed_param_dimensions = num_index_coefs + num_nests
        if init_values.shape[0] != assumed_param_dimensions:
            msg = "The initial values are of the wrong dimension"
            msg_1 = "It should be of dimension {}"
            msg_2 = "But instead it has dimension {}"
            raise ValueError(msg +
                             msg_1.format(assumed_param_dimensions) +
                             msg_2.format(init_values.shape[0]))

        return None

    def convenience_split_params(self, params, return_all_types=False):
        """
        Splits parameter vector into nest parameters and index parameters.

        Parameters
        ----------
        all_params : 1D ndarray.
            Should contain all of the parameters being estimated (i.e. all the
            nest coefficients and all of the index coefficients). All elements
            should be ints, floats, or longs.
        rows_to_nests : 2D scipy sparse array.
            There should be one row per observation per available alternative
            and one column per nest. This matrix maps the rows of the design
            matrix to the unique nests (on the columns).
        return_all_types : bool, optional.
            Determines whether or not a tuple of 4 elements will be returned
            (with one element for the nest, shape, intercept, and index
            parameters for this model). If False, a tuple of 2 elements will
            be returned, as described below. The tuple will contain the nest
            parameters and the index coefficients.

        Returns
        -------
        orig_nest_coefs : 1D ndarray.
            The nest coefficients being used for estimation. Note that these
            values are the logit of the inverse of the scale parameters for
            each lower level nest.
        index_coefs : 1D ndarray.
            The index coefficients of this nested logit model.

        Note
        ----
        If `return_all_types == True` then the function will return a tuple of
        four objects. In order, these objects will either be None or the arrays
        representing the arrays corresponding to the nest, shape, intercept,
        and index parameters.
        """
        return split_param_vec(params,
                               self.rows_to_nests,
                               return_all_types=return_all_types)

    def convenience_calc_probs(self, params):
        """
        Calculates the probabilities of the chosen alternative, and the long
        format probabilities for this model and dataset.
        """
        orig_nest_coefs, betas = self.convenience_split_params(params)
        natural_nest_coefs = nc.naturalize_nest_coefs(orig_nest_coefs)

        args = [natural_nest_coefs,
                betas,
                self.design,
                self.rows_to_obs,
                self.rows_to_nests]
        kwargs = {"chosen_row_to_obs": self.chosen_row_to_obs,
                  "return_type": "long_and_chosen_probs"}

        probability_results = general_calc_probabilities(*args, **kwargs)

        return probability_results

    def convenience_calc_log_likelihood(self, params):
        """
        Calculates the log-likelihood for this model and dataset.
        """
        orig_nest_coefs, betas = self.convenience_split_params(params)
        natural_nest_coefs = nc.naturalize_nest_coefs(orig_nest_coefs)

        args = [natural_nest_coefs,
                betas,
                self.design,
                self.rows_to_obs,
                self.rows_to_nests,
                self.choice_vector]
        kwargs = {"ridge": self.ridge}

        log_likelihood = general_log_likelihood(*args, **kwargs)

        return log_likelihood

    def convenience_calc_gradient(self, params):
        """
        Calculates the gradient of the log-likelihood for this model / dataset.
        """
        orig_nest_coefs, betas = self.convenience_split_params(params)

        args = [orig_nest_coefs,
                betas,
                self.design,
                self.choice_vector,
                self.rows_to_obs,
                self.rows_to_nests]

        return general_gradient(*args, ridge=self.ridge)

    def convenience_calc_hessian(self, params):
        """
        Calculates the hessian of the log-likelihood for this model / dataset.
        Note that this function name is INCORRECT with regard to the actual
        actions performed. The Nested Logit model uses the BHHH approximation
        to the Fisher Information Matrix in place of the actual hessian.
        """
        orig_nest_coefs, betas = self.convenience_split_params(params)

        args = [orig_nest_coefs,
                betas,
                self.design,
                self.choice_vector,
                self.rows_to_obs,
                self.rows_to_nests]

        approx_hess = bhhh_approx(*args, ridge=self.ridge)

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
        regard to the actual actions performed. The Nested Logit model uses a
        placeholder for the BHHH approximation of the Fisher Information Matrix
        because the BHHH approximation is already being used to approximate the
        hessian.

        This placeholder allows calculation of a value for the 'robust'
        standard errors, even though such a value is not useful since it is not
        correct...
        """
        placeholder_bhhh = np.diag(-1 * np.ones(params.shape[0]))

        return placeholder_bhhh


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

        # Determine the degenerate nests whose nesting parameters are to be
        # constrained to one. Note the following functions assume that the nest
        # parameters are placed before the index coefficients.
        fixed_params = identify_degenerate_nests(self.nest_spec)

        # Include the user specified parameters that are to be constrained to
        # their initial values
        if constrained_pos is not None:
            fixed_params.extend(constrained_pos)
        final_constrained_pos = sorted(list(set(fixed_params)))

        # Create the estimation object
        zero_vector = np.zeros(init_vals.shape)
        estimator_args = [self,
                          mapping_res,
                          ridge,
                          zero_vector,
                          split_param_vec]
        estimator_kwargs = {"constrained_pos": final_constrained_pos}
        nested_estimator = NestedEstimator(*estimator_args,
                                           **estimator_kwargs)

        # Perform one final check on the length of the initial values
        nested_estimator.check_length_of_initial_values(init_vals)

        # Get the estimation results
        estimation_res = estimate(init_vals,
                                  nested_estimator,
                                  method,
                                  loss_tol,
                                  gradient_tol,
                                  maxiter,
                                  print_res,
                                  use_hessian=False)

        # Store the estimation results
        self.store_fit_results(estimation_res)

        return None
