"""
@author:    Timothy Brathwaite
@name:      Bootstrap Controller
@summary:   This module provides functions that will control the bootstrapping
            procedure.
"""
from copy import deepcopy
import itertools

import numpy as np
import pandas as pd

from . import bootstrap_sampler as bs
from . import bootstrap_calcs as bc
from . import bootstrap_abc as abc
from .bootstrap_mle import retrieve_point_est

try:
    # Python 3.x does not natively support xrange
    from past.builtins import xrange
except ImportError:
    pass


def get_param_names(model_obj):
    """
    Extracts all the names to be displayed for the estimated parameters.

    Parameters
    ----------
    model_obj : an instance of an MNDC object.
        Should have the following attributes:
        `['ind_var_names', 'intercept_names', 'shape_names', 'nest_names']`.

    Returns
    -------
    all_names : list of strings.
        There will be one element for each estimated parameter. The order of
        the parameter names will be
        `['nest_parameters', 'shape_parameters', 'outside_intercepts',
          'index_coefficients']`.
    """
    # Get the index coefficient names
    all_names = deepcopy(model_obj.ind_var_names)
    # Add the intercept names if any exist
    if model_obj.intercept_names is not None:
        all_names = model_obj.intercept_names + all_names
    # Add the shape names if any exist
    if model_obj.shape_names is not None:
        all_names = model_obj.shape_names + all_names
    # Add the nest names if any exist
    if model_obj.nest_names is not None:
        all_names = model_obj.nest_names + all_names
    return all_names


class Boot(object):
    """
    Class to perform bootstrap resampling and to store and display its results.

    Parameters
    ----------
    model_obj : an instance or sublcass of the MNDC class.
    mle_params : 1D ndarray.
        Should contain the desired model's maximum likelihood point estimate.
    """
    def __init__(self,
                 model_obj,
                 mle_params):
        # Store the model object.
        self.model_obj = model_obj

        # Determine the parameter names
        param_names = get_param_names(model_obj)

        # Store the MLE parameters
        self.mle_params = pd.Series(mle_params, index=param_names)

        # Initialize the attributes that will be used later on.
        desired_attributes =\
            ["bootstrap_replicates", "jackknife_replicates",
             "percentile_interval", "bca_interval",
             "abc_interval", "conf_intervals",
             "conf_alpha", "summary"]
        for attr_name in desired_attributes:
            setattr(self, attr_name, None)

        return None

    def generate_bootstrap_replicates(self,
                                      num_samples,
                                      mnl_obj=None,
                                      mnl_init_vals=None,
                                      mnl_fit_kwargs=None,
                                      extract_init_vals=None,
                                      print_res=False,
                                      method="BFGS",
                                      loss_tol=1e-06,
                                      gradient_tol=1e-06,
                                      maxiter=1000,
                                      ridge=None,
                                      constrained_pos=None,
                                      boot_seed=None):
        """
        Parameters
        ----------
        num_samples : positive int.
            Specifies the number of bootstrap samples that are to be drawn.
        mnl_spec : OrderedDict or None, optional.
            If the model that is being estimated is not an MNL, then `mnl_spec`
            should be passed. This should be the specification used to estimate
            the MNL model that our desired model is based on.
            Default == None.
        mnl_names : OrderedDict or None, optional.
            If the model that is being estimated is not an MNL, then `mnl_spec`
            should be passed. This should be the name dictionary used to
            estimate the MNL model that our desired model is based on.
            Default == None.
        mnl_init_vals : 1D ndarray or None, optional.
            If the model that is being estimated is not an MNL, then
            `mnl_init_val` should be passed. Should contain the values used to
            begin the estimation process for the MNL model that is used to
            provide starting values for our desired model. Default == None.
        mnl_fit_kwargs : dict or None.
            If the model that is being estimated is not an MNL, then
            `mnl_fit_kwargs` should be passed.
        extract_init_vals : callable or None, optional.
            Should accept 3 arguments, in the following order. First, it should
            accept `orig_model_obj`. Second, it should accept a pandas Series
            of estimated parameters from the MNL model. The Series' index will
            be the names of the coefficients from `mnl_names`. Thirdly, it
            should accept an int denoting the number of parameters in the final
            choice model. The callable should return a 1D ndarray of starting
            values for the final choice model. Default == None.
        print_res : bool, optional.
            Determines whether the timing and initial and final log likelihood
            results will be printed as they they are determined.
            Default `== True`.
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
        ridge : int, float, long, or None, optional.
            Determines whether or not ridge regression is performed. If a
            scalar is passed, then that scalar determines the ridge penalty for
            the optimization. The scalar should be greater than or equal to
            zero. Default `== None`.
        constrained_pos : list or None, optional.
            Denotes the positions of the array of estimated parameters that are
            not to change from their initial values. If a list is passed, the
            elements are to be integers where no such integer is greater than
            `init_vals.size.` Default == None.
        boot_seed = non-negative int or None, optional.
            Denotes the random seed to be used when generating the bootstrap
            samples. If None, the sample generation process will generally be
            non-reproducible. Default == None.
        """
        # Check the passed arguments for validity.

        # Create an array of the observation ids
        obs_id_array = self.model_obj.data[self.model_obj.obs_id_col].values
        # Alias the alternative IDs and the Choice Array
        alt_id_array = self.model_obj.alt_IDs
        choice_array = self.model_obj.choices

        # Determine how many parameters are being estimated.
        num_params = self.mle_params.shape[0]

        # Figure out which observations are in each bootstrap sample.
        obs_id_per_sample =\
            bs.create_cross_sectional_bootstrap_samples(obs_id_array,
                                                        alt_id_array,
                                                        choice_array,
                                                        num_samples,
                                                        seed=boot_seed)

        # Get the dictionary of sub-dataframes for each observation id
        dfs_by_obs_id =\
            bs.create_deepcopied_groupby_dict(self.model_obj.data,
                                              self.model_obj.obs_id_col)

        # Create a column name for the bootstrap id columns.
        boot_id_col = "bootstrap_id"

        # Initialize an array to store the bootstrapped point estimates.
        point_estimates = np.empty((num_samples, num_params), dtype=float)

        # Get keyword arguments for final model estimation with new data.
        fit_kwargs = {"print_res": print_res,
                      "method": method,
                      "loss_tol": loss_tol,
                      "gradient_tol": gradient_tol,
                      "maxiter": maxiter,
                      "ridge": ridge,
                      "constrained_pos": constrained_pos,
                      "just_point": True}

        # Get the specification and name dictionary of the MNL model.
        mnl_spec = None if mnl_obj is None else mnl_obj.specification
        mnl_names = None if mnl_obj is None else mnl_obj.name_spec

        # Iterate through the bootstrap samples and perform the MLE
        for row in xrange(num_samples):
            # Get the bootstrapped dataframe
            bootstrap_df =\
                bs.create_bootstrap_dataframe(self.model_obj.data,
                                              self.model_obj.obs_id_col,
                                              obs_id_per_sample[row, :],
                                              dfs_by_obs_id,
                                              boot_id_col=boot_id_col)

            # Go through the necessary estimation routine to bootstrap the MLE.
            current_results =\
                retrieve_point_est(self.model_obj,
                                   bootstrap_df,
                                   boot_id_col,
                                   num_params,
                                   mnl_spec,
                                   mnl_names,
                                   mnl_init_vals,
                                   mnl_fit_kwargs,
                                   extract_init_vals=extract_init_vals,
                                   **fit_kwargs)

            # Store the bootstrapped point estimate.
            point_estimates[row] = current_results["x"]

        # Store the point estimates as a pandas dataframe
        self.bootstrap_replicates =\
            pd.DataFrame(point_estimates, columns=self.mle_params.index)

        return None

    def generate_jackknife_replicates(self,
                                      mnl_obj=None,
                                      mnl_init_vals=None,
                                      mnl_fit_kwargs=None,
                                      extract_init_vals=None,
                                      print_res=False,
                                      method="BFGS",
                                      loss_tol=1e-06,
                                      gradient_tol=1e-06,
                                      maxiter=1000,
                                      ridge=None,
                                      constrained_pos=None):
        # Take note of the observation id column that is to be used
        obs_id_col = self.model_obj.obs_id_col

        # Get the array of original observation ids
        orig_obs_id_array =\
            self.model_obj.data[obs_id_col].values

        # Get an array of the unique observation ids.
        unique_obs_ids = np.sort(np.unique(orig_obs_id_array))

        # Determine how many observations are in one's dataset.
        num_obs = unique_obs_ids.size
        # Determine how many parameters are being estimated.
        num_params = self.mle_params.size

        # Get keyword arguments for final model estimation with new data.
        fit_kwargs = {"print_res": print_res,
                      "method": method,
                      "loss_tol": loss_tol,
                      "gradient_tol": gradient_tol,
                      "maxiter": maxiter,
                      "ridge": ridge,
                      "constrained_pos": constrained_pos,
                      "just_point": True}

        # Get the specification and name dictionary of the MNL model.
        mnl_spec = None if mnl_obj is None else mnl_obj.specification
        mnl_names = None if mnl_obj is None else mnl_obj.name_spec

        # Initialize the array of jackknife replicates
        point_replicates = np.empty((num_obs, num_params), dtype=float)

        # Populate the array of jackknife replicates
        for pos, obs_id in enumerate(unique_obs_ids):
            # Create the dataframe without the current observation
            new_df = self.model_obj.data.loc[orig_obs_id_array != obs_id]
            # Get the point estimate for this new dataset
            current_results =\
                retrieve_point_est(self.model_obj,
                                   new_df,
                                   obs_id_col,
                                   num_params,
                                   mnl_spec,
                                   mnl_names,
                                   mnl_init_vals,
                                   mnl_fit_kwargs,
                                   extract_init_vals=extract_init_vals,
                                   **fit_kwargs)
            # Store the estimated parameters
            point_replicates[pos] = current_results['x']

        # Store the jackknife replicates as a pandas dataframe
        self.jackknife_replicates =\
            pd.DataFrame(point_replicates, columns=self.mle_params.index)
        return None

    def calc_log_likes_for_replicates(self, replicate='bootstrap'):
        raise NotImplementedError
        return None

    def calc_gradient_norm_for_replicates(self, replicate='bootstrap'):
        raise NotImplementedError
        return None

    def calc_percentile_interval(self, conf_percentage):
        # Get the alpha % that corresponds to the given confidence percentage.
        alpha = bc.get_alpha_from_conf_percentage(conf_percentage)
        # Create the column names for the dataframe of confidence intervals
        single_column_names =\
            ['{:.3g}%'.format(alpha / 2.0),
             '{:.3g}%'.format(100 - alpha / 2.0)]
        # Calculate the desired confidence intervals.
        conf_intervals =\
            bc.calc_percentile_interval(self.bootstrap_replicates.values,
                                        conf_percentage)
        # Store the desired confidence intervals
        self.percentile_interval =\
            pd.DataFrame(conf_intervals.T,
                         index=self.mle_params.index,
                         columns=single_column_names)
        return None

    def calc_bca_interval(self, conf_percentage):
        # Get the alpha % that corresponds to the given confidence percentage.
        alpha = bc.get_alpha_from_conf_percentage(conf_percentage)
        # Create the column names for the dataframe of confidence intervals
        single_column_names =\
            ['{:.3g}%'.format(alpha / 2.0),
             '{:.3g}%'.format(100 - alpha / 2.0)]
        # Bundle the arguments needed to create the desired confidence interval
        args = [self.bootstrap_replicates.values,
                self.jackknife_replicates.values,
                self.mle_params.values,
                conf_percentage]
        # Calculate the BCa confidence intervals.
        conf_intervals = bc.calc_bca_interval(*args)
        # Store the BCa confidence intervals.
        self.bca_interval = pd.DataFrame(conf_intervals.T,
                                         index=self.mle_params.index,
                                         columns=single_column_names)
        return None

    def calc_abc_interval(self,
                          conf_percentage,
                          init_vals,
                          epsilon=abc.EPSILON,
                          **fit_kwargs):
        # Get the alpha % that corresponds to the given confidence percentage.
        alpha = bc.get_alpha_from_conf_percentage(conf_percentage)
        # Create the column names for the dataframe of confidence intervals
        single_column_names =\
            ['{:.3g}%'.format(alpha / 2.0),
             '{:.3g}%'.format(100 - alpha / 2.0)]
        # Calculate the ABC confidence intervals
        conf_intervals =\
            abc.calc_abc_interval(self.model_obj,
                                  self.mle_params.values,
                                  init_vals,
                                  conf_percentage,
                                  epsilon=epsilon,
                                  **fit_kwargs)
        # Store the ABC confidence intervals
        self.abc_interval = pd.DataFrame(conf_intervals.T,
                                         index=self.mle_params.index,
                                         columns=single_column_names)
        return None

    def calc_conf_intervals(self,
                            conf_percentage,
                            interval_type='all',
                            init_vals=None,
                            epsilon=abc.EPSILON,
                            **fit_kwargs):
        if interval_type == 'pi':
            self.calc_percentile_interval(conf_percentage)
        elif interval_type == 'bca':
            self.calc_bca_interval(conf_percentage)
        elif interval_type == 'abc':
            self.calc_abc_interval(conf_percentage,
                                   init_vals,
                                   epsilon=epsilon,
                                   **fit_kwargs)
        elif interval_type == 'all':
            print("Calculating Percentile Confidence Intervals")
            self.calc_percentile_interval(conf_percentage)
            print("Calculating BCa Confidence Intervals")
            self.calc_bca_interval(conf_percentage)
            print("Calculating ABC Confidence Intervals")
            self.calc_abc_interval(conf_percentage,
                                   init_vals,
                                   epsilon=epsilon,
                                   **fit_kwargs)
            # Get the alpha % for the given confidence percentage.
            alpha = bc.get_alpha_from_conf_percentage(conf_percentage)
            # Get lists of the interval type names and the endpoint names
            interval_type_names = ['percentile_interval',
                                   'BCa_interval',
                                   'ABC_interval']
            endpoint_names = ['{:.3g}%'.format(alpha / 2.0),
                              '{:.3g}%'.format(100 - alpha / 2.0)]
            # Create the column names for the dataframe of confidence intervals
            multi_index_names =\
                list(itertools.product(interval_type_names, endpoint_names))
            df_column_index = pd.MultiIndex.from_tuples(multi_index_names)
            # Create the dataframe containing all confidence intervals
            self.all_intervals = pd.concat([self.percentile_interval,
                                            self.bca_interval,
                                            self.abc_interval],
                                            axis=1,
                                            ignore_index=True)
            # Store the column names for the combined confidence intervals
            self.all_intervals.columns = df_column_index
        else:
            msg = "interval_type MUST be in `['pi', 'bca', 'abc', 'all']`"
            raise ValueError(msg)
        return None
