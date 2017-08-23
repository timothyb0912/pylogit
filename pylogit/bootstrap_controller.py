"""
@author:    Timothy Brathwaite
@name:      Bootstrap Controller
@summary:   This module provides functions that will control the bootstrapping
            procedure.
"""
from copy import deepcopy

import numpy as np
import pandas as pd

from . import bootstrap_sampler as bs
from .bootstrap_mle import retrieve_point_est
from .display_names import model_type_to_display_name

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
        desired_attributes = ["point_samples",
                              "conf_intervals",
                              "conf_alpha",
                              "summary"]
        for attr_name in desired_attributes:
            setattr(self, attr_name, None)

        return None

    def bootstrap_params(num_samples,
                         mnl_obj=None,
                         mnl_init_vals=None,
                         mnl_fit_kwargs=None,
                         extract_init_vals=None
                         print_res=False,
                         method="BFGS",
                         loss_tol=1e-06,
                         gradient_tol=1e-06,
                         maxiter=1000,
                         ridge=None,
                         constrained_pos=None,
                         boot_seed = None,
                         **kwargs):
        """
        Parameters
        ----------
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
        # Get the 'fake' bootstrap observation ids.
        bootstrap_obs_ids = bs.create_bootstrap_id_array(obs_id_per_sample)
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
        self.point_samples = pd.DataFrame(point_estimates,
                                          columns=self.mle_params.index)

        return None

    def calc_log_likes_for_samples():
        raise NotImplementedError
        return None

    def calc_gradient_norm_for_samples():
        raise NotImplementedError
        return None

    def calc_conf_intervals():
        raise NotImplementedError
        return None
