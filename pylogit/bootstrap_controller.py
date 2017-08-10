"""
@author:    Timothy Brathwaite
@name:      Bootstrap Controller
@summary:   This module provides functions that will control the bootstrapping
            procedure.
"""
import numpy as np
import pandas as pd

from . import bootstrap_sampler as bs

try:
    # Python 3.x does not natively support xrange
    from past.builtins import xrange
except ImportError:
    pass


class Boot(object):
    """
    Class to perform bootstrap resampling and to store and display its results.

    Parameters
    ----------
    model_obj : and instance or sublcass of the MNDC class.
    """
    def __init__(self,
                 model_obj,
                 mle_params):
        # Store the model object.
        self.model_obj = model_obj

        # Determine the parameter names
        param_names = None

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
                         mle_params,
                         print_res=True,
                         method="BFGS",
                         loss_tol=1e-06,
                         gradient_tol=1e-06,
                         maxiter=1000,
                         ridge=None,
                         constrained_pos=None,
                         boot_seed = None,
                         **kwargs):
        # Check the passed arguments for validity.

        # Create an array of the observation ids
        obs_id_array = self.model_obj.data[self.model_obj.obs_id_col].values
        # Alias the alternative IDs and the Choice Array
        alt_id_array = self.model_obj.alt_IDs
        choice_array = self.model_obj.choices

        # Determine how many parameters are being estimated.
        num_params = self.mle_params.shape[0]

        # Determine the specification of the MNL model that corresponds to
        # whatever model we're actually implementing.

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

        # Iterate through the bootstrap samples and perform the MLE
        for row in xrange(num_samples):
            # Get the bootstrapped dataframe
            bootstrap_df =\
                bs.create_bootstrap_dataframe(self.model_obj.data,
                                              self.model_obj.obs_id_col,
                                              obs_id_per_sample[row, :],
                                              dfs_by_obs_id,
                                              boot_id_col=boot_id_col)

            # Go through the necessary estimation routine to bootsrap the MLE.
            current_results = None

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
