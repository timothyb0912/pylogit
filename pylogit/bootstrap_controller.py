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


def bootstrap(model_obj,
              num_samples,
              init_vals,
              print_res=True,
              method="BFGS",
              loss_tol=1e-06,
              gradient_tol=1e-06,
              maxiter=1000,
              ridge=None,
              constrained_pos=None,
              boot_seed = None,
              gradient_norms=False,
              alpha=0.05,
              **kwargs):
    # Check the passed arguments for validity.

    # Create an array of the observation ids
    obs_id_array = model_obj.data[model_obj.obs_id_col].values
    # Alias the alternative IDs and the Choice Array
    alt_id_array = model_obj.alt_IDs
    choice_array = model_obj.choices

    # Perform the mle estimation and store its results.
    mle_estimation_dict =\
        model_obj.fit_mle(init_vals,
                          print_res=print_res,
                          method=method,
                          loss_tol=loss_tol,
                          gradient_tol=gradient_tol,
                          maxiter=maxiter,
                          ridge=ridge,
                          constrained_pos=constrained_pos,
                          just_point=True,
                          **kwargs)

    # Determine how many parameters are being estimated.
    num_params = mle_estimation_dict["x"].shape[0]

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
        bs.create_deepcopied_groupby_dict(model_obj.data, model_obj.obs_id_col)

    # Create a column name for the bootstrap id columns.
    boot_id_col = "bootstrap_id"

    # Initialize an array to store the bootstrapped point estimates.
    point_estimates = np.empty((num_samples, num_params), dtype=float)

    # Iterate through the various samples that need to be bootstrapped, and
    # perform the MLE
    for row in xrange(num_samples):
        # Get the bootstrapped dataframe
        bootstrap_df =\
            bs.create_bootstrap_dataframe(model_obj.data,
                                          model_obj.obs_id_col,
                                          obs_id_per_sample[row, :],
                                          dfs_by_obs_id,
                                          boot_id_col=boot_id_col)

        # Go through the necessary estimation routine to bootsrap the MLE.
        current_results = None

        # Store the bootstrapped point estimate.
        point_estimates[row] = current_results["x"]

    # If desired, calculate the gradient norm for the full sample, using each
    # bootstrapped point estimate, using the full sample.

    # If desired, calculate the log-likelihood of each bootstrapped point
    # estimate using the full sample.

    # Calculate the confidence intervals using the bootstrap samples.

    # Package the results to be returned

    return None
