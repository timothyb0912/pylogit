"""
@author:    Timothy Brathwaite
@name:      Bootstrap Sampler
@summary:   This module provides functions that will perform the stratified
            resampling needed for the bootstrapping procedure.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

try:
    # Python 3.x does not natively support xrange
    from past.builtins import xrange
except ImportError:
    pass


def relate_obs_ids_to_chosen_alts(obs_id_array,
                                  alt_id_array,
                                  choice_array):
    """
    Creates a dictionary that relates each unique alternative id to the set of
    observations ids that chose the given alternative.

    Parameters
    ----------
    obs_id_array : 1D ndarray of ints.
        Should be a long-format array of observation ids. Each element should
        correspond to the unique id of the unit of observation that corresponds
        to the given row of the long-format data. Note that each unit of
        observation may have more than one associated choice situation.
    alt_id_array : 1D ndarray of ints.
        Should be a long-format array of alternative ids. Each element should
        denote the unique id of the alternative that corresponds to the given
        row of the long format data.
    choice_array : 1D ndarray of ints.
        Each element should be either a one or a zero, indicating whether the
        alternative on the given row of the long format data was chosen or not.

    Returns
    -------
    chosen_alts_to_obs_ids : dict.
        Each key will be a unique value from `alt_id_array`. Each key's value
        will be a 1D ndarray that contains the sorted, unique observation ids
        of those observational units that chose the given alternative.
    """
    # Figure out which units of observation chose each alternative.
    chosen_alts_to_obs_ids = {}

    for alt_id in np.sort(np.unique(alt_id_array)):
        # Determine which observations chose the current alternative.
        selection_condition =\
            np.where((alt_id_array == alt_id) & (choice_array == 1))

        # Store the sorted, unique ids that chose the current alternative.
        chosen_alts_to_obs_ids[alt_id] =\
            np.sort(np.unique(obs_id_array[selection_condition]))

    # Return the desired dictionary.
    return chosen_alts_to_obs_ids


def get_num_obs_choosing_each_alternative(obs_per_alt_dict):
    """
    Will create an ordered dictionary that records the number of units of
    observation that have chosen the given alternative (i.e. the associated
    dictionary key).

    Parameters
    ----------
    obs_per_alt_dict : dict.
        Each key should be a unique alternave id. Each key's value will be 1D
        ndarray that contains the sorted, unique observation ids of those
        observational units that chose the given alternative.
    tot_num_obs : int.
        Should the denote the total number of "observation id-choices". If a
        cross-sectional dataset is being used, then this value will simply be
        the number of unique observational units. If a panel dataset is being
        used, this value will be the sum over all observational units of the
        number of observed choice situations for each unit.
    """
    # Initialize the object that is to be returned.
    num_obs_per_group = OrderedDict()

    # Determine the number of unique units of observation per group.
    for alt_id in obs_per_alt_dict:
        num_obs_per_group[alt_id] = len(obs_per_alt_dict[alt_id])

    # Determine the total number of units of observation that will be chosen
    # for each bootstrap sample.
    tot_num_obs = sum([num_obs_per_group[g] for g in num_obs_per_group])

    # Return the desired objects.
    return num_obs_per_group, tot_num_obs


def create_cross_sectional_bootstrap_samples(obs_id_array,
                                             alt_id_array,
                                             choice_array,
                                             num_samples,
                                             seed=None):
    """
    Determines the unique observations that will be present in each bootstrap
    sample. This function DOES NOT create the new design matrices or a new
    long-format dataframe for each bootstrap sample. Note that these will be
    correct bootstrap samples for cross-sectional datasets. This function will
    not work correctly for panel datasets.
    """
    # Determine the units of observation that chose each alternative.
    chosen_alts_to_obs_ids =\
        relate_obs_ids_to_chosen_alts(obs_id_array, alt_id_array, choice_array)

    # Determine the number of unique units of observation per group and overall
    num_obs_per_group, tot_num_obs =\
        get_num_obs_choosing_each_alternative(chosen_alts_to_obs_ids)

    # Initialize the array that will store the observation ids for each sample
    ids_per_sample = np.empty((num_samples, tot_num_obs), dtype=float)

    if seed is not None:
        # Check the validity of the seed argument.
        if not isinstance(seed, (int, long)):
            msg = "`boot_seed` MUST be an int or long."
            raise ValueError(msg)

        # If desiring reproducibility, set the random seed within numpy
        np.random.seed(seed)

    # Initialize a variable to keep track of what column we're on.
    col_idx = 0
    for alt_id in num_obs_per_group:
        # Get the set of observations that chose the current alternative.
        relevant_ids = chosen_alts_to_obs_ids[alt_id]
        # Determine the number of needed resampled ids.
        resample_size = num_obs_per_group[alt_id]
        # Resample, with replacement, observations who chose this alternative.
        current_ids = (np.random.choice(relevant_ids,
                                        size=resample_size * num_samples,
                                        replace=True)
                                .reshape((num_samples, resample_size)))
        # Determine the last column index to use when storing the resampled ids
        end_col = col_idx + resample_size
        # Assign the sampled ids to the correct columns of ids_per_sample
        ids_per_sample[:, col_idx:end_col] = current_ids
        # Update the column index
        col_idx += resample_size

    # Return the resampled observation ids.
    return ids_per_sample


def create_bootstrap_id_array(obs_id_per_sample):
    """
    Creates a 2D ndarray that contains the 'bootstrap ids' for each replication
    of each unit of observation that is an the set of bootstrap samples.

    Parameters
    ----------
    obs_id_per_sample : 2D ndarray of ints.
        Should have one row for each bootsrap sample. Should have one column
        for each observational unit that is serving as a new bootstrap
        observational unit.

    Returns
    -------
    bootstrap_id_array : 2D ndarray of ints.
        Will have the same shape as `obs_id_per_sample`. Each element will
        denote the fake observational id in the new bootstrap dataset.
    """
    # Determine the shape of the object to be returned.
    num_rows, num_cols = obs_id_per_sample.shape
    # Create the array of bootstrap ids.
    bootstrap_id_array =\
        np.tile(np.arange(num_cols) +1, num_rows).reshape((num_rows, num_cols))
    # Return the desired object
    return bootstrap_id_array


def create_deepcopied_groupby_dict(orig_df, obs_id_col):
    """
    Will create a dictionary where each key corresponds to a unique value in
    `orig_df[obs_id_col]` and each value corresponds to all of the rows of
    `orig_df` where `orig_df[obs_id_col] == key`.

    Parameters
    ----------
    orig_df : pandas DataFrame.
        Should be long-format dataframe containing the data used to estimate
        the desired choice model.
    obs_id_col : str.
        Should be a column name within `orig_df`. Should denote the original
        observation id column.

    Returns
    -------
    groupby_dict : dict.
        Each key will be a unique value in `orig_df[obs_id_col]` and each value
        will be the rows of `orig_df` where `orig_df[obs_id_col] == key`.
    """
    # Get the observation id values
    obs_id_vals = orig_df[obs_id_col].values
    # Get the unique observation ids
    unique_obs_ids = np.unique(obs_id_vals)
    # Initialize the dictionary to be returned.
    groupby_dict = {}
    # Populate the dictionary with dataframes for each individual.
    for obs_id in unique_obs_ids:
        # Filter out only the rows corresponding to the current observation id.
        desired_rows = obs_id_vals == obs_id
        # Add the desired dataframe to the dictionary.
        groupby_dict[obs_id] = orig_df.loc[desired_rows].copy(deep=True)

    # Return the desired object.
    return groupby_dict


def create_bootstrap_dataframe(orig_df,
                               obs_id_col,
                               resampled_obs_ids_1d,
                               groupby_dict,
                               boot_id_col="bootstrap_id"):
    """
    Will create the altered dataframe of data needed to estimate a choice model
    with the particular observations that belong to the current bootstrap
    sample.

    Parameters
    ----------
    orig_df : pandas DataFrame.
        Should be long-format dataframe containing the data used to estimate
        the desired choice model.
    obs_id_col : str.
        Should be a column name within `orig_df`. Should denote the original
        observation id column.
    resampled_obs_ids_1d : 1D ndarray of ints.
        Each value should represent the alternative id of a given bootstrap
        replicate.
    groupby_dict : dict.
        Each key will be a unique value in `orig_df[obs_id_col]` and each value
        will be the rows of `orig_df` where `orig_df[obs_id_col] == key`.
    boot_id_col : str, optional.
        Denotes the new column that will be created to specify the bootstrap
        observation ids for choice model estimation.

    Returns
    -------
    bootstrap_df : pandas Dataframe.
        Will contain all the same columns as `orig_df` as well as the
        additional `boot_id_col`. For each value in `resampled_obs_ids_1d`,
        `bootstrap_df` will contain the long format rows from `orig_df` that
        have the given observation id.
    """
    # Check the validity of the passed arguments.
    if obs_id_col not in orig_df.columns:
        msg = "`obs_id_col` MUST be a column in `orig_df`"
        raise ValueError(msg)
    if not np.in1D(resampled_obs_ids_1d, orig_df[obs_id_col].values).all():
        msg =\
            ("All values in `resampled_obs_ids_1d` MUST be in
            "`orig_df[obs_id_col]`.")
        raise ValueError(msg)
    if boot_id_col in orig_df.columns:
        msg = "Ensure that `boot_id_col` is not in orig_df.columns."
        raise ValueError(msg)

    # Alias the observation id column
    obs_id_values = orig_df[obs_id_col].values

    # Initialize a list to store the component dataframes that will be
    # concatenated to form the final bootstrap_df
    component_dfs = []

    # Populate component_dfs
    for obs_id, boot_id in enumerate(resampled_obs_ids_1d):
        # Get the rows of orig_df that correspond to the desired observation id
        desired_rows = obs_id_values == obs_id
        # Extract the dataframe that we desire.
        extracted_df = groupby_dict[obs_id]
        # Add the bootstrap id value.
        extracted_df[boot_id_col] = boot_id
        # Store the component dataframe
        component_dfs.append(extracted_df)

    # Create and return the desired dataframe.
    bootstrap_df = pd.concat(component_dfs, axis=0, ignore_index=True)
    return bootstrap_df
