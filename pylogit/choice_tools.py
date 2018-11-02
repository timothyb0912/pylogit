# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 08:32:48 2016

@module:    choice_tools.py
@name:      Helpful Tools for Choice Model Estimation
@author:    Timothy Brathwaite
@summary:   Contains functions that help prepare one's data for choice model
            estimation or helps speed the estimation process (the 'mappings').
"""
from __future__ import absolute_import

import warnings
from collections import OrderedDict
from collections import Iterable
from numbers import Number

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse import issparse


def get_dataframe_from_data(data):
    """
    Parameters
    ----------
    data : string or pandas dataframe.
        If string, data should be an absolute or relative path to a CSV file
        containing the long format data for this choice model. Note long format
        has one row per available alternative for each observation. If pandas
        dataframe, the dataframe should be the long format data for the choice
        model.

    Returns
    -------
    dataframe : pandas dataframe of the long format data for the choice model.
    """
    if isinstance(data, str):
        if data.endswith(".csv"):
            dataframe = pd.read_csv(data)
        else:
            msg_1 = "data = {} is of unknown file type."
            msg_2 = " Please pass path to csv."
            raise ValueError(msg_1.format(data) + msg_2)
    elif isinstance(data, pd.DataFrame):
        dataframe = data
    else:
        msg_1 = "type(data) = {} is an invalid type."
        msg_2 = " Please pass pandas dataframe or path to csv."
        raise TypeError(msg_1.format(type(data)) + msg_2)

    return dataframe


# In a multinomial choice model, all data should be converted to "long format"
# Long format has one row per observation per available alternative for each
# person.

# Each row should have the same number of columns, so all utility coefficients
# must be part of each utility specification.

# This means that the variables which belong to coefficients that are not
# "naturally" in all utility equations must be redefined so that the
# variable == 0 when the utility equation does not correspond to the beta that
# is associated with the variable.

# Additionally, variables which show up in all utility equations but have
# different betas in all utility equations should be turned into separate
# variables--one per utility equation-- where the variable == 0 if the
# alternative does not correspond to the beta for that variable.


def ensure_object_is_ordered_dict(item, title):
    """
    Checks that the item is an OrderedDict. If not, raises ValueError.
    """
    assert isinstance(title, str)

    if not isinstance(item, OrderedDict):
        msg = "{} must be an OrderedDict. {} passed instead."
        raise TypeError(msg.format(title, type(item)))

    return None


def ensure_object_is_string(item, title):
    """
    Checks that the item is a string. If not, raises ValueError.
    """
    assert isinstance(title, str)

    if not isinstance(item, str):
        msg = "{} must be a string. {} passed instead."
        raise TypeError(msg.format(title, type(item)))

    return None


def ensure_object_is_ndarray(item, title):
    """
    Ensures that a given mapping matrix is a dense numpy array. Raises a
    helpful TypeError if otherwise.
    """
    assert isinstance(title, str)

    if not isinstance(item, np.ndarray):
        msg = "{} must be a np.ndarray. {} passed instead."
        raise TypeError(msg.format(title, type(item)))

    return None


def ensure_columns_are_in_dataframe(columns,
                                    dataframe,
                                    col_title='',
                                    data_title='data'):
    """
    Checks whether each column in `columns` is in `dataframe`. Raises
    ValueError if any of the columns are not in the dataframe.

    Parameters
    ----------
    columns : list of strings.
        Each string should represent a column heading in dataframe.
    dataframe : pandas DataFrame.
        Dataframe containing the data for the choice model to be estimated.
    col_title : str, optional.
        Denotes the title of the columns that were passed to the function.
    data_title : str, optional.
        Denotes the title of the dataframe that is being checked to see whether
        it contains the passed columns. Default == 'data'

    Returns
    -------
    None.
    """
    # Make sure columns is an iterable
    assert isinstance(columns, Iterable)
    # Make sure dataframe is a pandas dataframe
    assert isinstance(dataframe, pd.DataFrame)
    # Make sure title is a string
    assert isinstance(col_title, str)
    assert isinstance(data_title, str)

    problem_cols = [col for col in columns if col not in dataframe.columns]
    if problem_cols != []:
        if col_title == '':
            msg = "{} not in {}.columns"
            final_msg = msg.format(problem_cols, data_title)
        else:
            msg = "The following columns in {} are not in {}.columns: {}"
            final_msg = msg.format(col_title, data_title, problem_cols)

        raise ValueError(final_msg)

    return None


def check_argument_type(long_form, specification_dict):
    """
    Ensures that long_form is a pandas dataframe and that specification_dict
    is an OrderedDict, raising a ValueError otherwise.

    Parameters
    ----------
    long_form : pandas dataframe.
        Contains one row for each available alternative, for each observation.
    specification_dict : OrderedDict.
        Keys are a proper subset of the columns in `long_form_df`. Values are
        either a list or a single string, `"all_diff"` or `"all_same"`. If a
        list, the elements should be:

            - single objects that are within the alternative ID column of
              `long_form_df`
            - lists of objects that are within the alternative ID column of
              `long_form_df`. For each single object in the list, a unique
              column will be created (i.e. there will be a unique coefficient
              for that variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification_dict` values, a single column will be created for
              all the alternatives within iterable (i.e. there will be one
              common coefficient for the variables in the iterable).

    Returns
    -------
    None.
    """
    if not isinstance(long_form, pd.DataFrame):
        msg = "long_form should be a pandas dataframe. It is a {}"
        raise TypeError(msg.format(type(long_form)))

    ensure_object_is_ordered_dict(specification_dict, "specification_dict")

    return None


def ensure_alt_id_in_long_form(alt_id_col, long_form):
    """
    Ensures alt_id_col is in long_form, and raises a ValueError if not.

    Parameters
    ----------
    alt_id_col : str.
        Column name which denotes the column in `long_form` that contains the
        alternative ID for each row in `long_form`.
    long_form : pandas dataframe.
        Contains one row for each available alternative, for each observation.

    Returns
    -------
    None.
    """
    if alt_id_col not in long_form.columns:
        msg = "alt_id_col == {} is not a column in long_form."
        raise ValueError(msg.format(alt_id_col))

    return None


def ensure_specification_cols_are_in_dataframe(specification, dataframe):
    """
    Checks whether each column in `specification` is in `dataframe`. Raises
    ValueError if any of the columns are not in the dataframe.

    Parameters
    ----------
    specification : OrderedDict.
        Keys are a proper subset of the columns in `data`. Values are either a
        list or a single string, "all_diff" or "all_same". If a list, the
        elements should be:
            - single objects that are in the alternative ID column of `data`
            - lists of objects that are within the alternative ID column of
              `data`. For each single object in the list, a unique column will
              be created (i.e. there will be a unique coefficient for that
              variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification` values, a single column will be created for all
              the alternatives within the iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    dataframe : pandas DataFrame.
        Dataframe containing the data for the choice model to be estimated.

    Returns
    -------
    None.
    """
    # Make sure specification is an OrderedDict
    try:
        assert isinstance(specification, OrderedDict)
    except AssertionError:
        raise TypeError("`specification` must be an OrderedDict.")
    # Make sure dataframe is a pandas dataframe
    assert isinstance(dataframe, pd.DataFrame)

    problem_cols = []
    dataframe_cols = dataframe.columns
    for key in specification:
        if key not in dataframe_cols:
            problem_cols.append(key)
    if problem_cols != []:
        msg = "The following keys in the specification are not in 'data':\n{}"
        raise ValueError(msg.format(problem_cols))

    return None


def check_type_and_values_of_specification_dict(specification_dict,
                                                unique_alternatives):
    """
    Verifies that the values of specification_dict have the correct type, have
    the correct structure, and have valid values (i.e. are actually in the set
    of possible alternatives). Will raise various errors if / when appropriate.

    Parameters
    ----------
    specification_dict : OrderedDict.
        Keys are a proper subset of the columns in `long_form_df`. Values are
        either a list or a single string, `"all_diff"` or `"all_same"`. If a
        list, the elements should be:

            - single objects that are within the alternative ID column of
              `long_form_df`
            - lists of objects that are within the alternative ID column of
              `long_form_df`. For each single object in the list, a unique
              column will be created (i.e. there will be a unique coefficient
              for that variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification_dict` values, a single column will be created for
              all the alternatives within iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    unique_alternatives : 1D ndarray.
        Should contain the possible alternative id's for this dataset.

    Returns
    -------
    None.
    """
    for key in specification_dict:
        specification = specification_dict[key]
        if isinstance(specification, str):
            if specification not in ["all_same", "all_diff"]:
                msg = "specification_dict[{}] not in ['all_same', 'all_diff']"
                raise ValueError(msg.format(key))

        elif isinstance(specification, list):
            # Imagine that the specification is [[1, 2], 3]
            # group would be [1, 2]
            # group_item would be 1 or 2. group_item should never be a list.
            for group in specification:
                group_is_list = isinstance(group, list)
                if group_is_list:
                    for group_item in group:
                        if isinstance(group_item, list):
                            msg = "Wrong structure for specification_dict[{}]"
                            msg_2 = " Values can be a list of lists of ints,"
                            msg_3 = " not lists of lists of lists of ints."
                            total_msg = msg.format(key) + msg_2 + msg_3
                            raise ValueError(total_msg)

                        elif group_item not in unique_alternatives:
                            msg_1 = "{} in {} in specification_dict[{}]"
                            msg_2 = " is not in long_format[alt_id_col]"
                            total_msg = (msg_1.format(group_item, group, key) +
                                         msg_2)
                            raise ValueError(total_msg)
                else:
                    if group not in unique_alternatives:
                        msg_1 = "{} in specification_dict[{}]"
                        msg_2 = " is not in long_format[alt_id_col]"
                        raise ValueError(msg_1.format(group, key) + msg_2)

        else:
            msg = "specification_dict[{}] must be 'all_same', 'all_diff', or"
            msg_2 = " a list."
            raise TypeError(msg.format(key) + msg_2)

    return None


def check_keys_and_values_of_name_dictionary(names,
                                             specification_dict,
                                             num_alts):
    """
    Check the validity of the keys and values in the names dictionary.

    Parameters
    ----------
    names : OrderedDict, optional.
        Should have the same keys as `specification_dict`. For each key:

            - if the corresponding value in `specification_dict` is "all_same",
              then there should be a single string as the value in names.
            - if the corresponding value in `specification_dict` is "all_diff",
              then there should be a list of strings as the value in names.
              There should be one string in the value in names for each
              possible alternative.
            - if the corresponding value in `specification_dict` is a list,
              then there should be a list of strings as the value in names.
              There should be one string the value in names per item in the
              value in `specification_dict`.
    specification_dict : OrderedDict.
        Keys are a proper subset of the columns in `long_form_df`. Values are
        either a list or a single string, `"all_diff"` or `"all_same"`. If a
        list, the elements should be:

            - single objects that are within the alternative ID column of
              `long_form_df`
            - lists of objects that are within the alternative ID column of
              `long_form_df`. For each single object in the list, a unique
              column will be created (i.e. there will be a unique coefficient
              for that variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification_dict` values, a single column will be created for
              all the alternatives within iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    num_alts : int.
        The number of alternatives in this dataset's universal choice set.

    Returns
    -------
    None.
    """
    if names.keys() != specification_dict.keys():
        msg = "names.keys() does not equal specification_dict.keys()"
        raise ValueError(msg)

    for key in names:
        specification = specification_dict[key]
        name_object = names[key]
        if isinstance(specification, list):
            try:
                assert isinstance(name_object, list)
                assert len(name_object) == len(specification)
                assert all([isinstance(x, str) for x in name_object])
            except AssertionError:
                msg = "names[{}] must be a list AND it must have the same"
                msg_2 = " number of strings as there are elements of the"
                msg_3 = " corresponding list in specification_dict"
                raise ValueError(msg.format(key) + msg_2 + msg_3)

        else:
            if specification == "all_same":
                if not isinstance(name_object, str):
                    msg = "names[{}] should be a string".format(key)
                    raise TypeError(msg)

            else:    # This means speciffication == 'all_diff'
                try:
                    assert isinstance(name_object, list)
                    assert len(name_object) == num_alts
                except AssertionError:
                    msg_1 = "names[{}] should be a list with {} elements,"
                    msg_2 = " 1 element for each possible alternative"
                    msg = (msg_1.format(key, num_alts) + msg_2)
                    raise ValueError(msg)

    return None


def ensure_all_columns_are_used(num_vars_accounted_for,
                                dataframe,
                                data_title='long_data'):
    """
    Ensure that all of the columns from dataframe are in the list of used_cols.
    Will raise a helpful UserWarning if otherwise.

    Parameters
    ----------
    num_vars_accounted_for : int.
        Denotes the number of variables used in one's function.
    dataframe : pandas dataframe.
        Contains all of the data to be converted from one format to another.
    data_title : str, optional.
        Denotes the title by which `dataframe` should be referred in the
        UserWarning.

    Returns
    -------
    None.
    """
    dataframe_vars = set(dataframe.columns.tolist())
    num_dataframe_vars = len(dataframe_vars)

    if num_vars_accounted_for == num_dataframe_vars:
        pass

    elif num_vars_accounted_for < num_dataframe_vars:
        msg = "Note, there are {:,} variables in {} but the inputs"
        msg_2 = " ind_vars, alt_specific_vars, and subset_specific_vars only"
        msg_3 = " account for {:,} variables."

        warnings.warn(msg.format(num_dataframe_vars, data_title) +
                      msg_2 + msg_3.format(num_vars_accounted_for))

    else:  # This means num_vars_accounted_for > num_dataframe_vars
        msg = "There are more variable specified in ind_vars, "
        msg_2 = "alt_specific_vars, and subset_specific_vars ({:,}) than there"
        msg_3 = " are variables in {} ({:,})"
        warnings.warn(msg +
                      msg_2.format(num_vars_accounted_for) +
                      msg_3.format(data_title, num_dataframe_vars))

    return None


def check_dataframe_for_duplicate_records(obs_id_col, alt_id_col, df):
    """
    Checks a cross-sectional dataframe of long-format data for duplicate
    observations. Duplicate observations are defined as rows with the same
    observation id value and the same alternative id value.

    Parameters
    ----------
    obs_id_col : str.
        Denotes the column in `df` that contains the observation ID
        values for each row.
    alt_id_col : str.
        Denotes the column in `df` that contains the alternative ID
        values for each row.
    df : pandas dataframe.
        The dataframe of long format data that is to be checked for duplicates.

    Returns
    -------
    None.
    """
    if df.duplicated(subset=[obs_id_col, alt_id_col]).any():
        msg = "One or more observation-alternative_id pairs is not unique."
        raise ValueError(msg)

    return None


def ensure_num_chosen_alts_equals_num_obs(obs_id_col, choice_col, df):
    """
    Checks that the total number of recorded choices equals the total number of
    observations. If this is not the case, raise helpful ValueError messages.

    Parameters
    ----------
    obs_id_col : str.
        Denotes the column in `df` that contains the observation ID values for
        each row.
    choice_col : str.
        Denotes the column in `long_data` that contains a one if the
        alternative pertaining to the given row was the observed outcome for
        the observation pertaining to the given row and a zero otherwise.
    df : pandas dataframe.
        The dataframe whose choices and observations will be checked.

    Returns
    -------
    None.
    """
    num_obs = df[obs_id_col].unique().shape[0]
    num_choices = df[choice_col].sum()

    if num_choices < num_obs:
        msg = "One or more observations have not chosen one "
        msg_2 = "of the alternatives available to him/her"
        raise ValueError(msg + msg_2)
    if num_choices > num_obs:
        msg = "One or more observations has chosen multiple alternatives"
        raise ValueError(msg)

    return None


def check_type_and_values_of_alt_name_dict(alt_name_dict, alt_id_col, df):
    """
    Ensures that `alt_name_dict` is a dictionary and that its keys are in the
    alternative id column of `df`. Raises helpful errors if either condition
    is not met.

    Parameters
    ----------
    alt_name_dict : dict.
        A dictionary whose keys are the possible values in
        `df[alt_id_col].unique()`. The values should be the name that one
        wants to associate with each alternative id.
    alt_id_col : str.
        Denotes the column in `df` that contains the alternative ID values for
        each row.
    df : pandas dataframe.
        The dataframe of long format data that contains the alternative IDs.

    Returns
    -------
    None.
    """
    if not isinstance(alt_name_dict, dict):
        msg = "alt_name_dict should be a dictionary. Passed value was a {}"
        raise TypeError(msg.format(type(alt_name_dict)))

    if not all([x in df[alt_id_col].values for x in alt_name_dict.keys()]):
        msg = "One or more of alt_name_dict's keys are not "
        msg_2 = "in long_data[alt_id_col]"
        raise ValueError(msg + msg_2)

    return None


def ensure_ridge_is_scalar_or_none(ridge):
    """
    Ensures that `ridge` is either None or a scalar value. Raises a helpful
    TypeError otherwise.

    Parameters
    ----------
    ridge : int, float, long, or None.
        Scalar value or None, determining the L2-ridge regression penalty.

    Returns
    -------
    None.
    """
    if (ridge is not None) and not isinstance(ridge, Number):
        msg_1 = "ridge should be None or an int, float, or long."
        msg_2 = "The passed value of ridge had type: {}".format(type(ridge))
        raise TypeError(msg_1 + msg_2)

    return None


def create_design_matrix(long_form,
                         specification_dict,
                         alt_id_col,
                         names=None):
    """
    Parameters
    ----------
    long_form : pandas dataframe.
        Contains one row for each available alternative, for each observation.
    specification_dict : OrderedDict.
        Keys are a proper subset of the columns in `long_form_df`. Values are
        either a list or a single string, `"all_diff"` or `"all_same"`. If a
        list, the elements should be:

            - single objects that are within the alternative ID column of
              `long_form_df`
            - lists of objects that are within the alternative ID column of
              `long_form_df`. For each single object in the list, a unique
              column will be created (i.e. there will be a unique coefficient
              for that variable in the corresponding utility equation of the
              corresponding alternative). For lists within the
              `specification_dict` values, a single column will be created for
              all the alternatives within iterable (i.e. there will be one
              common coefficient for the variables in the iterable).
    alt_id_col : str.
        Column name which denotes the column in `long_form` that contains the
        alternative ID for each row in `long_form`.
    names : OrderedDict, optional.
        Should have the same keys as `specification_dict`. For each key:

            - if the corresponding value in `specification_dict` is "all_same",
              then there should be a single string as the value in names.
            - if the corresponding value in `specification_dict` is "all_diff",
              then there should be a list of strings as the value in names.
              There should be one string in the value in names for each
              possible alternative.
            - if the corresponding value in `specification_dict` is a list,
              then there should be a list of strings as the value in names.
              There should be one string the value in names per item in the
              value in `specification_dict`.
        Default == None.

    Returns
    -------
    design_matrix, var_names: tuple with two elements.
        First element is the design matrix, a numpy array with some number of
        columns and as many rows as are in `long_form`. Each column corresponds
        to a coefficient to be estimated. The second element is a list of
        strings denoting the names of each coefficient, with one variable name
        per column in the design matrix.
    """
    ##########
    # Check that the arguments meet this functions assumptions.
    # Fail gracefully if the arguments do not meet the function's requirements.
    #########
    check_argument_type(long_form, specification_dict)

    ensure_alt_id_in_long_form(alt_id_col, long_form)

    ensure_specification_cols_are_in_dataframe(specification_dict, long_form)

    # Find out what and how many possible alternatives there are
    unique_alternatives = np.sort(long_form[alt_id_col].unique())
    num_alternatives = len(unique_alternatives)

    check_type_and_values_of_specification_dict(specification_dict,
                                                unique_alternatives)

    # Check the user passed dictionary of names if the user passed such a list
    if names is not None:
        ensure_object_is_ordered_dict(names, "names")

        check_keys_and_values_of_name_dictionary(names,
                                                 specification_dict,
                                                 num_alternatives)

    ##########
    # Actually create the design matrix
    ##########
    # Create a list of the columns of independent variables
    independent_vars = []
    # Create a list of variable names
    var_names = []

    # Create the columns of the design matrix based on the specification dict.
    for variable in specification_dict:
        specification = specification_dict[variable]
        if specification == "all_same":
            # Create the variable column
            independent_vars.append(long_form[variable].values)
            # Create the column name
            var_names.append(variable)
        elif specification == "all_diff":
            for alt in unique_alternatives:
                # Create the variable column
                independent_vars.append((long_form[alt_id_col] == alt).values *
                                        long_form[variable].values)
                # create the column name
                var_names.append("{}_{}".format(variable, alt))
        else:
            for group in specification:
                if isinstance(group, list):
                    # Create the variable column
                    independent_vars.append(
                                     long_form[alt_id_col].isin(group).values *
                                     long_form[variable].values)
                    # Create the column name
                    var_names.append("{}_{}".format(variable, str(group)))

                else:  # the group is an integer
                    # Create the variable column
                    new_col_vals = ((long_form[alt_id_col] == group).values *
                                    long_form[variable].values)
                    independent_vars.append(new_col_vals)
                    # Create the column name
                    var_names.append("{}_{}".format(variable, group))

    # Create the final design matrix
    design_matrix = np.hstack((x[:, None] for x in independent_vars))

    # Use the list of names passed by the user, if the user passed such a list
    if names is not None:
        var_names = []
        for value in names.values():
            if isinstance(value, str):
                var_names.append(value)
            else:
                for inner_name in value:
                    var_names.append(inner_name)

    return design_matrix, var_names


def get_original_order_unique_ids(id_array):
    """
    Get the unique id's of id_array, in their original order of appearance.

    Parameters
    ----------
    id_array : 1D ndarray.
        Should contain the ids that we want to extract the unique values from.

    Returns
    -------
    original_order_unique_ids : 1D ndarray.
        Contains the unique ids from `id_array`, in their original order of
        appearance.
    """
    assert isinstance(id_array, np.ndarray)
    assert len(id_array.shape) == 1

    # Get the indices of the unique IDs in their order of appearance
    # Note the [1] is because the np.unique() call will return both the sorted
    # unique IDs and the indices
    original_unique_id_indices =\
        np.sort(np.unique(id_array, return_index=True)[1])

    # Get the unique ids, in their original order of appearance
    original_order_unique_ids = id_array[original_unique_id_indices]

    return original_order_unique_ids


def create_row_to_some_id_col_mapping(id_array):
    """
    Parameters
    ----------
    id_array : 1D ndarray.
        All elements of the array should be ints representing some id related
        to the corresponding row.

    Returns
    -------
    rows_to_ids : 2D scipy sparse array.
        Will map each row of id_array to the unique values of `id_array`. The
        columns of the returned sparse array will correspond to the unique
        values of `id_array`, in the order of appearance for each of these
        unique values.
    """
    # Get the unique ids, in their original order of appearance
    original_order_unique_ids = get_original_order_unique_ids(id_array)

    # Create a matrix with the same number of rows as id_array but a single
    # column for each of the unique IDs. This matrix will associate each row
    # as belonging to a particular observation using a one and using a zero to
    # show non-association.
    rows_to_ids = (id_array[:, None] ==
                   original_order_unique_ids[None, :]).astype(int)
    return rows_to_ids


def create_sparse_mapping(id_array, unique_ids=None):
    """
    Will create a scipy.sparse compressed-sparse-row matrix that maps
    each row represented by an element in id_array to the corresponding
    value of the unique ids in id_array.

    Parameters
    ----------
    id_array : 1D ndarray of ints.
        Each element should represent some id related to the corresponding row.
    unique_ids : 1D ndarray of ints, or None, optional.
        If not None, each element should be present in `id_array`. The elements
        in `unique_ids` should be present in the order in which one wishes them
        to appear in the columns of the resulting sparse array. For the
        `row_to_obs` and `row_to_mixers` mappings, this should be the order of
        appearance in `id_array`. If None, then the unique_ids will be created
        from `id_array`, in the order of their appearance in `id_array`.

    Returns
    -------
    mapping : 2D scipy.sparse CSR matrix.
        Will contain only zeros and ones. `mapping[i, j] == 1` where
        `id_array[i] == unique_ids[j]`. The id's corresponding to each column
        are given by `unique_ids`. The rows correspond to the elements of
        `id_array`.
    """
    # Create unique_ids if necessary
    if unique_ids is None:
        unique_ids = get_original_order_unique_ids(id_array)

    # Check function arguments for validity
    assert isinstance(unique_ids, np.ndarray)
    assert isinstance(id_array, np.ndarray)
    assert unique_ids.ndim == 1
    assert id_array.ndim == 1

    # Figure out which ids in id_array are represented in unique_ids
    represented_ids = np.in1d(id_array, unique_ids)
    # Determine the number of rows in id_array that are in unique_ids
    num_non_zero_rows = represented_ids.sum()
    # Figure out the dimensions of the resulting sparse matrix
    num_rows = id_array.size
    num_cols = unique_ids.size
    # Specify the non-zero values that will be present in the sparse matrix.
    data = np.ones(num_non_zero_rows, dtype=int)
    # Specify which rows will have non-zero entries in the sparse matrix.
    row_indices = np.arange(num_rows)[represented_ids]
    # Map the unique id's to their respective columns
    unique_id_dict = dict(zip(unique_ids, np.arange(num_cols)))
    # Figure out the column indices of the non-zero entries, and do so in a way
    # that avoids a key error (i.e. only look up ids that are represented)
    col_indices =\
        np.array([unique_id_dict[x] for x in id_array[represented_ids]])

    # Create and return the sparse matrix
    return csr_matrix((data, (row_indices, col_indices)),
                      shape=(num_rows, num_cols))


##########
# Create a function to create the mapping matrices from long form rows to the
# associated observations, from long form rows to alternative IDs, and
# from "chosen" long form rows to the associated observations
##########
def create_long_form_mappings(long_form,
                              obs_id_col,
                              alt_id_col,
                              choice_col=None,
                              nest_spec=None,
                              mix_id_col=None,
                              dense=False):
    """
    Parameters
    ----------
    long_form : pandas dataframe.
        Contains one row for each available alternative for each observation.
    obs_id_col : str.
        Denotes the column in `long_form` which contains the choice situation
        observation ID values for each row of `long_form`. Note each value in
        this column must be unique (i.e., individuals with repeat observations
        have unique `obs_id_col` values for each choice situation, and
        `obs_id_col` values are unique across individuals).
    alt_id_col : str.
        Denotes the column in long_form which contains the alternative ID
        values for each row of `long_form`.
    choice_col : str, optional.
        Denotes the column in long_form which contains a one if the alternative
        pertaining to the given row was the observed outcome for the
        observation pertaining to the given row and a zero otherwise.
        Default == None.
    nest_spec : OrderedDict, or None, optional.
        Keys are strings that define the name of the nests. Values are lists of
        alternative ids, denoting which alternatives belong to which nests.
        Each alternative id must only be associated with a single nest!
        Default == None.
    mix_id_col : str, optional.
        Denotes the column in long_form that contains the identification values
        used to denote the units of observation over which parameters are
        randomly distributed.
    dense : bool, optional.
        Determines whether or not scipy sparse matrices will be returned or
        dense numpy arrays.

    Returns
    -------
    mapping_dict : OrderedDict.
        Keys will be `["rows_to_obs", "rows_to_alts", "chosen_row_to_obs",
        "rows_to_nests"]`. If `choice_col` is None, then the value for
        `chosen_row_to_obs` will be None. Likewise, if `nest_spec` is None,
        then the value for `rows_to_nests` will be None. The value for
        "rows_to_obs" will map the rows of the `long_form` to the unique
        observations (on the columns) in their order of appearance. The value
        for `rows_to_alts` will map the rows of the `long_form` to the unique
        alternatives which are possible in the dataset (on the columns), in
        sorted order--not order of appearance. The value for
        `chosen_row_to_obs`, if not None, will map the rows of the `long_form`
        that contain the chosen alternatives to the specific observations those
        rows are associated with (denoted by the columns). The value of
        `rows_to_nests`, if not None, will map the rows of the `long_form` to
        the nest (denoted by the column) that contains the row's alternative.
        If `dense==True`, the returned values will be dense numpy arrays.
        Otherwise, the returned values will be scipy sparse arrays.
    """
    # Get the id_values from the long_form dataframe
    obs_id_values = long_form[obs_id_col].values
    alt_id_values = long_form[alt_id_col].values

    # Create a matrix with the same number of rows as long_form but a single
    # column for each of the unique IDs. This matrix will associate each row
    # as belonging to a particular observation using a one and using a zero to
    # show non-association.
    rows_to_obs = create_sparse_mapping(obs_id_values)

    # Determine all of the unique alternative IDs
    all_alternatives = np.sort(np.unique(alt_id_values))

    # Create a matrix with the same number of rows as long_form but a single
    # column for each of the unique alternatives. This matrix will associate
    # each row as belonging to a particular alternative using a one and using
    # a zero to show non-association.
    rows_to_alts = create_sparse_mapping(alt_id_values,
                                         unique_ids=all_alternatives)

    if choice_col is not None:
        # Create a matrix to associate each row with the same number of
        # rows as long_form but a 1 only if that row corresponds to an
        # alternative that a given observation (denoted by the columns)
        # chose.
        chosen_row_to_obs = csr_matrix(rows_to_obs.multiply(
                                long_form[choice_col].values[:, None]))
    else:
        chosen_row_to_obs = None

    if nest_spec is not None:
        # Determine how many nests there are
        num_nests = len(nest_spec)
        # Create a mapping between the alternative ids and their nests
        alt_id_to_nest_name = {}
        for key in nest_spec:
            for element in nest_spec[key]:
                alt_id_to_nest_name[element] = key

        # Create a mapping between nest names and nest ids
        nest_ids = np.arange(1, num_nests + 1)
        nest_name_to_nest_id = dict(zip(nest_spec.keys(), nest_ids))

        # Create an array of the nest ids of each row
        nest_id_vec = np.array([nest_name_to_nest_id[alt_id_to_nest_name[x]]
                                for x in alt_id_values])

        # Create the mapping matrix between each row and the nests
        rows_to_nests = create_sparse_mapping(nest_id_vec, unique_ids=nest_ids)

    else:
        rows_to_nests = None

    if mix_id_col is not None:
        # Create a mapping matrix between each row and each 'mixing unit'
        mix_id_array = long_form[mix_id_col].values
        rows_to_mixers = create_sparse_mapping(mix_id_array)

    else:
        rows_to_mixers = None

    # Create the dictionary of mapping matrices that is to be returned
    mapping_dict = OrderedDict()
    mapping_dict["rows_to_obs"] = rows_to_obs
    mapping_dict["rows_to_alts"] = rows_to_alts
    mapping_dict["chosen_row_to_obs"] = chosen_row_to_obs
    mapping_dict["rows_to_nests"] = rows_to_nests
    mapping_dict["rows_to_mixers"] = rows_to_mixers

    # Return the dictionary of mapping matrices.
    # If desired, convert the mapping matrices to dense matrices
    if dense:
        for key in mapping_dict:
            if mapping_dict[key] is not None:
                mapping_dict[key] = mapping_dict[key].A

    return mapping_dict


def convert_long_to_wide(long_data,
                         ind_vars,
                         alt_specific_vars,
                         subset_specific_vars,
                         obs_id_col,
                         alt_id_col,
                         choice_col,
                         alt_name_dict=None,
                         null_value=np.nan):
    """
    Converts a 'long format' dataframe of cross-sectional discrete choice data
    into a 'wide format' version of the same data.

    Parameters
    ----------
    long_data : pandas dataframe.
        Contains one row for each available alternative for each observation.
        Should have the specified `[obs_id_col, alt_id_col, choice_col]` column
        headings. The dtypes of all columns should be numeric.
    ind_vars : list of strings.
        Each element should be a column heading in `long_data` that denotes a
        variable that varies across observations but not across alternatives.
    alt_specific_vars : list of strings.
        Each element should be a column heading in `long_data` that denotes a
        variable that varies not only across observations but also across all
        alternatives.
    subset_specific_vars : dict.
        Each key should be a string that is a column heading of `long_data`.
        Each value should be a list of alternative ids denoting the subset of
        alternatives which the variable (i.e. the key) over actually varies.
        These variables should vary across individuals and across some
        alternatives.
    obs_id_col : str.
        Denotes the column in `long_data` that contains the observation ID
        values for each row.
    alt_id_col : str.
        Denotes the column in `long_data` that contains the alternative ID
        values for each row.
    choice_col : str.
        Denotes the column in `long_data` that contains a one if the
        alternative pertaining to the given row was the observed outcome for
        the observation pertaining to the given row and a zero otherwise.
    alt_name_dict : dict or None, optional
        If not None, should be a dictionary whose keys are the possible values
        in `long_data[alt_id_col].unique()`. The values should be the name
        that one wants to associate with each alternative id. Default == None.
    null_value : int, float, long, or `np.nan`, optional.
        The passed value will be used to fill cells in the wide format
        dataframe when that cell is unknown for a given individual. This is
        most commonly the case when there is a variable that varies across
        alternatives and one of the alternatives is not available for a given
        indvidual. The `null_value` will be inserted for that individual for
        that variable. Default == `np.nan`.

    Returns
    -------
    final_wide_df : pandas dataframe.
        Will contain one row per observational unit. Will contain an
        observation id column of the same name as `obs_id_col`. Will also
        contain a choice column of the same name as `choice_col`. Will contain
        one availability column per unique, observed alternative in the
        dataset. Will contain one column per variable in `ind_vars`. Will
        contain one column per alternative per variable in `alt_specific_vars`.
        Will contain one column per specified alternative per variable in
        `subset_specific_vars`.
    """
    ##########
    # Check that all columns of long_data are being
    # used in the conversion to wide format
    ##########
    num_vars_accounted_for = sum([len(x) for x in
                                  [ind_vars, alt_specific_vars,
                                   subset_specific_vars,
                                   [obs_id_col, alt_id_col, choice_col]]])

    ensure_all_columns_are_used(num_vars_accounted_for, long_data)

    ##########
    # Check that all columns one wishes to use are actually in long_data
    ##########
    ensure_columns_are_in_dataframe(ind_vars,
                                    long_data,
                                    col_title="ind_vars",
                                    data_title='long_data')

    ensure_columns_are_in_dataframe(alt_specific_vars,
                                    long_data,
                                    col_title="alt_specific_vars",
                                    data_title='long_data')

    ensure_columns_are_in_dataframe(subset_specific_vars.keys(),
                                    long_data,
                                    col_title="subset_specific_vars",
                                    data_title='long_data')

    identifying_cols = [choice_col, obs_id_col, alt_id_col]
    identifying_col_string = "[choice_col, obs_id_col, alt_id_col]"
    ensure_columns_are_in_dataframe(identifying_cols,
                                    long_data,
                                    col_title=identifying_col_string,
                                    data_title='long_data')

    ##########
    # Make sure that each observation-alternative pair is unique
    ##########
    check_dataframe_for_duplicate_records(obs_id_col, alt_id_col, long_data)

    ##########
    # Make sure each observation chose an alternative that's available.
    ##########
    # Make sure that the number of chosen alternatives equals the number of
    # individuals.
    ensure_num_chosen_alts_equals_num_obs(obs_id_col, choice_col, long_data)

    ##########
    # Check that the alternative ids in the alt_name_dict are actually the
    # alternative ids used in the long_data alt_id column.
    ##########
    if alt_name_dict is not None:
        check_type_and_values_of_alt_name_dict(alt_name_dict,
                                               alt_id_col,
                                               long_data)

    ##########
    # Figure out how many rows/columns should be in the wide format dataframe
    ##########
    # Note that the number of rows in wide format is the number of observations
    num_obs = long_data[obs_id_col].unique().shape[0]

    # Figure out the total number of possible alternatives for the dataset
    num_alts = long_data[alt_id_col].unique().shape[0]

    ############
    # Calculate the needed number of colums
    ############
    # For each observation, there is at least one column-- the observation id,
    num_cols = 1
    # We should have one availability column per alternative in the dataset
    num_cols += num_alts
    # We should also have one column to record the choice of each observation
    num_cols += 1
    # We should also have one column for each individual specific variable
    num_cols += len(ind_vars)
    # We should also have one column for each alternative specific variable,
    # for each alternative
    num_cols += len(alt_specific_vars) * num_alts
    # We should have one column for each subset alternative specific variable
    # for each alternative over which the variable varies
    for col in subset_specific_vars:
        num_cols += len(subset_specific_vars[col])

    ##########
    # Create the columns of the new dataframe
    ##########
    #####
    # Create the individual specific variable columns,
    # along with the observation id column
    #####
    new_df = long_data[[obs_id_col] + ind_vars].drop_duplicates()

    # Reset the index so that the index is not based on long_data
    new_df.reset_index(inplace=True)

    #####
    # Create the choice column in the wide data format
    #####
    new_df[choice_col] = long_data.loc[long_data[choice_col] == 1,
                                       alt_id_col].values

    #####
    # Create the availability columns
    #####
    # Get the various long form mapping matrices
    mapping_res = create_long_form_mappings(long_data,
                                            obs_id_col,
                                            alt_id_col)
    row_to_obs = mapping_res["rows_to_obs"]
    row_to_alt = mapping_res["rows_to_alts"]

    # Get the matrix of observations (rows) to available alternatives (columns)
    obs_to_alt = row_to_obs.T.dot(row_to_alt).todense()

    # Determine the unique alternative IDs in the order used in obs_to_alt
    alt_id_values = long_data[alt_id_col].values
    all_alternatives = np.sort(np.unique(alt_id_values))

    # Create the names for the availability columns
    if alt_name_dict is None:
        availability_col_names = ["availability_{}".format(int(x))
                                  for x in all_alternatives]
    else:
        availability_col_names = ["availability_{}".format(alt_name_dict[x])
                                  for x in all_alternatives]

    # Create a dataframe containing the availability columns for this dataset
    availability_df = pd.DataFrame(obs_to_alt,
                                   columns=availability_col_names)

    #####
    # Create the alternative specific and subset
    # alternative specific variable columns
    #####
    # For each alternative specific variable, create a wide format dataframe
    alt_specific_dfs = []
    for col in alt_specific_vars + list(subset_specific_vars.keys()):
        # Get the relevant values from the long format dataframe
        relevant_vals = long_data[col].values[:, None]
        # Create an wide format array of the relevant values
        obs_to_var = row_to_obs.T.dot(row_to_alt.multiply(relevant_vals))
        # Ensure that the wide format array is an ndarray with of dtype float
        if issparse(obs_to_var):
            obs_to_var = obs_to_var.toarray()
        # Ensure that obs_to_var has a float dtype
        obs_to_var = obs_to_var.astype(float)

        # Place a null value in columns where the alternative is not available
        # to a given observation
        if (obs_to_alt == 0).any():
            obs_to_var[np.where(obs_to_alt == 0)] = null_value

        # Create column names for the alternative specific variable columns
        if alt_name_dict is None:
            obs_to_var_names = ["{}_{}".format(col, int(x))
                                for x in all_alternatives]
        else:
            obs_to_var_names = ["{}_{}".format(col, alt_name_dict[x])
                                for x in all_alternatives]

        # Subset obs_to_vars and obs_to_var_names if col is in
        # subset_specific_vars
        if col in subset_specific_vars:
            # Calculate the relevant column indices for
            # the specified subset of alternatives
            relevant_alt_ids = subset_specific_vars[col]
            relevant_col_idx = np.where(np.in1d(all_alternatives,
                                                relevant_alt_ids))[0]
        else:
            relevant_col_idx = None

        # Create a dataframe containing the alternative specific variables
        # or the subset alternative specific variables for the given
        # variable in the long format dataframe
        if relevant_col_idx is None:
            obs_to_var_df = pd.DataFrame(obs_to_var,
                                         columns=obs_to_var_names)
        else:
            obs_to_var_df = pd.DataFrame(obs_to_var[:, relevant_col_idx],
                                         columns=[obs_to_var_names[x] for
                                                  x in relevant_col_idx])

        # Store the current alternative specific variable columns/dataframe
        alt_specific_dfs.append(obs_to_var_df)

    # Combine all of the various alternative specific variable dataframes
    final_alt_specific_df = pd.concat(alt_specific_dfs, axis=1)

    ##########
    # Construct the final wide format dataframe to be returned
    ##########
    final_wide_df = pd.concat([new_df[[obs_id_col]],
                               new_df[[choice_col]],
                               availability_df,
                               new_df[ind_vars],
                               final_alt_specific_df],
                              axis=1)

    # Make sure one has the correct number of rows and columns in
    # the final dataframe
    if final_wide_df.shape != (num_obs, num_cols):
        msg_1 = "There is an error with the dataframe that will be returned"
        msg_2 = "The shape of the dataframe should be {}".format((num_obs,
                                                                  num_cols))
        msg_3 = "Instead, the returned dataframe will have shape: {}"
        total_msg = msg_1 + msg_2 + msg_3.format(final_wide_df.shape)
        warnings.warn(total_msg)

    # Return the wide format dataframe
    return final_wide_df


def check_wide_data_for_blank_choices(choice_col, wide_data):
    """
    Checks `wide_data` for null values in the choice column, and raises a
    helpful ValueError if null values are found.

    Parameters
    ----------
    choice_col : str.
        Denotes the column in `wide_data` that is used to record each
        observation's choice.
    wide_data : pandas dataframe.
        Contains one row for each observation. Should contain `choice_col`.

    Returns
    -------
    None.
    """
    if wide_data[choice_col].isnull().any():
        msg_1 = "One or more of the values in wide_data[choice_col] is null."
        msg_2 = " Remove null values in the choice column or fill them in."
        raise ValueError(msg_1 + msg_2)

    return None


def ensure_unique_obs_ids_in_wide_data(obs_id_col, wide_data):
    """
    Ensures that there is one observation per row in wide_data. Raises a
    helpful ValueError if otherwise.

    Parameters
    ----------
    obs_id_col : str.
        Denotes the column in `wide_data` that contains the observation ID
        values for each row.
    wide_data : pandas dataframe.
        Contains one row for each observation. Should contain the specified
        `obs_id_col` column.

    Returns
    -------
    None.
    """
    if len(wide_data[obs_id_col].unique()) != wide_data.shape[0]:
        msg = "The values in wide_data[obs_id_col] are not unique, "
        msg_2 = "but they need to be."
        raise ValueError(msg + msg_2)

    return None


def ensure_chosen_alternatives_are_in_user_alt_ids(choice_col,
                                                   wide_data,
                                                   availability_vars):
    """
    Ensures that all chosen alternatives in `wide_df` are present in the
    `availability_vars` dict. Raises a helpful ValueError if not.

    Parameters
    ----------
    choice_col : str.
        Denotes the column in `wide_data` that contains a one if the
        alternative pertaining to the given row was the observed outcome for
        the observation pertaining to the given row and a zero otherwise.
    wide_data : pandas dataframe.
        Contains one row for each observation. Should contain the specified
        `choice_col` column.
    availability_vars : dict.
        There should be one key value pair for each alternative that is
        observed in the dataset. Each key should be the alternative id for the
        alternative, and the value should be the column heading in `wide_data`
        that denotes (using ones and zeros) whether an alternative is
        available/unavailable, respectively, for a given observation.
        Alternative id's, i.e. the keys, must be integers.

    Returns
    -------
    None.
    """
    if not wide_data[choice_col].isin(availability_vars.keys()).all():
        msg = "One or more values in wide_data[choice_col] is not in the user "
        msg_2 = "provided alternative ids in availability_vars.keys()"
        raise ValueError(msg + msg_2)

    return None


def ensure_each_wide_obs_chose_an_available_alternative(obs_id_col,
                                                        choice_col,
                                                        availability_vars,
                                                        wide_data):
    """
    Checks whether or not each observation with a restricted choice set chose
    an alternative that was personally available to him or her. Will raise a
    helpful ValueError if this is not the case.

    Parameters
    ----------
    obs_id_col : str.
        Denotes the column in `wide_data` that contains the observation ID
        values for each row.
    choice_col : str.
        Denotes the column in `wide_data` that contains a one if the
        alternative pertaining to the given row was the observed outcome for
        the observation pertaining to the given row and a zero otherwise.
    availability_vars : dict.
        There should be one key value pair for each alternative that is
        observed in the dataset. Each key should be the alternative id for the
        alternative, and the value should be the column heading in `wide_data`
        that denotes (using ones and zeros) whether an alternative is
        available/unavailable, respectively, for a given observation.
        Alternative id's, i.e. the keys, must be integers.
    wide_data : pandas dataframe.
        Contains one row for each observation. Should have the specified
        `[obs_id_col, choice_col] + availability_vars.values()` columns.

    Returns
    -------
    None
    """
    # Determine the various availability values for each observation
    wide_availability_values = wide_data[list(
        availability_vars.values())].values

    # Isolate observations for whom one or more alternatives are unavailable
    unavailable_condition = ((wide_availability_values == 0).sum(axis=1)
                                                            .astype(bool))

    # Iterate over the observations with one or more unavailable alternatives
    # Check that each such observation's chosen alternative was available
    problem_obs = []
    for idx, row in wide_data.loc[unavailable_condition].iterrows():
        if row.at[availability_vars[row.at[choice_col]]] != 1:
            problem_obs.append(row.at[obs_id_col])

    if problem_obs != []:
        msg = "The following observations chose unavailable alternatives:\n{}"
        raise ValueError(msg.format(problem_obs))

    return None


def ensure_all_wide_alt_ids_are_chosen(choice_col,
                                       alt_specific_vars,
                                       availability_vars,
                                       wide_data):
    """
    Checks to make sure all user-specified alternative id's, both in
    `alt_specific_vars` and `availability_vars` are observed in the choice
    column of `wide_data`.
    """
    sorted_alt_ids = np.sort(wide_data[choice_col].unique())
    try:
        problem_ids = [x for x in availability_vars
                       if x not in sorted_alt_ids]
        problem_type = "availability_vars"
        assert problem_ids == []

        problem_ids = []
        for new_column in alt_specific_vars:
            for alt_id in alt_specific_vars[new_column]:
                if alt_id not in sorted_alt_ids and alt_id not in problem_ids:
                    problem_ids.append(alt_id)
        problem_type = "alt_specific_vars"
        assert problem_ids == []
    except AssertionError:
        msg = "The following alternative ids from {} are not "
        msg_2 = "observed in wide_data[choice_col]:\n{}"
        raise ValueError(msg.format(problem_type) + msg_2.format(problem_ids))

    return None


def ensure_contiguity_in_observation_rows(obs_id_vector):
    """
    Ensures that all rows pertaining to a given choice situation are located
    next to one another. Raises a helpful ValueError otherwise. This check is
    needed because the hessian calculation function requires the design matrix
    to have contiguity in rows with the same observation id.

    Parameters
    ----------
    rows_to_obs : 2D scipy sparse array.
        Should map each row of the long format dataferame to the unique
        observations in the dataset.
    obs_id_vector : 1D ndarray of ints.
        Should contain the id (i.e. a unique integer) that corresponds to each
        choice situation in the dataset.

    Returns
    -------
    None.
    """
    # Check that the choice situation id for each row is larger than or equal
    # to the choice situation id of the preceding row.
    contiguity_check_array = (obs_id_vector[1:] - obs_id_vector[:-1]) >= 0
    if not contiguity_check_array.all():
        problem_ids = obs_id_vector[np.where(~contiguity_check_array)]
        msg_1 = "All rows pertaining to a given choice situation must be "
        msg_2 = "contiguous. \nRows pertaining to the following observation "
        msg_3 = "id's are not contiguous: \n{}"
        raise ValueError(msg_1 + msg_2 + msg_3.format(problem_ids.tolist()))
    else:
        return None


def convert_wide_to_long(wide_data,
                         ind_vars,
                         alt_specific_vars,
                         availability_vars,
                         obs_id_col,
                         choice_col,
                         new_alt_id_name=None):
    """
    Will convert a cross-sectional dataframe of discrete choice data from wide
    format to long format.

    Parameters
    ----------
    wide_data : pandas dataframe.
        Contains one row for each observation. Should have the specified
        `[obs_id_col, choice_col] + availability_vars.values()` columns.
    ind_vars : list of strings.
        Each element should be a column heading in `wide_data` that denotes a
        variable that varies across observations but not across alternatives.
    alt_specific_vars : dict.
        Each key should be a string that will be a column heading of the
        returned, long format dataframe. Each value should be a dictionary
        where the inner key is the alternative id and the value is the column
        heading in wide data that specifies the value of the outer key for the
        associated alternative. The variables denoted by the outer key should
        vary across individuals and across some or all alternatives.
    availability_vars : dict.
        There should be one key value pair for each alternative that is
        observed in the dataset. Each key should be the alternative id for the
        alternative, and the value should be the column heading in `wide_data`
        that denotes (using ones and zeros) whether an alternative is
        available/unavailable, respectively, for a given observation.
        Alternative id's, i.e. the keys, must be integers.
    obs_id_col : str.
        Denotes the column in `wide_data` that contains the observation ID
        values for each row.
    choice_col : str.
        Denotes the column in `wide_data` that contains a one if the
        alternative pertaining to the given row was the observed outcome for
        the observation pertaining to the given row and a zero otherwise.
    new_alt_id_name : str, optional.
        If not None, should be a string. This string will be used as the column
        heading for the alternative id column in the returned 'long' format
        dataframe. If not passed, this column will be called `'alt_id'`.
        Default == None.

    Returns
    -------
    final_long_df : pandas dataframe.
        Will contain one row for each available alternative for each
        observation. Will contain an observation id column of the same name as
        `obs_id_col`. Will also contain a choice column of the same name as
        `choice_col`. Will also contain an alternative id column called
        `alt_id` if `new_alt_id_col == None`, or `new_alt_id` otherwise. Will
        contain one column per variable in `ind_vars`. Will contain one column
        per key in `alt_specific_vars`.
    """
    ##########
    # Check that all columns of wide_data are being
    # used in the conversion to long format
    ##########
    all_alt_specific_cols = []
    for var_dict in alt_specific_vars.values():
        all_alt_specific_cols.extend(var_dict.values())

    vars_accounted_for = set(ind_vars +
                             # converto list explicitly to support
                             # both python 2 and 3
                             list(availability_vars.values()) +
                             [obs_id_col, choice_col] +
                             all_alt_specific_cols)
    num_vars_accounted_for = len(vars_accounted_for)

    ensure_all_columns_are_used(num_vars_accounted_for,
                                wide_data,
                                data_title='wide_data')

    ##########
    # Check that all columns one wishes to use are actually in wide_data
    ##########
    ensure_columns_are_in_dataframe(ind_vars,
                                    wide_data,
                                    col_title='ind_vars',
                                    data_title='wide_data')

    ensure_columns_are_in_dataframe(availability_vars.values(),
                                    wide_data,
                                    col_title='availability_vars',
                                    data_title='wide_data')

    for new_column in alt_specific_vars:
        for alt_id in alt_specific_vars[new_column]:
            old_column = alt_specific_vars[new_column][alt_id]
            ensure_columns_are_in_dataframe([old_column],
                                            wide_data,
                                            col_title="alt_specific_vars",
                                            data_title='wide_data')

    ensure_columns_are_in_dataframe([choice_col, obs_id_col],
                                    wide_data,
                                    col_title='[choice_col, obs_id_col]',
                                    data_title='wide_data')

    ##########
    # Check the integrity of the various columns present in wide_data
    ##########
    # Make sure the observation id's are unique (i.e. one per row)
    ensure_unique_obs_ids_in_wide_data(obs_id_col, wide_data)

    # Make sure there are no blank values in the choice column
    check_wide_data_for_blank_choices(choice_col, wide_data)

    ##########
    # Check that the user-provided alternative ids are observed
    # in the realized choices.
    ##########
    ensure_all_wide_alt_ids_are_chosen(choice_col,
                                       alt_specific_vars,
                                       availability_vars,
                                       wide_data)

    ##########
    # Check that the realized choices are all in the
    # user-provided alternative ids
    ##########
    ensure_chosen_alternatives_are_in_user_alt_ids(choice_col,
                                                   wide_data,
                                                   availability_vars)

    ##########
    # Make sure each observation chose a personally available alternative.
    ##########
    ensure_each_wide_obs_chose_an_available_alternative(obs_id_col,
                                                        choice_col,
                                                        availability_vars,
                                                        wide_data)

    ##########
    # Figure out how many rows/columns should be in the long format dataframe
    ##########
    # Note that the number of rows in long format is the
    # number of available alternatives across all observations
    sorted_alt_ids = np.sort(wide_data[choice_col].unique())
    sorted_availability_cols = [availability_vars[x] for x in sorted_alt_ids]
    num_rows = wide_data[sorted_availability_cols].sum(axis=0).sum()

    #####
    # Calculate the needed number of colums
    #####
    # For each observation, there is at least one column-- the observation id,
    num_cols = 1
    # We should also have one alternative id column
    num_cols += 1
    # We should also have one column to record the choice of each observation
    num_cols += 1
    # We should also have one column for each individual specific variable
    num_cols += len(ind_vars)
    # We should also have one column for each alternative specific variable,
    num_cols += len(alt_specific_vars.keys())

    ##########
    # Create the columns of the new dataframe
    ##########
    #####
    # Create the observation id column,
    #####
    # Determine the various availability values for each observation
    wide_availability_values = wide_data[list(
        availability_vars.values())].values
    new_obs_id_col = (wide_availability_values *
                      wide_data[obs_id_col].values[:, None]).ravel()
    # Make sure the observation id column has an integer data type
    new_obs_id_col = new_obs_id_col.astype(int)

    #####
    # Create the independent variable columns. Store them in a list.
    #####
    new_ind_var_cols = []
    for var in ind_vars:
        new_ind_var_cols.append((wide_availability_values *
                                 wide_data[var].values[:, None]).ravel())

    #####
    # Create the choice column in the long data format
    #####
    wide_choice_data = (wide_data[choice_col].values[:, None] ==
                        sorted_alt_ids[None, :])
    new_choice_col = wide_choice_data.ravel()
    # Make sure the choice column has an integer data type
    new_choice_col = new_choice_col.astype(int)

    #####
    # Create the alternative id column
    #####
    new_alt_id_col = (wide_availability_values *
                      sorted_alt_ids[None, :]).ravel().astype(int)
    # Make sure the alternative id column has an integer data type
    new_alt_id_col = new_alt_id_col.astype(int)

    #####
    # Create the alternative specific and subset
    # alternative specific variable columns
    #####
    # For each alternative specific variable, create a wide format array.
    # Then unravel that array to have the long format column for the
    # alternative specific variable. Store all the long format columns
    # in a list
    new_alt_specific_cols = []
    for new_col in alt_specific_vars:
        new_wide_alt_specific_cols = []
        for alt_id in sorted_alt_ids:
            # This will extract the correct values for the alternatives over
            # which the alternative specific variables vary
            if alt_id in alt_specific_vars[new_col]:
                rel_wide_column = alt_specific_vars[new_col][alt_id]
                new_col_vals = wide_data[rel_wide_column].values[:, None]
                new_wide_alt_specific_cols.append(new_col_vals)
            # This will create placeholder zeros for the alternatives that
            # the alternative specific variables do not vary over
            else:
                new_wide_alt_specific_cols.append(np.zeros(
                                                    (wide_data.shape[0], 1)))
        concatenated_long_column = np.concatenate(new_wide_alt_specific_cols,
                                                  axis=1).ravel()
        new_alt_specific_cols.append(concatenated_long_column)

    ##########
    # Construct the final wide format dataframe to be returned
    ##########
    # Identify rows that correspond to unavailable alternatives
    availability_condition = wide_availability_values.ravel() != 0

    # Figure out the names of all of the columns in the final
    # dataframe
    alt_id_column_name = ("alt_id" if new_alt_id_name is None
                          else new_alt_id_name)
    final_long_columns = ([obs_id_col,
                           alt_id_column_name,
                           choice_col] +
                          ind_vars +
                          list(alt_specific_vars.keys()))

    # Create a 'record array' of the final dataframe's columns
    # Note that record arrays are constructed from a list of 1D
    # arrays hence the array unpacking performed below for
    # new_ind_var_cols and new_alt_specific_cols
    all_arrays = ([new_obs_id_col,
                   new_alt_id_col,
                   new_choice_col] +
                  new_ind_var_cols +
                  new_alt_specific_cols)

    # Be sure to remove rows corresponding to unavailable alternatives
    # When creating the record array.
    df_recs = np.rec.fromarrays([all_arrays[pos][availability_condition]
                                 for pos in range(len(all_arrays))],
                                names=final_long_columns)

    # Create the final dataframe
    final_long_df = pd.DataFrame.from_records(df_recs)

    ##########
    # Make sure one has the correct number of rows and columns in
    # the final dataframe
    ##########
    try:
        assert final_long_df.shape == (num_rows, num_cols)
    except AssertionError:
        msg_1 = "There is an error with the dataframe that will be returned."
        msg_2 = "The shape of the dataframe should be {}".format((num_rows,
                                                                  num_cols))
        msg_3 = "Instead, the returned dataframe will have shape: {}"
        total_msg = "\n".join([msg_1, msg_2, msg_3])
        warnings.warn(total_msg.format(final_long_df.shape))

    # Return the wide format dataframe
    return final_long_df


def convert_mixing_names_to_positions(mixing_names, ind_var_names):
    """
    Parameters
    ----------
    mixing_names : list.
        All elements should be strings. Denotes the names of the index
        variables that are being treated as random variables.
    ind_var_names : list.
        All elements should be strings, representing (in order) the variables
        in the index.

    Returns
    -------
     list.
         All elements should be ints. Elements will be the position of each of
         the elements in mixing name, in the `ind_var_names` list.
    """
    return [ind_var_names.index(name) for name in mixing_names]
