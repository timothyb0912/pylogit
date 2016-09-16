# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 08:32:48 2016

@module:    choice_tools.py
@name:      Helpful Tools for Choice Model Estimation
@author:    Timothy Brathwaite
@summary:   Contains functions that help prepare one's data for choice model
            estimation or helps speed the estimation process (the 'mappings').
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


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
        raise ValueError(msg_1.format(type(data)) + msg_2)

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

##########
# Amend create_design_matrix to also function with binary discrete outcomes

# EDIT: The function will work as desired, provided that the long format
#       dataframe has rows for both of the alternatives for each observation
#       that is capable of being associated with both alternatives. The
#       fundamental need is to construct binary long form dataframes from wide
#       format data, but that is a completely different and as of yet
#       unwritten function.
#########

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
    if not isinstance(long_form, pd.DataFrame):
        msg = "long_form should be a pandas dataframe. It is a {}"
        raise ValueError(msg.format(type(long_form)))

    if not isinstance(specification_dict, OrderedDict):
        msg = "specification_dict should be an OrderedDict. It is a {}"
        raise ValueError(msg.format(type(specification_dict)))

    if alt_id_col not in long_form.columns:
        msg = "alt_id_col == {} is not a column in long_form."
        raise ValueError(msg.format(alt_id_col))

    # Find out what how many possible alternatives there are
    unique_alternatives = np.sort(long_form[alt_id_col].unique())
    num_alternatives = len(unique_alternatives)

    for key in specification_dict:
        if key not in long_form.columns:
            msg = "{} from specification_dict.keys() is not a long_form column"
            raise ValueError(msg.format(key))

        specification = specification_dict[key]
        if isinstance(specification, str):
            if specification not in ["all_same", "all_diff"]:
                msg = "specification_dict[{}] not in ['all_same', 'all_diff']"
                raise ValueError(msg.format(key))

        elif isinstance(specification, list):
            for group in specification:
                group_is_list = isinstance(group, list)
                if group_is_list:
                    for group_item in group:
                        if not isinstance(group_item, list):
                            if group_item not in unique_alternatives:
                                msg_1 = "{} in {} in specification_dict[{}]"
                                msg_2 = " is not in long_format[alt_id_col]"
                                raise ValueError(msg_1.format(group_item,
                                                              group,
                                                              key) + msg_2)

                        else:
                            for inner_item in group_item:
                                if inner_item not in unique_alternatives:
                                    msg = "{} in {} in specification_dict[{}]"
                                    msg_2 = "is not in long_format[alt_id_col]"
                                    raise ValueError(msg.format(inner_item,
                                                                group_item,
                                                                key) +
                                                     " " + msg_2)

                else:
                    if group not in unique_alternatives:
                        msg_1 = "{} in specification_dict[{}]"
                        msg_2 = " is not in long_format[alt_id_col]"
                        raise ValueError(msg_1.format(group, key) + msg_2)

        else:
            msg_1 = "specification_dict[{}]"
            msg_2 = " is not 'all_same', 'all_diff', or a list"
            raise Exception(msg_1.format(key) + msg_2)

    # Check the user passed dictionary of names if the user passed such a list
    if names is not None:
        if not isinstance(names, OrderedDict):
            msg = "names must be an OrderedDict. {} passed instead"
            raise ValueError(msg.format(type(names)))

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
                        raise ValueError(msg)

                else:    # This means speciffication == 'all_diff'
                    try:
                        assert isinstance(name_object, list)
                        assert len(name_object) == num_alternatives
                    except AssertionError:
                        msg_1 = "names[{}] should be a list with {} elements,"
                        msg_2 = " 1 element for each possible alternative"
                        msg = (msg_1.format(key, num_alternatives) + msg_2)
                        raise ValueError(msg)

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

                else:
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
    # Get the indices of the unique IDs in their order of appearance
    # Note the [1] is because the np.unique() call will return both the sorted
    # unique IDs and the indices
    original_unique_id_indices = np.sort(np.unique(id_array,
                                                   return_index=True)[1])

    # Get the unique ids, in their original order of appearance
    original_order_unique_ids = id_array[original_unique_id_indices]

    # Create a matrix with the same number of rows as id_array but a single
    # column for each of the unique IDs. This matrix will associate each row
    # as belonging to a particular observation using a one and using a zero to
    # show non-association.
    rows_to_ids = (id_array[:, None] ==
                   original_order_unique_ids[None, :]).astype(int)
    return rows_to_ids


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
    rows_to_obs = create_row_to_some_id_col_mapping(obs_id_values)

    # Determine all of the unique alternative IDs
    all_alternatives = np.sort(np.unique(alt_id_values))

    # Create a matrix with the same number of rows as long_form but a single
    # column for each of the unique alternatives. This matrix will associate
    # each row as belonging to a particular alternative using a one and using
    # a zero to show non-association.
    rows_to_alts = (alt_id_values[:, None] ==
                    all_alternatives[None, :]).astype(int)

    if choice_col is not None:
        # Create a matrix to associate each row with the same number of
        # rows as long_form but a 1 only if that row corresponds to an
        # alternative that a given observation (denoted by the columns)
        # chose.
        chosen_row_to_obs = rows_to_obs *\
                            long_form[choice_col].values[:, None]
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
        nest_ids = range(1, num_nests + 1)
        nest_name_to_nest_id = dict(zip(nest_spec.keys(), nest_ids))

        # Create an array of the nest ids of each row
        nest_id_vec = np.array([nest_name_to_nest_id[alt_id_to_nest_name[x]]
                                for x in alt_id_values])

        # Create the mapping matrix between each row and the nests
        rows_to_nests = (nest_id_vec[:, None] ==
                         np.array(nest_ids)[None, :]).astype(int)
    else:
        rows_to_nests = None

    if mix_id_col is not None:
        # Create a mapping matrix between each row and each 'mixing unit'
        mix_id_array = long_form[mix_id_col].values
        rows_to_mixers = create_row_to_some_id_col_mapping(mix_id_array)
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
    # If desired, convert the mapping matrices to sparse matrices
    if dense:
        return mapping_dict
    else:
        for key in mapping_dict:
            if mapping_dict[key] is not None:
                mapping_dict[key] = csr_matrix(mapping_dict[key])
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
        values for each row of `long_data`.
    alt_id_col : str.
        Denotes the column in `long_data` that contains the alternative ID
        values for each row of `long_form`.
    choice_col : str.
        Denotes the column in `long_data` that contains a one if the
        alternative pertaining to the given row was the observed outcome for
        the observation pertaining to the given row and a zero otherwise.
    alt_name_dict : dict or None, optional
        If not None, should be a dictionary whose keys are the possible values
        in `long_data[alt_id_col].uniuque()`. The values should be the name
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
    # Check that all columns of wide_data are being
    # used in the conversion to long format
    ##########
    vars_accounted_for = (ind_vars +
                          alt_specific_vars +
                          subset_specific_vars +
                          [obs_id_col, alt_id_col, choice_col])
    num_vars_accounted_for = len(vars_accounted_for)

    dataframe_vars = set(long_data.columns.tolist())
    num_dataframe_vars = len(dataframe_vars)

    if num_vars_accounted_for < num_dataframe_vars:
        msg = "Note, there are {:,} variables in long_data but the inputs"
        msg_2 = " ind_vars, alt_specific_vars, and subset_specific_vars only"
        msg_3 = " account for {:,} variables"

        missing_vars = dataframe_vars.difference(vars_accounted_for)
        msg_4 = "The variables that are unaccounted for are:\n{}.\n"

        print(msg.format(num_dataframe_vars) +
              msg_2 + msg_3.format(num_vars_accounted_for))
        print("\n" + msg_4.format(missing_vars))
    elif num_vars_accounted_for > num_dataframe_vars:
        msg = "There are more variable specified in ind_vars, "
        msg_2 = "alt_specific_vars, and subset_specific_vars ({:,}) than there"
        msg_3 = " are variables in long_data ({:,})"
        print(msg +
              msg_2.format(num_vars_accounted_for) +
              msg_3.format(num_dataframe_vars))

    ##########
    # Check that all columns one wishes to use are actually in long_data
    ##########
    long_columns = long_data.columns.tolist()
    try:
        problem_cols = [x for x in ind_vars if x not in long_columns]
        problem_type = "ind_vars"
        assert problem_cols == []

        problem_cols = [x for x in alt_specific_vars
                        if x not in long_columns]
        problem_type = "alt_specific_vars"
        assert problem_cols == []

        problem_cols = [x for x in subset_specific_vars.keys()
                        if x not in long_columns]
        problem_type = "subset_specific_vars"
        assert problem_cols == []

        problem_cols = [x for x in [choice_col, obs_id_col, alt_id_col]
                        if x not in long_columns]
        problem_type = ["choice_col, obs_id_col", "alt_id_col"]
        assert problem_cols == []

    except AssertionError as e:
        msg = "The following columns in {} are not in long_data"
        print(msg.format(problem_type))
        print(problem_cols)
        raise e

    ##########
    # Make sure that each observation-alternative pair is unique
    ##########
    try:
        assert long_data.duplicated(subset=[obs_id_col,
                                            alt_id_col]).any() == False
    except AssertionError as e:
        print("One or more observation-alternative_id pairs is not unique.")
        raise e

    ##########
    # Make sure each observation chose an alternative that's available.
    ##########
    # Make sure that the number of chosen alternatives equals the number of
    # individuals.
    try:
        num_obs = long_data[obs_id_col].unique().shape[0]
        num_choices = long_data[choice_col].sum()
        assert num_choices == num_obs
    except AssertionError as e:
        if num_choices < num_obs:
            msg = "One or more observations have not chosen one "
            msg_2 = "of the alternatives available to him/her"
            print(msg + msg_2)
        if num_choices > num_obs:
            print("One or more observations has chosen multiple alternatives")
        raise e

    ##########
    # Check that the alternative ids in the alt_name_dict are actually the
    # alternative ids used in the long_data alt_id column.
    ##########
    if alt_name_dict is not None:
        try:
            assert isinstance(alt_name_dict, dict)
        except AssertionError as e:
            msg = "alt_name_dict should be a dictionary. Passed value was a {}"
            print(msg.format(type(alt_name_dict)))
            raise e

        try:
            assert all([x in long_data[alt_id_col].values
                        for x in alt_name_dict.keys()])
        except AssertionError as e:
            msg = "One or more of alt_name_dict's keys are not "
            msg_2 = "in long_data[alt_id_col]"
            print(msg + msg_2)
            raise e

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
    row_to_obs, row_to_alt = create_long_form_mappings(long_data,
                                                       obs_id_col,
                                                       alt_id_col)

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
    for col in alt_specific_vars + subset_specific_vars.keys():
        obs_to_var = row_to_obs.T.dot(row_to_alt.multiply(
                                              long_data[col].values[:, None]))

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
    try:
        assert final_wide_df.shape == (num_obs, num_cols)
    except AssertionError as e:
        print("There is an error with the dataframe that will be returned")
        print("The shape of the dataframe should be {}".format((num_obs,
                                                                num_cols)))
        msg = "Instead, the returned dataframe will have shape: {}"
        print(msg.format(final_wide_df.shape))

    # Return the wide format dataframe
    return final_wide_df


def convert_wide_to_long(wide_data, ind_vars, alt_specific_vars,
                         availability_vars, obs_id_col, choice_col,
                         new_alt_id_name=None):
    """
    Parameters
    ----------
    wide_data : pandas dataframe.
        Contains one row for each observation. Should have the specified
        `[obs_id_col, choice_col] + availability_vars.values()` columns.
    ind_vars : list of strings.
        Each element should be a column heading in wide_data that denotes a
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
        Denotes the column in `long_data` that contains the observation ID
        values for each row of `long_data`.
    choice_col : str.
        Denotes the column in `long_data` that contains a one if the
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
                             availability_vars.values() +
                             [obs_id_col, choice_col] +
                             all_alt_specific_cols)
    num_vars_accounted_for = len(vars_accounted_for)

    dataframe_vars = set(wide_data.columns.tolist())
    num_dataframe_vars = len(dataframe_vars)

    if num_vars_accounted_for < num_dataframe_vars:
        msg = "Note, there are {:,} variables in wide_data but the inputs"
        msg_2 = " ind_vars, alt_specific_vars, and availability_vars only"
        msg_3 = " account for {:,} variables."

        missing_vars = dataframe_vars.difference(vars_accounted_for)
        msg_4 = "The variables that are unaccounted for are:\n{}.\n"

        print(msg.format(num_dataframe_vars) +
              msg_2 + msg_3.format(num_vars_accounted_for))
        print("\n" + msg_4.format(missing_vars))
    elif num_vars_accounted_for > num_dataframe_vars:
        msg = "There are more variable specified in ind_vars,"
        msg_2 = " alt_specific_vars, and subset_specific_vars ({:,})"
        msg_3 = " than there are variables in wide_data ({:,})"
        print(msg +
              msg_2.format(num_vars_accounted_for) +
              msg_3.format(num_dataframe_vars))

    ##########
    # Check that all columns one wishes to use are actually in wide_data
    ##########
    wide_columns = wide_data.columns.tolist()
    try:
        problem_cols = [x for x in ind_vars if x not in wide_columns]
        problem_type = "ind_vars"
        assert problem_cols == []

        problem_cols = [x for x in availability_vars.values()
                        if x not in wide_columns]
        problem_type = "availability_vars"
        assert problem_cols == []

        problem_cols = []
        for new_column in alt_specific_vars:
            for alt_id in alt_specific_vars[new_column]:
                old_column = alt_specific_vars[new_column][alt_id]
                if old_column not in wide_columns:
                    problem_cols.append(old_column)
        problem_type = "alt_specific_vars"
        assert problem_cols == []

        problem_cols = [x for x in [choice_col, obs_id_col]
                        if x not in wide_columns]
        problem_type = ["choice_col, obs_id_col"]
        assert problem_cols == []

    except AssertionError as e:
        msg = "The following columns in {} are not in wide_data"
        print(msg.format(problem_type))
        print(problem_cols)
        raise e

    ##########
    # Check the integrity of the various columns present in wide_data
    ##########
    # Make sure the observation id's are unique (i.e. one per row)
    try:
        assert len(wide_data[obs_id_col].unique()) == wide_data.shape[0]
    except AssertionError as e:
        msg = "The values in wide_data[obs_id_col] are not unique, "
        msg_2 = "but they need to be."
        print(msg + msg_2)
        raise e

    # Make sure there are no blank values in the choice column
    try:
        assert wide_data[choice_col].notnull().all()
    except AssertionError as e:
        print("One or more of the values in wide_data[choice_col] is null.")
        print("Remove null values in the choice column or fill them in.")
        raise e

    ##########
    # Check that the user-provided alternative ids are observed
    # in the realized choices.
    ##########
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
    except AssertionError as e:
        msg = "The following alternative ids from {} are not "
        msg_2 = "observed in wide_data[choice_col]:"
        print(msg.format(problem_type) + msg_2)
        print(problem_ids)
        raise e

    ##########
    # Check that the realized choices are all in the
    # user-provided alternative ids
    ##########
    try:
        assert wide_data[choice_col].isin(availability_vars.keys()).all()
    except AssertionError as e:
        msg = "One or more values in wide_data[choice_col] is not in the user "
        msg_2 = "provided alternative ids in availability_vars.keys()"
        print(msg + msg_2)
        raise e

    ##########
    # Make sure each observation chose a personally available alternative.
    ##########
    # Determine the various availability values for each observation
    wide_availability_values = wide_data[availability_vars.values()].values

    # Isolate observations for whom one or more alternatives are unavailable
    unavailable_condition = ((wide_availability_values == 0).sum(axis=1)
                                                            .astype(bool))

    # Iterate over the observations with one or more unavailable alternatives
    # Check that each such observation's chosen alternative was available
    problem_obs = []
    for idx, row in wide_data.loc[unavailable_condition].iterrows():
        try:
            assert row.at[availability_vars[row.at[choice_col]]] == 1
        except AssertionError:
            problem_obs.append(row.at[obs_id_col])

    try:
        assert problem_obs == []
    except AssertionError as e:
        print("The following observations chose unavailable alternatives:")
        print(problem_obs)
        raise e

    ##########
    # Figure out how many rows/columns should be in the long format dataframe
    ##########
    # Note that the number of rows in long format is the
    # number of available alternatives across all observations
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
                          alt_specific_vars.keys())

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
    except AssertionError as e:
        print("There is an error with the dataframe that will be returned")
        print("The shape of the dataframe should be {}".format((num_rows,
                                                                num_cols)))
        msg = "Instead, the returned dataframe will have shape: {}"
        print(msg.format(final_long_df.shape))

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
