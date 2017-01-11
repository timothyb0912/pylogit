# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 22:07:30 2016

@module:    generalized_choice_model
@name:      Python Based Conditional Logit-type Models
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating multinomial, asymmetric
            conditional choice models (and standard conditional logit models).
@notes:     "Under the hood", this module indirectly or directly relies upon
            the following files:
            [base_multinomial_cm_v2.py,
             choice_calcs.py,
             choice_tools.py,
             conditional_logit.py,
             asym_logit.py,
             uneven_logit.py,
             scobit.py,
             clog_log.py,
             nested_logit.py
             mixed_logit.py]
"""
from . import conditional_logit as mnl
from . import asym_logit
from . import uneven_logit
from . import scobit
from . import clog_log
from . import nested_logit
from . import mixed_logit

# Create a dictionary relating the model type parameter to the class that
# the general choice model should inherit from
model_type_to_class = {"MNL": mnl.MNL,
                       "Asym": asym_logit.MNAL,
                       "Cloglog": clog_log.MNCL,
                       "Scobit": scobit.MNSL,
                       "Uneven": uneven_logit.MNUL,
                       "Nested Logit": nested_logit.NestedLogit,
                       "Mixed Logit": mixed_logit.MixedLogit}

# Create a dictionary relating the model type parameter to the name of the
# class that the general choice model should inherit from
model_type_to_class_name = {"MNL": "MNL",
                            "Asym": "MNAL",
                            "Cloglog": "MNCL",
                            "Scobit": "MNSL",
                            "Uneven": "MNUL",
                            "Nested Logit": "NestedLogit",
                            "Mixed Logit": "MixedLogit"}

# Store the names of the model_type kwargs that are valid.
valid_model_types = model_type_to_class.keys()


# Create a function that checks the user's model type and ensures its validity
def ensure_valid_model_type(specified_type, model_type_list):
    """
    Checks to make sure that `specified_type` is in `model_type_list` and
    raises a helpful error if this is not the case.

    Parameters
    ----------
    specified_type : str.
        Denotes the user-specified model type that is to be checked.
    model_type_list : list of strings.
        Contains all of the model types that are acceptable kwarg values.

    Returns
    -------
    None.
    """
    if specified_type not in model_type_list:
        msg_1 = "The specified model_type was not valid."
        msg_2 = "Valid model-types are {}".format(model_type_list)
        msg_3 = "The passed model-type was: {}".format(specified_type)
        total_msg = "\n".join([msg_1, msg_2, msg_3])
        raise ValueError(total_msg)
    return None


def create_choice_model(data,
                        alt_id_col,
                        obs_id_col,
                        choice_col,
                        specification,
                        model_type,
                        intercept_ref_pos=None,
                        shape_ref_pos=None,
                        names=None,
                        intercept_names=None,
                        shape_names=None,
                        nest_spec=None,
                        mixing_id_col=None,
                        mixing_vars=None):
    """
    Parameters
    ----------
    data : string or pandas dataframe.
        If `data` is a string, it should be an absolute or relative path to
        a CSV file containing the long format data for this choice model.
        Note long format has one row per available alternative for each
        observation. If `data` is a pandas dataframe, `data` should already
        be in long format.
    alt_id_col : string.
        Should denote the column in data that contains the alternative
        identifiers for each row.
    obs_id_col : string.
        Should denote the column in data that contains the observation
        identifiers for each row.
    choice_col : string.
        Should denote the column in data which contains the ones and zeros
        that denote whether or not the given row corresponds to the chosen
        alternative for the given individual.
    specification : OrderedDict.
        Keys are a proper subset of the columns in `long_form_df`. Values are
        either a list or a single string, `all_diff` or `all_same`. If a list,
        the elements should be:
            1) single objects that are within the alternative ID column of
               `long_form_df`
            2) lists of objects that are within the alternative ID column of
               `long_form_df`. For each single object in the list, a unique
                column will be created (i.e. there will be a unique
                coefficient for that variable in the corresponding utility
                equation of the corresponding alternative). For lists within
                the `specification_dict` values, a single column will be
                created for all the alternatives within iterable (i.e. there
                will be one common coefficient for the variables in the
                iterable).
    model_type : string.
        Denotes the model type of the choice_model being instantiated.
        Should be one of the following values:

            - "MNL"
            - "Asym"
            - "Cloglog"
            - "Scobit"
            - "Uneven"
            - "Nested Logit"
            - "Mixed Logit"
    intercept_ref_pos : int, optional.
        Valid only when the intercepts being estimated are not part of the
        index. Specifies the alternative in the ordered array of unique
        alternative ids whose intercept or alternative-specific constant is
        not estimated, to ensure model identifiability. Default == None.
    shape_ref_pos : int, optional.
        Specifies the alternative in the ordered array of unique
        alternative ids whose shape parameter is not estimated, to ensure
        model identifiability. Default == None.
    names : OrderedDict or None, optional.
        Should have the same keys as `specification`. For each key:

            - if the corresponding value in `specification` is
              "all_same", then there should be a single string as the value
              in names.
            - if the corresponding value in `specification` is "all_diff",
              then there should be a list of strings as the value in names.
              There should be one string in the value in names for each
              possible alternative.
            - if the corresponding value in `specification` is a list, then
              there should be a list of strings as the value in names.
              There should be one string the value in names per item in the
              value in `specification`.
        Default == None.
    intercept_names : list of strings or None, optional.
        If a list is passed, then the list should have the same number of
        elements as there are possible alternatives in data, minus 1. Each
        element of the list should be the name of the corresponding
        alternative's intercept term, in sorted order of the possible
        alternative IDs. If None is passed, the resulting names that are
        shown in the estimation results will be
        ["Outside_ASC_{}".format(x) for x in shape_names]. Default = None.
    shape_names : list of strings or None, optional.
        If a list is passed, then the list should have the same number of
        elements as there are possible alternative IDs in data. Each
        element of the list should be a string denoting the name of the
        corresponding alternative, in sorted order of the possible
        alternative IDs. The resulting names which are shown in the
        estimation results will be
        ["shape_{}".format(x) for x in shape_names]. Default = None.
    nest_spec : OrderedDict or None, optional.
        Keys are strings that define the name of the nests. Values are
        lists of alternative ids, denoting which alternatives belong to
        which nests. Each alternative id  only be associated with a single
        nest! Default == None.
    mixing_id_col : str, or None, optional.
        Should be a column heading in `data`. Should denote the column in
        `data` which contains the identifiers of the units of observation
        over which the coefficients of the model are thought to be randomly
        distributed. If `model_type == "Mixed Logit"`, then `mixing_id_col`
        must be passed. Default == None.
    mixing_vars : list, or None, optional.
        All elements of the list should be strings. Each string should be
        present in the values of `names.values()` and they're associated
        variables should only be index variables (i.e. part of the design
        matrix). If `model_type == "Mixed Logit"`, then `mixing_vars` must
        be passed. Default == None.

    Returns
    -------
    model_obj : instantiation of the Choice Model Class corresponding
        to the model type passed as the function argument. The returned
        object will have been instantiated with the arguments passed to
        this function.
    """
    # Make sure the model type is valid
    ensure_valid_model_type(model_type, valid_model_types)

    # Carry out the appropriate instantiation process for the chosen
    # choice model
    model_kwargs = {"intercept_ref_pos": intercept_ref_pos,
                    "shape_ref_pos": shape_ref_pos,
                    "names": names,
                    "intercept_names": intercept_names,
                    "shape_names": shape_names,
                    "nest_spec": nest_spec,
                    "mixing_id_col": mixing_id_col,
                    "mixing_vars": mixing_vars}
    return model_type_to_class[model_type](data,
                                           alt_id_col,
                                           obs_id_col,
                                           choice_col,
                                           specification,
                                           **model_kwargs)
