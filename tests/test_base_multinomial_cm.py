"""
Tests for the base_multinomial_cm_v2.py file. These tests do not include tests
of the functions that perform the mathematical calculations necessary to
estimate the predict function.
"""
from __future__ import print_function

import unittest
import os
from collections import OrderedDict
from copy import deepcopy
from functools import reduce

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.stats

import pylogit.base_multinomial_cm_v2 as base_cm
import pylogit.choice_calcs as choice_calcs

try:
    # in Python 3 range returns an iterator instead of list
    # to maintain backwards compatibility use "old" version of range
    from past.builtins import range
except ImportError:
    pass

# Create a generic TestCase class so that we can define a single setUp method
# that is used by all the test suites.
class GenericTestCase(unittest.TestCase):
    def setUp(self):
        """
        Create a fake dataset and specification from which we can initialize a
        choice model.
        """
        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two
        # alternatives. There is one generic variable. Two alternative
        # specific constants and all three shape parameters are used.

        # Create the betas to be used during the tests
        self.fake_betas = np.array([-0.6])

        # Create the fake outside intercepts to be used during the tests
        self.fake_intercepts = np.array([1, 0.5])

        # Create names for the intercept parameters
        self.fake_intercept_names = ["ASC 1", "ASC 2"]

        # Record the position of the intercept that is not being estimated
        self.fake_intercept_ref_pos = 2

        # Create the shape parameters to be used during the tests. Note that
        # these are the reparameterized shape parameters, thus they will be
        # exponentiated in the fit_mle process and various calculations.
        self.fake_shapes = np.array([-1, 1])

        # Create names for the intercept parameters
        self.fake_shape_names = ["Shape 1", "Shape 2"]

        # Record the position of the shape parameter that is being constrained
        self.fake_shape_ref_pos = 2

        # Create an array of all model parameters
        self.fake_all_params = np.concatenate((self.fake_shapes,
                                               self.fake_intercepts,
                                               self.fake_betas))

        # The mapping between rows and alternatives is given below.
        self.fake_rows_to_alts = csr_matrix(np.array([[1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 0, 1]]))

        # Create the fake design matrix with columns denoting X
        # The intercepts are not included because they are kept outside the
        # index in the scobit model.
        self.fake_design = np.array([[1],
                                     [2],
                                     [3],
                                     [1.5],
                                     [3.5]])

        # Create the index array for this set of choice situations
        self.fake_index = self.fake_design.dot(self.fake_betas)

        # Create the needed dataframe for the Asymmetric Logit constructor
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                     "alt_id": [1, 2, 3, 1, 3],
                                     "choice": [0, 1, 0, 0, 1],
                                     "x": self.fake_design[:, 0],
                                     "intercept": [1 for i in range(5)]})

        # Record the various column names
        self.alt_id_col = "alt_id"
        self.obs_id_col = "obs_id"
        self.choice_col = "choice"

        # Create the index specification  and name dictionaryfor the model
        self.fake_specification = OrderedDict()
        self.fake_names = OrderedDict()
        self.fake_specification["x"] = [[1, 2, 3]]
        self.fake_names["x"] = ["x (generic coefficient)"]

        # Create a fake nest specification for the model
        self.fake_nest_spec = OrderedDict()
        self.fake_nest_spec["Nest 1"] = [1, 3]
        self.fake_nest_spec["Nest 2"] = [2]

        # Bundle args and kwargs used to construct the Asymmetric Logit model.
        self.constructor_args = [self.fake_df,
                                 self.alt_id_col,
                                 self.obs_id_col,
                                 self.choice_col,
                                 self.fake_specification]

        # Create a variable for the kwargs being passed to the constructor
        self.constructor_kwargs = {"intercept_ref_pos":
                                   self.fake_intercept_ref_pos,
                                   "shape_ref_pos": self.fake_shape_ref_pos,
                                   "names": self.fake_names,
                                   "intercept_names":
                                   self.fake_intercept_names,
                                   "shape_names": self.fake_shape_names,
                                   "nest_spec": self.fake_nest_spec}

        # Create a generic model object
        self.model_obj = base_cm.MNDC_Model(*self.constructor_args,
                                            **self.constructor_kwargs)


class InitializationTests(GenericTestCase):
    """
    This suite of tests should ensure that the logic in the initialization
    process is correctly executed.
    """

    def test_column_presence_in_data(self):
        """
        Ensure that the check for the presence of key columns works.
        """
        # Create column headings that are not in the dataframe used for testing
        fake_alt_id_col = "foo"
        fake_obs_id_col = "bar"
        fake_choice_col = "gerbil"
        bad_columns = [fake_alt_id_col, fake_obs_id_col, fake_choice_col]
        good_columns = [self.alt_id_col, self.obs_id_col, self.choice_col]

        for pos, bad_col in enumerate(bad_columns):
            # Create a set of arguments for the constructor where some of the
            # arguments are obviously incorrect
            column_list = deepcopy(good_columns)
            column_list[pos] = bad_col

            # Create the list of needed arguments
            args = [column_list, self.fake_df]

            self.assertRaises(ValueError,
                              base_cm.ensure_columns_are_in_dataframe,
                              *args)

            if pos == 2:
                self.assertRaises(ValueError,
                                  base_cm.ensure_columns_are_in_dataframe,
                                  *args,
                                  col_title="test_columns")

        # Make sure good columns don't trigger an error
        good_results = base_cm.ensure_columns_are_in_dataframe(good_columns,
                                                               self.fake_df)
        self.assertIsNone(good_results)

        return None

    def test_specification_column_presence_in_data(self):
        """
        Ensure that the check for the presence of specification columns works.
        """
        # Create column headings that are not in the dataframe used for testing
        bad_specification_col = "foo"
        bad_spec_1 = deepcopy(self.fake_specification)

        # to support Python 2 and 3 convert keys explicitly to list
        good_col = list(self.fake_specification.keys())[0]
        bad_spec_1[bad_specification_col] = bad_spec_1[good_col]

        # Create a second bad specification dictionary by simply using a dict
        # instead of an OrderedDict.
        bad_spec_2 = dict.update(self.fake_specification)

        # Create the list of needed arguments
        for bad_specification, error in [(bad_spec_1, ValueError),
                                         (bad_spec_2, TypeError)]:
            args = [bad_specification, self.fake_df]
            func = base_cm.ensure_specification_cols_are_in_dataframe

            self.assertRaises(error, func, *args)

        return None

    def test_numeric_validity_check_for_specification_cols(self):
        """
        Ensure that ValueErrors are raised if a column has a non-numeric
        dtype, positive or negative infinity vaues, or NaN values.
        """
        # Create a variety of "bad" columns for 'x'
        bad_exogs = [np.array(['foo', 'bar', 'gerbil', 'sat', 'sun']),
                     np.array([1, 2, 3, np.NaN, 1]),
                     np.array([1, 2, np.inf, 0.5, 0.9]),
                     np.array([1, 2, -np.inf, 0.5, 0.9]),
                     np.array([1, 'foo', -np.inf, 0.5, 0.9])]

        fake_df = deepcopy(self.fake_df)

        for bad_array in bad_exogs:
            # Overwrite the original x value
            del fake_df["x"]
            fake_df["x"] = bad_array

            self.assertRaises(ValueError,
                              base_cm.ensure_valid_nums_in_specification_cols,
                              *[self.fake_specification, fake_df])

        return None

    def test_ensure_ref_position_is_valid(self):
        """
        Checks that ValueError is raised for the various ways a ref_position
        might be invalid.
        """
        # Set the number of alternatives for the model and the title of the
        # parameters being estimated.
        num_alts = 3
        param_title = 'intercept_names'
        good_ref = 2
        args = [good_ref, num_alts, param_title]

        # Make ref_position None when estimating intercept!
        # Make ref_position something other than None or an int
        # Make ref_position an int outside [0, num_alts - 1]
        for bad_ref, error in [(None, ValueError),
                               ('turtle', TypeError),
                               (-1, ValueError),
                               (3, ValueError)]:
            args[0] = bad_ref
            self.assertRaises(error,
                              base_cm.ensure_ref_position_is_valid,
                              *args)

        return None

    def test_too_few_shape_or_intercept_names(self):
        """
        Ensure ValueError is raised if we have too few shape / intercept names.
        """
        names = ["Param 1", "Param 2"]
        num_alts = 4
        constrained_param = True
        for param_string in ["shape_params", "intercept_params"]:
            args = [names, num_alts, constrained_param, param_string]
            self.assertRaises(ValueError,
                              base_cm.check_length_of_shape_or_intercept_names,
                              *args)

        return None

    def test_too_many_shape_or_intercept_names(self):
        """
        Ensure ValueError is raised if we have too many shape/intercept names.
        """
        names = ["Param 1", "Param 2", "Param 3"]
        num_alts = 3
        constrained_param = True
        param_string = "shape_params"
        args = [names, num_alts, constrained_param, param_string]
        self.assertRaises(ValueError,
                          base_cm.check_length_of_shape_or_intercept_names,
                          *args)

        return None

    def test_ensure_object_is_ordered_dict(self):
        """
        Ensures that TypeError is raised if nest_spec is not an OrderedDict
        """
        new_nest_spec = {"Nest_1": [1, 2],
                         "Nest_2": [3]}

        self.assertRaises(TypeError,
                          base_cm.ensure_object_is_ordered_dict,
                          new_nest_spec,
                          "nest_spec")

        return None

    def test_check_type_of_nest_spec_keys_and_values(self):
        """
        Ensures that TypeError is raised if the keys of nest_spec are not
        strings and if the values of nest_spec are not lists.
        """
        new_nest_spec_1 = {1: [1, 2],
                           "Nest_2": [3]}

        new_nest_spec_2 = {"Nest_1": (1, 2),
                           "Nest_2": (3,)}

        for bad_spec in [new_nest_spec_1, new_nest_spec_2]:
            self.assertRaises(TypeError,
                              base_cm.check_type_of_nest_spec_keys_and_values,
                              bad_spec)

        return None

    def test_check_for_empty_nests_in_nest_spec(self):
        """
        Ensures that ValueError is raised if any of the values of nest_spec are
        empty lists.
        """
        new_nest_spec = {"Nest_1": [1, 2],
                         "Nest_2": []}

        self.assertRaises(ValueError,
                          base_cm.check_for_empty_nests_in_nest_spec,
                          new_nest_spec)

        return None

    def test_ensure_alt_ids_in_nest_spec_are_ints(self):
        """
        Ensure that ValueError is raised when non-integer elements are passed
        in the lists used as values in nest_spec.
        """
        new_nest_spec_1 = {"Nest_1": [1, '2'],
                           "Nest_2": [3]}
        new_nest_spec_2 = {"Nest_1": [1, 2],
                           "Nest_2": [None]}

        for bad_spec in [new_nest_spec_1, new_nest_spec_2]:
            list_elements = reduce(lambda x, y: x + y,
                                   [bad_spec[key] for key in bad_spec])

            self.assertRaises(ValueError,
                              base_cm.ensure_alt_ids_in_nest_spec_are_ints,
                              *[bad_spec, list_elements])

        return None

    def test_ensure_alt_ids_are_only_in_one_nest(self):
        """
        Ensure ValueError is raised when alternative ids are in multiple nests.
        """
        new_nest_spec = {"Nest_1": [1, 2],
                         "Nest_2": [2, 3]}

        list_elements = reduce(lambda x, y: x + y,
                               [new_nest_spec[key] for key in new_nest_spec])

        self.assertRaises(ValueError,
                          base_cm.ensure_alt_ids_are_only_in_one_nest,
                          *[new_nest_spec, list_elements])

        return None

    def test_ensure_all_alt_ids_have_a_nest(self):
        """
        Ensure ValueError is raised when any alternative id lacks a nest.
        """
        new_nest_spec = {"Nest_1": [1],
                         "Nest_2": [3]}

        list_elements = reduce(lambda x, y: x + y,
                               [new_nest_spec[key] for key in new_nest_spec])

        all_ids = [1, 2, 3]

        self.assertRaises(ValueError,
                          base_cm.ensure_all_alt_ids_have_a_nest,
                          *[new_nest_spec, list_elements, all_ids])

        return None

    def test_ensure_nest_alts_are_valid_alts(self):
        """
        Ensure ValueError is raised when any alternative id in the nest_spec
        is not contained in the universal choice set for this dataset.
        """
        new_nest_spec = {"Nest_1": [1, 2],
                         "Nest_2": [3, 4]}

        list_elements = reduce(lambda x, y: x + y,
                               [new_nest_spec[key] for key in new_nest_spec])

        all_ids = [1, 2, 3]

        self.assertRaises(ValueError,
                          base_cm.ensure_nest_alts_are_valid_alts,
                          *[new_nest_spec, list_elements, all_ids])

        return None

    def test_add_intercept_to_dataframe(self):
        """
        Ensure an intercept column is added to the dataset when appropriate.
        """
        new_specification = deepcopy(self.fake_specification)
        new_specification["intercept"] = [0, 1]

        original_df = self.fake_df.copy()
        del original_df["intercept"]

        # Apply the function to this dataset
        self.assertEqual("intercept" in original_df, False)

        base_cm.add_intercept_to_dataframe(new_specification, original_df)

        self.assertEqual("intercept" in original_df, True)
        self.assertEqual(original_df["intercept"].unique().size, 1)
        self.assertEqual(original_df["intercept"].unique(), 1)

        return None

    def test_ensure_all_mixing_vars_are_in_the_name_dict(self):
        """
        Ensures that, when using the
        `ensure_all_mixing_vars_are_in_the_name_dict` function, ValueErrors
        are raised when invalid mixing_vars arguments are used, and that None
        is returned otherwise.
        """
        # Create 'good' and 'bad' mixing_vars arguments
        good_mixing_vars = ["Tim", "Sreeta"]
        bad_mixing_vars = ["Tim", "Sreeta", "Feras"]

        # Create a name_dict for testing purposes
        name_dict = OrderedDict()
        name_dict["x"] = ["Tim", "Sreeta"]

        # Create a list of ind_var_names for testing_purposes
        independent_variable_names = name_dict["x"]

        # Alias the function to be tested
        func = base_cm.ensure_all_mixing_vars_are_in_the_name_dict

        # Record part of the msgs that one expects to see with and without
        # the name_dict
        msg_with_name_dict = "passed name dictionary: "
        msg_without_name_dict = "The default names that were generated were"

        # Perform the requisite tests
        self.assertIsNone(func(good_mixing_vars,
                               name_dict,
                               independent_variable_names))
        self.assertIsNone(func(None,
                               name_dict,
                               independent_variable_names))
        self.assertRaises(ValueError,
                          func,
                          bad_mixing_vars,
                          name_dict,
                          independent_variable_names)
        self.assertRaisesRegexp(ValueError,
                                msg_with_name_dict,
                                func,
                                bad_mixing_vars,
                                name_dict,
                                independent_variable_names)
        self.assertRaisesRegexp(ValueError,
                                msg_without_name_dict,
                                func,
                                bad_mixing_vars,
                                None,
                                independent_variable_names)

        return None

    def test_ensure_all_alternatives_are_chosen(self):
        """
        Ensures that a ValueError is raised if and only if some alternatives
        that were available in the dataset were not chosen in any choice
        situations.
        """
        # Create fake dataframes for the test.
        good_df = pd.DataFrame({"obs_id": [1, 1, 2, 2],
                                "alt_id": [1, 2, 1, 2],
                               "choice": [0, 1, 1, 0]})

        bad_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2, 2],
                               "alt_id": [1, 2, 3, 1, 2, 3],
                               "choice": [0, 1, 0, 1, 0, 0]})

        # Alias the function to be tested
        func = base_cm.ensure_all_alternatives_are_chosen

        # Perform the requisite tests
        self.assertIsNone(func("alt_id", "choice", good_df))
        self.assertRaisesRegexp(ValueError,
                                "The following alternative ID's were not"
                                " chosen in any choice situation:",
                                func,
                                "alt_id",
                                "choice",
                                bad_df)

        return None

class PredictHelperTests(GenericTestCase):
    """
    This suite tests the behavior of `check_param_list_validity()` and the
    functions called by this method.
    """

    def test_check_num_rows_of_parameter_array(self):
        """
        Ensure a ValueError is raised if the number of rows in an array is
        incorrect.
        """
        expected_num_rows = 4
        title = 'test_array'

        for i in [-1, 1]:
            test_array = np.zeros((expected_num_rows + i, 3))

            func_args = [test_array, expected_num_rows, title]
            self.assertRaises(ValueError,
                              base_cm.check_num_rows_of_parameter_array,
                              *func_args)

        # Test the behavior when there is no problem either
        test_array = np.zeros((expected_num_rows, 3))
        func_args = [test_array, expected_num_rows, title]
        func_results = base_cm.check_num_rows_of_parameter_array(*func_args)
        self.assertIsNone(func_results)

        return None

    def test_check_type_and_size_of_param_list(self):
        """
        Ensure that a ValueError is raised if param_list is not a list with the
        expected number of elements
        """
        expected_length = 4
        bad_param_list_1 = set(range(4))
        bad_param_list_2 = range(5)
        # Note that for the purposes of the function being tested, good is
        # defined as a list with four elements. Other functions check the
        # content of those elements
        good_param_list = range(4)

        for param_list in [bad_param_list_1, bad_param_list_2]:
            self.assertRaises(ValueError,
                              base_cm.check_type_and_size_of_param_list,
                              param_list,
                              expected_length)

        args = [good_param_list, expected_length]
        func_results = base_cm.check_type_and_size_of_param_list(*args)
        self.assertIsNone(func_results)

        return None

    def test_check_type_of_param_list_elements(self):
        """
        Ensures a TypeError is raised if the first element of param_list is
        not an ndarray and if each of the subsequent elements are not None or
        ndarrays.
        """
        bad_param_list_1 = ['foo', np.zeros(2)]
        bad_param_list_2 = [np.zeros(2), 'foo']
        good_param_list = [np.zeros(2), np.ones(2)]
        good_param_list_2 = [np.zeros(2), None]

        for param_list in [bad_param_list_1, bad_param_list_2]:
            self.assertRaises(TypeError,
                              base_cm.check_type_of_param_list_elements,
                              param_list)

        for param_list in [good_param_list, good_param_list_2]:
            args = [param_list]
            func_results = base_cm.check_type_of_param_list_elements(*args)
            self.assertIsNone(func_results)

        return None

    def test_check_num_columns_in_param_list_arrays(self):
        """
        Ensures a ValueError is raised if the various arrays in param_list do
        not all have the same number of columns
        """
        bad_param_list = [np.zeros((2, 3)), np.zeros((2, 4))]

        good_param_list_1 = [np.zeros((2, 3)), np.ones((2, 3))]
        good_param_list_2 = [np.zeros((2, 3)), None]

        self.assertRaises(ValueError,
                          base_cm.check_num_columns_in_param_list_arrays,
                          bad_param_list)

        for param_list in [good_param_list_1, good_param_list_2]:
            args = [param_list]
            results = base_cm.check_num_columns_in_param_list_arrays(*args)
            self.assertIsNone(results)

        return None

    def test_check_dimensional_equality_of_param_list_arrays(self):
        """
        Ensure that a ValueError is raised if the various arrays in param_list
        do not have the same number of dimensions.
        """
        bad_param_list_1 = [np.zeros((2, 3)), np.ones(2)]
        bad_param_list_2 = [np.zeros(3), np.ones((2, 3))]

        good_param_list_1 = [np.zeros((2, 3)), np.ones((2, 3))]
        good_param_list_2 = [np.zeros((2, 3)), None]

        # alias the function of interest so it fits on one line
        func = base_cm.check_dimensional_equality_of_param_list_arrays
        for param_list in [bad_param_list_1, bad_param_list_2]:
            self.assertRaises(ValueError, func, param_list)

        for param_list in [good_param_list_1, good_param_list_2]:
            self.assertIsNone(func(param_list))

        return None

    def test_check_param_list_validity(self):
        """
        Go thorough all possible types of 'bad' param_list arguments and
        ensure that the appropriate ValueErrors are raised. Ensure that 'good'
        param_list arguments make it through the function successfully
        """
        # Create a series of good parameter lists that should make it through
        # check_param_list_validity()
        good_list_1 = None
        good_list_2 = [np.zeros(1), np.ones(2), np.ones(2), np.ones(2)]
        good_list_3 = [np.zeros((1, 3)),
                       np.ones((2, 3)),
                       np.ones((2, 3)),
                       np.ones((2, 3))]

        good_lists = [good_list_1, good_list_2, good_list_3]

        # Create a series of bad parameter lists that should all result in
        # ValueErrors being raised.
        bad_list_1 = set(range(4))
        bad_list_2 = range(5)
        bad_list_3 = ['foo', np.zeros(2)]
        bad_list_4 = [np.zeros(2), 'foo']
        bad_list_5 = [np.zeros((2, 3)), np.zeros((2, 4))]
        bad_list_6 = [np.zeros((2, 3)), np.ones(2)]
        bad_list_7 = [np.zeros(3), np.ones((2, 3))]

        bad_lists = [bad_list_1, bad_list_2, bad_list_3,
                     bad_list_4, bad_list_5, bad_list_6,
                     bad_list_7]

        # Alias the function of interest to ensure it fits on one line
        func = self.model_obj.check_param_list_validity

        for param_list in good_lists:
            self.assertIsNone(func(param_list))

        for param_list in bad_lists:
            self.assertRaises(ValueError, func, param_list)

        return None

    def test_check_for_choice_col_based_on_return_long_probs(self):
        """
        Ensure that function appropriately raises a ValueError if choice_col
        is None and return_long_probs is False. Ensure that the function
        returns None otherwise.
        """
        # Alias the function being tested
        func = base_cm.check_for_choice_col_based_on_return_long_probs

        # Create a "good" and a "bad" set of arguments
        good_args = [[True, None], [False, "choice"]]
        bad_args = [False, None]

        # Note the error message that should be raised.
        msg = "If return_long_probs == False, then choice_col cannote be None."

        # Perform the tests
        for arg_set in good_args:
            self.assertIsNone(func(*arg_set))

        self.assertRaisesRegexp(ValueError,
                                msg,
                                func,
                                *bad_args)

        return None


class BaseModelMethodTests(GenericTestCase):
    """
    This suite tests the behavior of various methods for the base MNDC_Model.
    """

    def test_fit_mle_error(self):
        """
        Ensures that NotImplementedError is raised if someone tries to call the
        fit_mle method from the base MNDC_Model.
        """
        # Create a set of fake arguments.
        self.assertRaises(NotImplementedError,
                          self.model_obj.fit_mle,
                          np.arange(5))

        return None

    def test_to_pickle(self):
        """
        Ensure the to_pickle method works as expected
        """
        bad_filepath = 1234
        good_filepath = "test_model"

        self.assertRaises(ValueError, self.model_obj.to_pickle, bad_filepath)

        # Ensure that the file does not alread exist.
        self.assertFalse(os.path.exists(good_filepath + ".pkl"))

        # Use the function to be sure that the desired file gets created.
        self.model_obj.to_pickle(good_filepath)

        self.assertTrue(os.path.exists(good_filepath + ".pkl"))

        # Remove the newly created file to avoid needlessly creating files.
        os.remove(good_filepath + ".pkl")

        return None

    def test_print_summary(self):
        """
        Ensure that a NotImplementedError is raised when print_summaries is
        called before a model has actually been estimated.
        """
        # When the model object has no summary and fit_summary attributes,
        # raise a NotImplementedError
        self.assertRaises(NotImplementedError,
                          self.model_obj.print_summaries)

        # When the model object has summary and fit_summary attributes, print
        # them and return None.
        self.model_obj.summary = 'wombat'
        self.model_obj.fit_summary = 'koala'

        self.assertIsNone(self.model_obj.print_summaries())

        return None

    def test_get_statsmodels_summary(self):
        """
        Ensure that a NotImplementedError is raised if we try to get a
        statsmodels summary before estimating a model.
        """
        # When the model object has no 'estimation_success' attribute and we,
        # try to get a statsmodels_summary, raise a NotImplementedError
        self.assertRaises(NotImplementedError,
                          self.model_obj.get_statsmodels_summary)

        return None

    def test_conf_int(self):
        """
        Ensure that the confidence interval function returns expected results.
        """
        model_obj = self.model_obj
        model_obj.params = pd.Series([1.0, -1.0], index=["ASC", "x"])

        # Calculate the z-critical corresponding to a 2-sided 95% confidence
        # interval for a standardized variable.
        z_crit = 1.959963984540054

        # Specify a desired confidence interval
        interval_array = np.array([[0.5, 1.5],
                                   [-1.2, -0.8]])
        interval_df = pd.DataFrame(interval_array,
                                   index=model_obj.params.index,
                                   columns=["lower", "upper"])
        # Back out the needed standard errors
        std_errs = (interval_array[:, 1] - model_obj.params.values) / z_crit
        model_obj.standard_errors = pd.Series(std_errs, index=["ASC", "x"])

        # Get the function results
        df_func_results = model_obj.conf_int(return_df=True)
        array_func_results = model_obj.conf_int(return_df=True)
        subset_results = model_obj.conf_int(coefs=["ASC"])

        # Compare the results with what they should equal
        npt.assert_allclose(array_func_results, interval_array)
        self.assertTrue((df_func_results == interval_df).all().all())
        npt.assert_allclose(subset_results, interval_array[0, :][None, :])

        return None


class PostEstimationTests(GenericTestCase):
    """
    This suite of tests should ensure that the logic in the store_fit_results
    function is correctly executed.
    """
    def setUp(self):
        """
        Perform additional setup materials needed to test the store estimation
        results functions.
        """
        # Create the betas to be used during the tests
        self.fake_betas = np.array([-0.6])

        # Create the fake outside intercepts to be used during the tests
        self.fake_intercepts = np.array([1, 0.5])

        # Create names for the intercept parameters
        self.fake_intercept_names = ["ASC 1", "ASC 2"]

        # Record the position of the intercept that is not being estimated
        self.fake_intercept_ref_pos = 2

        # Create the shape parameters to be used during the tests. Note that
        # these are the reparameterized shape parameters, thus they will be
        # exponentiated in the fit_mle process and various calculations.
        self.fake_shapes = np.array([-1, 1])

        # Create names for the intercept parameters
        self.fake_shape_names = ["Shape 1", "Shape 2"]

        # Record the position of the shape parameter that is being constrained
        self.fake_shape_ref_pos = 2

        # Create an array of all model parameters
        self.fake_all_params = np.concatenate((self.fake_shapes,
                                               self.fake_intercepts,
                                               self.fake_betas))

        # The mapping between rows and alternatives is given below.
        self.fake_rows_to_alts = csr_matrix(np.array([[1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 0, 1]]))

        # Create the fake design matrix with columns denoting X
        # The intercepts are not included because they are kept outside the
        # index in the scobit model.
        self.fake_design = np.array([[1],
                                     [2],
                                     [3],
                                     [1.5],
                                     [3.5]])

        # Create the index array for this set of choice situations
        self.fake_index = self.fake_design.dot(self.fake_betas)

        # Create the needed dataframe for the Asymmetric Logit constructor
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                     "alt_id": [1, 2, 3, 1, 3],
                                     "choice": [0, 1, 0, 0, 1],
                                     "x": self.fake_design[:, 0],
                                     "intercept": [1 for i in range(5)]})

        # Record the various column names
        self.alt_id_col = "alt_id"
        self.obs_id_col = "obs_id"
        self.choice_col = "choice"

        # Create the index specification  and name dictionaryfor the model
        self.fake_specification = OrderedDict()
        self.fake_names = OrderedDict()
        self.fake_specification["x"] = [[1, 2, 3]]
        self.fake_names["x"] = ["x (generic coefficient)"]

        # Create a fake nest specification for the model
        self.fake_nest_spec = OrderedDict()
        self.fake_nest_spec["Nest 1"] = [1, 3]
        self.fake_nest_spec["Nest 2"] = [2]

        # Bundle args and kwargs used to construct the Asymmetric Logit model.
        self.constructor_args = [self.fake_df,
                                 self.alt_id_col,
                                 self.obs_id_col,
                                 self.choice_col,
                                 self.fake_specification]

        # Create a variable for the kwargs being passed to the constructor
        self.constructor_kwargs = {"intercept_ref_pos":
                                   self.fake_intercept_ref_pos,
                                   "shape_ref_pos": self.fake_shape_ref_pos,
                                   "names": self.fake_names,
                                   "intercept_names":
                                   self.fake_intercept_names,
                                   "shape_names": self.fake_shape_names,
                                   "nest_spec": self.fake_nest_spec}

        # Create a generic model object
        self.model_obj = base_cm.MNDC_Model(*self.constructor_args,
                                            **self.constructor_kwargs)

        # Create the attributes and post-estimation dictionary that is needed
        self.log_likelihood = -10
        self.fitted_probs = np.array([0.6, 0.78])
        self.long_fitted_probs = np.array([0.1, 0.6, 0.3, 0.22, 0.78])
        self.long_residuals = np.array([-0.1, 0.4, -0.3, 0.78, -0.78])
        self.ind_chi_squareds = (np.square(self.long_residuals) /
                                 self.long_fitted_probs)
        self.chi_square = self.ind_chi_squareds.sum()
        self.estimation_success = True
        self.estimation_message = "Estimation converged."
        self.null_log_likelihood = -23
        self.rho_squared = 1 - (self.log_likelihood / self.null_log_likelihood)
        self.rho_bar_squared = (self.rho_squared +
                                self.fake_all_params.shape[0] /
                                self.null_log_likelihood)
        self.estimation_message = "Succeded. This is just a test."
        self.estimation_success = True

        return None

    def test_check_result_dict_for_needed_keys(self):
        """
        Ensure that the _check_result_dict_for_needed_keys method raises a
        helpful ValueError if the results dict is missing a needed key and that
        the function returns None otherwise.
        """
        # Create a fake estimation results dictionary.
        base_dict = {x: None for x in base_cm.needed_result_keys}

        # Alias the function being tested
        func = self.model_obj._check_result_dict_for_needed_keys

        for key in base_cm.needed_result_keys:
            # Delete the needed key from the dictionary
            del base_dict[key]
            # Make sure that we get a value error when testing the function
            self.assertRaisesRegexp(ValueError,
                                    "keys are missing",
                                    func,
                                    base_dict)
            # Add the needed key back to the dictionary.
            base_dict[key] = None

        self.assertIsNone(func(base_dict))

        return None

    def test_create_results_summary(self):
        """
        Ensure that the expected summary dataframe is created when
        `_create_results_summary` is called. Ensure that if any of the
        necessary attributes are missing from the model object, then a
        NotImplementedError is raised.
        """
        # Note the attributes that are needed to create the results summary
        needed_attributes = ["params",
                             "standard_errors",
                             "tvalues",
                             "pvalues",
                             "robust_std_errs",
                             "robust_t_stats",
                             "robust_p_vals"]

        # Initialize a very simple series for each of these attributes.
        basic_series = pd.Series([5], index=["x"])
        for attr in needed_attributes:
            setattr(self.model_obj, attr, basic_series.copy())

        # Alias the function that is neeeded
        func = self.model_obj._create_results_summary

        # Note part of the error msg that is expected
        msg = "Call this function only after setting/calculating all other"

        # Check that the necessary NotImplementedErrors are raised.
        for attr in needed_attributes:
            delattr(self.model_obj, attr)
            # Make sure that we get a value error when testing the function
            self.assertRaisesRegexp(NotImplementedError,
                                    msg,
                                    func)
            # Set the attribute back
            setattr(self.model_obj, attr, basic_series.copy())

        # Check that the summary has all the elements that are expected
        func()
        self.assertIsInstance(self.model_obj.summary, pd.DataFrame)
        self.assertEqual(self.model_obj.summary.iloc[0].tolist(),
                         [5 for x in needed_attributes])

        return None

    def test_record_values_for_fit_summary_and_statsmodels(self):
        """
        Ensure that _record_values_for_fit_summary_and_statsmodels stores the
        desired attributes and values on the model object.
        """
        # Record the attributes that are needed for the function to work
        needed_attributes = ["fitted_probs",
                             "params",
                             "log_likelihood",
                             "standard_errors"]

        # Create a dictionary that maps the needed objects to their respective
        # values
        np.random.seed(0)
        values = [self.fitted_probs,
                  self.fake_all_params,
                  self.log_likelihood,
                  np.random.uniform(size=self.fake_all_params.shape[0])]
        attr_to_values = dict(zip(needed_attributes, values))

        # Store the input values on the model object
        for key in attr_to_values:
            setattr(self.model_obj, key, attr_to_values[key])

        # Alias the function that is being tested
        func = self.model_obj._record_values_for_fit_summary_and_statsmodels

        # Check that the function raises an appropriate error when a needed
        # attribute is missing
        for key in attr_to_values:
            # Delete the attribute
            delattr(self.model_obj, key)
            # Ensure the correct error is raised
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            error_msg = msg + msg_2
            self.assertRaisesRegexp(NotImplementedError,
                                    error_msg,
                                    func)
            # Put the attribute back.
            setattr(self.model_obj, key, attr_to_values[key])

        # Record the new attribute names and values that will be created
        new_attr_and_values = {"nobs": self.fitted_probs.shape[0],
                               "df_model": self.fake_all_params.shape[0],
                               "df_resid": self.fitted_probs.shape[0] -
                                           self.fake_all_params.shape[0],
                               "llf": self.log_likelihood,
                               "bse": attr_to_values["standard_errors"]}

        # Check that the desired attributes are all set when we call the
        # function with all of the needed inputs
        func()
        for key, value in new_attr_and_values.items():
            if key != "bse":
                self.assertEqual(value, getattr(self.model_obj, key))
            else:
                npt.assert_allclose(value, getattr(self.model_obj, key))

        return None

    def test_create_fit_summary(self):
        """
        Ensure that the appropriate error is raised when create_fit_summary is
        called without the correct input attributes stored on the model
        instance and ensure that the correct summary series is created when the
        function is called with the correct inputs.
        """
        # Make sure we have all attributes needed to create the results summary
        needed_attributes = ["df_model",
                             "nobs",
                             "null_log_likelihood",
                             "log_likelihood",
                             "rho_squared",
                             "rho_bar_squared",
                             "estimation_message"]
        correct_values = [self.fake_all_params.shape[0],
                          self.fitted_probs.shape[0],
                          self.null_log_likelihood,
                          self.log_likelihood,
                          self.rho_squared,
                          self.rho_bar_squared,
                          self.estimation_message]
        attr_to_values = dict(zip(needed_attributes, correct_values))

        # Store the input values on the model object
        for key in attr_to_values:
            setattr(self.model_obj, key, attr_to_values[key])

        # Alias the function that is being tested
        func = self.model_obj._create_fit_summary

        # Check that the function raises an appropriate error when a needed
        # attribute is missing
        for key in attr_to_values:
            # Delete the attribute
            delattr(self.model_obj, key)
            # Ensure the correct error is raised
            msg = "Call this function only after setting/calculating all other"
            msg_2 = " estimation results attributes"
            error_msg = msg + msg_2
            self.assertRaisesRegexp(NotImplementedError,
                                    error_msg,
                                    func)
            # Put the attribute back.
            setattr(self.model_obj, key, attr_to_values[key])

        # Note the desired index names of the values in the fit summary
        desired_index_names = ["Number of Parameters",
                               "Number of Observations",
                               "Null Log-Likelihood",
                               "Fitted Log-Likelihood",
                               "Rho-Squared",
                               "Rho-Bar-Squared",
                               "Estimation Message"]

        # Check that the desired attributes are all set when we call the
        # function with all of the needed inputs
        func()
        self.assertIsInstance(self.model_obj.fit_summary, pd.Series)
        self.assertEqual(self.model_obj.fit_summary.tolist(), correct_values)
        self.assertEqual(self.model_obj.fit_summary.index.tolist(),
                         desired_index_names)

        return None

    def test_store_inferential_results(self):
        """
        Ensure that appropriate errors are raised if incorrect arguments are
        passed to the function
        """
        # Create the arrays to be stored
        example_array = np.arange(4)
        example_array_2d = np.arange(16).reshape((4, 4))
        # Create the necessary arguments
        index_names = ["feras", "sreeta", "tim", "mustapha"]
        attribute_name = "phd"
        series_name = "doctoral"
        column_names = ["club", "116", "1st", "floor"]

        # Alias the function being tested
        func = self.model_obj._store_inferential_results

        # Make sure Assertion Errors are raised when using incorrect arguments
        self.assertRaises(AssertionError,
                          func,
                          example_array,
                          index_names,
                          attribute_name)

        self.assertRaises(AssertionError,
                          func,
                          example_array_2d,
                          index_names,
                          attribute_name)

        # Make sure that the function stores the correct attributes
        for example in [example_array, example_array_2d]:
            args = [example, index_names, attribute_name]
            kwargs = {"series_name": series_name, "column_names": column_names}
            func(*args, **kwargs)

            created_attribute = getattr(self.model_obj, attribute_name)
            # Make sure the attribute is of the correct type
            if len(example.shape) == 1:
                self.assertIsInstance(created_attribute, pd.Series)
            else:
                self.assertIsInstance(created_attribute, pd.DataFrame)
                # Make sure the attribute has the correct column names
                self.assertEqual(created_attribute.columns.tolist(),
                                 column_names)
            # Make sure the attribute has the correct index names
            self.assertEqual(created_attribute.index.tolist(), index_names)
            # Make sure the attribute has the correct values
            npt.assert_allclose(example, created_attribute.values)

        return None

    def test_store_optional_parameters(self):
        """
        Ensure that the function correctly stores optional parameters on the
        passed lists.
        """
        # Create a fake all_params and all_names list.
        all_params = [self.fake_shapes]
        all_names = deepcopy(self.fake_shape_names)

        # Create the necessary arguments for the function.
        name_list_attribute = "intercept_names"
        default_name_str = "ASC {}"
        param_attr_name = "intercepts"
        series_name = "intercepts"

        # Alias the function being tested
        func = self.model_obj._store_optional_parameters

        # Place the needed objects on the model object
        setattr(self.model_obj, name_list_attribute, self.fake_intercept_names)

        # Test the function
        func_args = [self.fake_intercepts,
                     name_list_attribute,
                     default_name_str,
                     all_names,
                     all_params,
                     param_attr_name,
                     series_name]
        new_all_names, new_all_params = func(*func_args)
        self.assertEqual(new_all_names,
                         self.fake_intercept_names + self.fake_shape_names)
        npt.assert_allclose(new_all_params[0], self.fake_intercepts)

        # Delete the name list attribute and try the function again
        setattr(self.model_obj, name_list_attribute, None)
        new_all_names, new_all_params = func(*func_args)
        self.assertEqual(new_all_names,
                         self.fake_intercept_names + self.fake_shape_names)
        npt.assert_allclose(new_all_params[0], self.fake_intercepts)

        return None

    def test_adjust_inferential_results_for_parameter_constraints(self):
        """
        Ensure that the adjustment for constrained parameters works as desired,
        placeing NaNs in the locations of the various inferential result series
        where a parameter was constrained.
        """
        # Take note of the various series that are created to hold the
        # inferential results.
        inferential_attributes = ["standard_errors",
                                  "tvalues",
                                  "pvalues",
                                  "robust_std_errs",
                                  "robust_t_stats",
                                  "robust_p_vals"]
        # Set a random seed for reproducibility
        np.random.seed(0)
        # Create the data needed for the various inferential results
        data = np.random.uniform(size=(5, len(inferential_attributes) + 1))
        dataframe = pd.DataFrame(data,
                                 columns=inferential_attributes + ["params"])
        # Store the inferential arrays on the model object
        for key in dataframe.columns:
            setattr(self.model_obj, key, dataframe[key])
        # Alias the function that is to be tested
        model_obj = self.model_obj
        func = model_obj._adjust_inferential_results_for_parameter_constraints

        # Set the constraints
        constraints = [0]
        dataframe.loc[0, :] = np.nan

        # Perform the tests
        func(constraints)
        for key in dataframe.columns[:-1]:
            new_attribute = getattr(self.model_obj, key)
            self.assertTrue(np.isnan(new_attribute.iloc[0]))
            npt.assert_allclose(new_attribute.values[1:],
                                dataframe[key].values[1:])

        return None

    def test_store_generic_inference_results(self):
        """
        Ensure that we can correctly store the given variables that are common
        to all inferential procedures after model estimation.
        """
        # Set a random seed for reproducibility
        np.random.seed(0)
        # Create the data needed for the various inferential results
        data = np.random.uniform(low=0,
                                 high=1,
                                 size=(5, 2))

        # Create fake names for the needed results
        all_names = ["Elly", "Feras", "Sreeta", "Mustapha", "Tim"]
        self.model_obj.ind_var_names = all_names
        all_params = [data[:, 0]]
        # assert all_params.shape[0] == len(all_names)

        # Create a fake hessian and fake fisher info
        fake_hessian = np.diag(-4 * np.ones(data.shape[0]))
        fake_fisher = np.diag(2 * np.ones(data.shape[0]))
        cov_matrix = np.diag(0.25 * np.ones(data.shape[0]))

        # Create the dictionary that is needed for this function
        needed_dict = {"utility_coefs": data[:, 0],
                       "final_gradient": data[:, 1],
                       "final_hessian": fake_hessian,
                       "fisher_info": fake_fisher}

        # Alias the function being tested
        func = self.model_obj._store_generic_inference_results

        # Determine what attributes should be created
        expected_attributes = ["coefs",
                               "gradient",
                               "hessian",
                               "cov",
                               "params",
                               "standard_errors",
                               "tvalues",
                               "pvalues",
                               "fisher_information",
                               "robust_cov",
                               "robust_std_errs",
                               "robust_t_stats",
                               "robust_p_vals"]

        # Perform the tests
        func(needed_dict, all_params, all_names)
        for attr_name in expected_attributes:
            print(attr_name)
            self.assertTrue(hasattr(self.model_obj, attr_name))
            self.assertTrue(isinstance(getattr(self.model_obj, attr_name),
                                       (pd.Series, pd.DataFrame)))
        npt.assert_allclose(self.model_obj.coefs.values,
                            needed_dict["utility_coefs"])
        npt.assert_allclose(self.model_obj.params.values,
                            needed_dict["utility_coefs"])
        npt.assert_allclose(self.model_obj.gradient.values,
                            needed_dict["final_gradient"])
        npt.assert_allclose(self.model_obj.hessian.values, fake_hessian)
        npt.assert_allclose(self.model_obj.cov.values, cov_matrix)
        npt.assert_allclose(self.model_obj.fisher_information.values,
                            fake_fisher)
        args = [self.model_obj.standard_errors.values,
                0.5 * np.ones(needed_dict["utility_coefs"].shape[0])]
        npt.assert_allclose(*args)

        expected_t_stats = (needed_dict["utility_coefs"] / 0.5)
        args = [self.model_obj.tvalues.values, expected_t_stats]
        npt.assert_allclose(*args)

        expected_p_vals = 2 * scipy.stats.norm.sf(np.abs(expected_t_stats))
        npt.assert_allclose(self.model_obj.pvalues, expected_p_vals)

        args = [fake_hessian, fake_fisher]
        expected_robust_cov = choice_calcs.calc_asymptotic_covariance(*args)
        npt.assert_allclose(self.model_obj.robust_cov.values,
                            expected_robust_cov)

        npt.assert_allclose(self.model_obj.robust_std_errs.values,
                            np.sqrt(np.diag(expected_robust_cov)))

        robust_t_stats = data[:, 0] / self.model_obj.robust_std_errs.values
        npt.assert_allclose(self.model_obj.robust_t_stats, robust_t_stats)

        robust_p_vals = 2 * scipy.stats.norm.sf(np.abs(robust_t_stats))
        npt.assert_allclose(self.model_obj.robust_p_vals, robust_p_vals)

        return None

    def test_addition_of_mixing_variables_to_ind_vars(self):
        """
        Ensure that the mixing variables are added to the individual variables.
        """
        # Set the mixing variables
        self.model_obj.mixing_vars = ["Test", "of", "mixing", "addition"]
        self.model_obj.ind_var_names = ["Generic X"]

        # Note what the result should be
        expected_names = (self.model_obj.ind_var_names +
                          ["Sigma " + x for x in self.model_obj.mixing_vars])
        # Use the given function
        self.model_obj._add_mixing_variable_names_to_individual_vars()
        # Perform the test
        self.assertEqual(self.model_obj.ind_var_names, expected_names)

        return None

    def test_get_statsmodels_summary(self):
        """
        Ensure correct formatting and return of a statsmodels summary table.
        Note that we only explicitly check the numbers in the table of
        estimation results.
        """
        # Set the type of this model
        model_type = "Test Model Object"
        self.model_obj.model_type = model_type

        # Set the needed attributes
        self.model_obj.estimation_success = self.estimation_success
        self.model_obj.nobs = self.fitted_probs.shape[0]
        self.model_obj.df_model = 1
        self.model_obj.df_resid = self.model_obj.nobs - self.model_obj.df_model
        self.model_obj.rho_squared = self.rho_squared
        self.model_obj.rho_bar_squared = self.rho_bar_squared
        self.model_obj.llf = self.log_likelihood
        self.model_obj.null_log_likelihood = self.null_log_likelihood

        # Store the inferential results that will go into the table
        self.model_obj.coefs = pd.Series(self.fake_betas,
                                         index=self.fake_names["x"],
                                         name="coefs")
        self.model_obj.params = self.model_obj.coefs.copy()
        self.model_obj.params.name = "params"
        self.model_obj.bse = pd.Series(np.array([0.3]),
                                       index=self.fake_names["x"],
                                       name="standard_errors")
        self.model_obj.standard_errors = self.model_obj.bse.copy()
        self.model_obj.tvalues = self.model_obj.params / self.model_obj.bse
        self.model_obj.pvalues =\
            pd.Series(2 * scipy.stats.norm.sf(np.abs(self.model_obj.tvalues)),
                      index=self.fake_names["x"], name="p_values")

        # Alias the function that will be tested
        func = self.model_obj.get_statsmodels_summary

        # Try the various tests
        try:
            from statsmodels.iolib.summary import Summary
            # Handle the different ways of accessing the StringIO module in
            # different python versions.
            import sys
            if sys.version_info[0] < 3:
                from StringIO import StringIO
            else:
                from io import StringIO

            # Get the summary
            summary = func()
            self.assertIsInstance(summary, Summary)

            # Convert the two tables of the summary into pandas dataframes
            # table_1_buffer = StringIO(summary.tables[0].as_csv())
            table_2_buffer = StringIO(summary.tables[1].as_csv())
            table_2_df = pd.read_csv(table_2_buffer)

            # Figure out the numerical values that should be displayed in the
            # table that is shown to users.
            expected_values = np.array([self.model_obj.params.iat[0],
                                        self.model_obj.bse.iat[0],
                                        self.model_obj.tvalues.iat[0],
                                        self.model_obj.pvalues.iat[0]])
            # Note that the summary table rounds values to the third decimal
            # place.
            expected_values = np.round(expected_values, decimals=3)

            # Only look at the numerical values, (minus the confidence
            # intervals that are tested elsewhere).
            summary_vals = table_2_df.iloc[0, 1:-1].values.astype(np.float64)
            npt.assert_allclose(summary_vals, expected_values)

            return None

        except ImportError:
            return None
