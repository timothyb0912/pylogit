"""
Tests for the base_multinomial_cm_v2.py file. These tests do not include tests of
the functions that perform the mathematical calculations necessary to estimate
the predict function.
"""
import warnings
import unittest
import os
import pickle
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import pylogit.base_multinomial_cm_v2 as base_cm


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

        return None
            

    def test_specification_column_presence_in_data(self):
        """
        Ensure that the check for the presence of specification columns works.
        """
        # Create column headings that are not in the dataframe used for testing
        bad_specification_col = "foo"
        bad_spec_1 = deepcopy(self.fake_specification)

        good_col = self.fake_specification.keys()[0]
        bad_spec_1[bad_specification_col] = bad_spec_1[good_col]

        # Create a second bad specification dictionary by simply using a dict
        # instead of an OrderedDict.
        bad_spec_2 = dict.update(self.fake_specification)

        # Create the list of needed arguments
        for bad_specification in [bad_spec_1, bad_spec_2]:
            args = [bad_specification, self.fake_df]
            func = base_cm.ensure_specification_cols_are_in_dataframe

            self.assertRaises(ValueError, func, *args)

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
        for bad_ref in [None, 'turtle', -1, 3]:
            args[0] = bad_ref
            self.assertRaises(ValueError,
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

    def test_ensure_nest_spec_is_ordered_dict(self):
        """
        Ensures that ValueError is raised if nest_spec is not an OrderedDict
        """
        new_nest_spec = {"Nest_1": [1, 2],
                         "Nest_2": [3]}

        self.assertRaises(ValueError,
                          base_cm.ensure_nest_spec_is_ordered_dict,
                          new_nest_spec)

        return None

    def test_check_type_of_nest_spec_keys_and_values(self):
        """
        Ensures that ValueError is raised if the keys of nest_spec are not
        strings and if the values of nest_spec are not lists.
        """
        new_nest_spec_1 = {1: [1, 2],
                           "Nest_2": [3]}

        new_nest_spec_2 = {"Nest_1": (1, 2),
                           "Nest_2": (3,)}

        for bad_spec in [new_nest_spec_1, new_nest_spec_2]:
            self.assertRaises(ValueError,
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
        Ensures a ValueError is raised if the first element of param_list is
        not an ndarray and if each of the subsequent elements are not None or
        ndarrays.
        """
        bad_param_list_1 = ['foo', np.zeros(2)]
        bad_param_list_2 = [np.zeros(2), 'foo']
        good_param_list = [np.zeros(2), np.ones(2)]
        good_param_list_2 = [np.zeros(2), None]

        for param_list in [bad_param_list_1, bad_param_list_2]:
            self.assertRaises(ValueError,
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
