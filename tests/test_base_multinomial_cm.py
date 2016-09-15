"""
Tests for the base_multinomial_cm_v2.py file. These tests do not include tests of
the functions that perform the mathematical calculations necessary to estimate
the predict function.
"""
import warnings
import unittest
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import pylogit.base_multinomial_cm_v2 as base_cm

class InitializationTests(unittest.TestCase):
    """
    This suite of tests should ensure that the logic in the initialization
    process is correctly executed.
    """
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
                                   "shape_names": self.fake_shape_names}

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
        bad_specification = deepcopy(self.fake_specification)

        good_col = self.fake_specification.keys()[0]
        bad_specification[bad_specification_col] = bad_specification[good_col]

        # Create the list of needed arguments
        args = [bad_specification, self.fake_df]

        self.assertRaises(ValueError,
                          base_cm.ensure_specification_cols_are_in_dataframe,
                          *args)

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

