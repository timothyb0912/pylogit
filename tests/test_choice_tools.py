"""
Tests for the choice_tools.py file.
"""
import unittest
import os
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import pylogit.choice_tools as ct
import pylogit.base_multinomial_cm_v2 as base_cm


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

        # Create the needed dataframe for the choice mdodel constructor
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                     "alt_id": [1, 2, 3, 1, 3],
                                     "choice": [0, 1, 0, 0, 1],
                                     "x": self.fake_design[:, 0],
                                     "intercept": [1 for i in range(5)]})

        # Record the various column names
        self.alt_id_col = "alt_id"
        self.obs_id_col = "obs_id"
        self.choice_col = "choice"

        # Create the index specification  and name dictionary for the model
        self.fake_specification = OrderedDict()
        self.fake_names = OrderedDict()
        self.fake_specification["x"] = [[1, 2, 3]]
        self.fake_names["x"] = ["x (generic coefficient)"]

        # Create a fake nest specification for the model
        self.fake_nest_spec = OrderedDict()
        self.fake_nest_spec["Nest 1"] = [1, 3]
        self.fake_nest_spec["Nest 2"] = [2]

         # Bundle args and kwargs used to construct the choice model.
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


class ArgumentValidationTests(GenericTestCase):
    def test_get_dataframe_from_data(self):
        # Create a test csv file.
        self.fake_df.to_csv("test_csv.csv", index=False)
        # Ensure that the dataframe is recovered
        func_df = ct.get_dataframe_from_data("test_csv.csv")
        self.assertIsInstance(func_df, pd.DataFrame)
        self.assertTrue((self.fake_df == func_df).all().all())
        # Remove the csv file
        os.remove("test_csv.csv")

        # Pass the dataframe and ensure that it is returned
        func_df = ct.get_dataframe_from_data(self.fake_df)
        self.assertIsInstance(func_df, pd.DataFrame)
        self.assertTrue((self.fake_df == func_df).all().all())

        # Test all the ways that a ValueError should or could be raised
        bad_args = [("test_json.json", ValueError),
                    (None, TypeError),
                    (77, TypeError)]
        for arg, error in bad_args:
            self.assertRaises(error, ct.get_dataframe_from_data, arg)

        return None

    def test_argument_type_check(self):
        # Isolate arguments that are correct, and ensure the function being
        # tested returns None.
        good_args = [self.fake_df, self.fake_specification]
        self.assertIsNone(ct.check_argument_type(*good_args))

        # Assemble a set of incorrect arguments and ensure the function raises
        # a ValueError.
        generic_dict = dict()
        generic_dict.update(self.fake_specification)
        bad_args = [[good_args[0], "foo"],
                    [good_args[0], generic_dict],
                    [self.fake_df.values, good_args[1]],
                    [False, good_args[1]],
                    [None, None]]

        for args in bad_args:
            self.assertRaises(TypeError, ct.check_argument_type, *args)

        return None

    def test_alt_id_col_inclusion_check(self):
        self.assertIsNone(ct.ensure_alt_id_in_long_form(self.alt_id_col,
                                                        self.fake_df))
        bad_cols = ["foo", 23, None]
        for col in bad_cols:
            self.assertRaises(ValueError, ct.ensure_alt_id_in_long_form,
                              col, self.fake_df)

        return None

    def test_check_type_and_values_of_specification_dict(self):
        # Ensure that a correct specification dict raises no errors.
        test_func = ct.check_type_and_values_of_specification_dict
        unique_alternatives = np.arange(1, 4)
        good_args = [self.fake_specification, unique_alternatives]
        self.assertIsNone(test_func(*good_args))

        # Create various bad specification dicts to make sure the function
        # raises the correct errors.
        bad_spec_1 = deepcopy(self.fake_specification)
        bad_spec_1["x"] = "incorrect_string"

        # Use a structure that is incorrect (group_items should only be ints
        # not lists)
        bad_spec_2 = deepcopy(self.fake_specification)
        bad_spec_2["x"] = [[1, 2], [[3]]]

        # Use an alternative that is not in the universal choice set
        bad_spec_3 = deepcopy(self.fake_specification)
        bad_spec_3["x"] = [[1, 2, 4]]

        bad_spec_4 = deepcopy(self.fake_specification)
        bad_spec_4["x"] = [1, 2, 4]

        # Use a completely wrong type
        bad_spec_5 = deepcopy(self.fake_specification)
        bad_spec_5["x"] = set([1, 2, 3])

        for bad_spec, error in [(bad_spec_1, ValueError),
                                (bad_spec_2, ValueError),
                                (bad_spec_3, ValueError),
                                (bad_spec_4, ValueError),
                                (bad_spec_5, TypeError)]:
            self.assertRaises(error, test_func, bad_spec, unique_alternatives)

        return None

    def test_check_keys_and_values_of_name_dictionary(self):
        # Ensure that a correct name dict raises no errors.
        test_func = ct.check_keys_and_values_of_name_dictionary
        num_alts = 3
        args = [self.fake_names, self.fake_specification, num_alts]
        self.assertIsNone(test_func(*args))

        # Create various bad specification dicts to make sure the function
        # raises the correct errors.
        bad_names_1 = deepcopy(self.fake_names)
        bad_names_1["y"] = "incorrect_string"

        # Use a completely wrong type
        bad_names_2 = deepcopy(self.fake_names)
        bad_names_2["x"] = set(["generic x"])

        # Use an incorrect number of elements
        bad_names_3 = deepcopy(self.fake_names)
        bad_names_3["x"] = ["generic x1", "generic x2"]

        # Use the wrong type for the name
        bad_names_4 = deepcopy(self.fake_names)
        bad_names_4["x"] = [23]

        for bad_names in [bad_names_1, 
                          bad_names_2,
                          bad_names_3,
                          bad_names_4]:
            args[0] = bad_names
            self.assertRaises(ValueError, test_func, *args)

        # Use two different specifications to test what could go wrong
        new_spec_1 = deepcopy(self.fake_specification)
        new_spec_1["x"] = "all_same"

        bad_names_5 = deepcopy(self.fake_names)
        bad_names_5["x"] = False

        new_spec_2 = deepcopy(self.fake_specification)
        new_spec_2["x"] = "all_diff"
        
        bad_names_6 = deepcopy(self.fake_names)
        bad_names_6["x"] = False

        bad_names_7 = deepcopy(self.fake_names)
        bad_names_7["x"] = ["foo", "bar"]

        for new_names, new_spec in [(bad_names_5, new_spec_1),
                                    (bad_names_6, new_spec_2),
                                    (bad_names_7, new_spec_2)]:
            args[0], args[1] = new_names, new_spec
            self.assertRaises(ValueError, test_func, *args)

        return None

