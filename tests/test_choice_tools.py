"""
Tests for the choice_tools.py file.
"""
import unittest
import os
import warnings
from collections import OrderedDict
from copy import deepcopy

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
        """
        Ensure that appropriate errors are raised when get_dataframe_from_data
        receives incorrect arguments, and that the function returns the
        expected results when correct arguments are passed.
        """
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
        """
        Ensure that the appropriate errors are raised when arguments of
        incorrect type are passed to "check_argument_type".
        """
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
        """"
        Ensure that the function correctly returns None when the
        alternative_id_col is in the long format dataframe and that ValueErrors
        are raised when the column is not the long format dataframe.
        """
        self.assertIsNone(ct.ensure_alt_id_in_long_form(self.alt_id_col,
                                                        self.fake_df))
        bad_cols = ["foo", 23, None]
        for col in bad_cols:
            self.assertRaises(ValueError, ct.ensure_alt_id_in_long_form,
                              col, self.fake_df)

        return None

    def test_check_type_and_values_of_specification_dict(self):
        """
        Ensure that the various type and structure checks for the specification
        dictionary are working.
        """
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
        """
        Ensure that the checks of the keys and values of the name dictionary
        are working as expected.
        """
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

        for names, spec, error in [(bad_names_5, new_spec_1, TypeError),
                                   (bad_names_6, new_spec_2, ValueError),
                                   (bad_names_7, new_spec_2, ValueError)]:
            args[0], args[1] = names, spec
            self.assertRaises(error, test_func, *args)

        return None

    def test_create_design_matrix(self):
        """
        Ensure that create_design_matrix returns the correct numpy arrays for
        model estimation.
        """
        # Create a long format dataframe with variables of all types (generic,
        # alternative-specific, and subset specific)
        self.fake_df["y"] = np.array([12, 9, 0.90, 16, 4])
        self.fake_df["z"] = np.array([2, 6, 9, 10, 1])
        self.fake_df["m"] = np.array([2, 2, 2, 6, 6])

        # Amend the specification of 'x'
        self.fake_specification["x"] = "all_same"
        self.fake_names["x"] = "x (generic coefficient)"

        # Add the new variables to the specification and name dictionaries
        self.fake_specification["y"] = "all_diff"
        self.fake_names["y"] = ["y_alt_1", "y_alt_2", "y_alt_3"]

        self.fake_specification["z"] = [[1, 2], 3]
        self.fake_names["z"] = ["z_alts_1_2", "z_alt_3"]

        self.fake_specification["m"] = [1, 2]
        self.fake_names["m"] = ["m_alt_1", "m_alt_2"]

        # Create the numpy array that should be returned
        expected = np.array([[1, 12, 0, 0, 2, 0, 2, 0],
                             [2, 0, 9, 0, 6, 0, 0, 2],
                             [3, 0, 0, 0.9, 0, 9, 0, 0],
                             [1.5, 16, 0, 0, 10, 0, 6, 0],
                             [3.5, 0, 0, 4, 0, 1, 0, 0]])

        expected_names = ([self.fake_names["x"]] +
                          self.fake_names["y"] +
                          self.fake_names["z"] +
                          self.fake_names["m"])

        # Compare the expected array with the returned array
        func_results = ct.create_design_matrix(self.fake_df,
                                               self.fake_specification,
                                               self.alt_id_col,
                                               self.fake_names)
        func_design, func_names = func_results

        self.assertIsInstance(func_design, np.ndarray)
        self.assertEqual(func_design.shape, (5, 8))
        npt.assert_allclose(func_design, expected)
        self.assertEqual(expected_names, func_names)

        return None

    def test_ensure_all_columns_are_used(self):
        """
        Ensure appropriate warnings are raised when there are more /less
        variables in one's dataframe than are accounted for in one's function.
        """
        # Make sure that None is returned when there is no problem.
        num_vars_used = self.fake_df.columns.size
        self.assertIsNone(ct.ensure_all_columns_are_used(num_vars_used,
                                                         self.fake_df))

        # Test to ensure that a warning message is raised when using
        # a number of colums different from the number in the dataframe.
        with warnings.catch_warnings(record=True) as context:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            for pos, package in enumerate([(-1, "only"), (1, "more")]):
                i, msg = package
                num_vars_used = self.fake_df.columns.size + i
                ct.ensure_all_columns_are_used(num_vars_used, self.fake_df)
                # Check that the warning has been created.
                self.assertEqual(len(context), pos + 1)
                self.assertIsInstance(context[-1].category, type(UserWarning))
                self.assertIn(msg, str(context[-1].message))

        return None

    def test_check_dataframe_for_duplicate_records(self):
        """
        Ensure that ValueError is raised only when the passed dataframe has
        duplicate observation-id and alternative-id pairs.
        """
        # Alias the function that is to be tested
        func = ct.check_dataframe_for_duplicate_records
        # Ensure that the function returns None when given data that is okay.
        good_args = [self.obs_id_col, self.alt_id_col, self.fake_df]
        self.assertIsNone(func(*good_args))
        # Make sure a ValueError is raised when one has repeat obs-id and
        # alt-id pairs.
        bad_df = self.fake_df.copy()
        bad_df.loc[3, "obs_id"] = 1

        bad_args = deepcopy(good_args)
        bad_args[2] = bad_df

        self.assertRaises(ValueError, func, *bad_args)

        return None

    def test_ensure_num_chosen_alts_equals_num_obs(self):
        """
        Ensure that ValueError is raised only when the passed dataframe's
        number of choices does not equal the declared number of observations.
        """
        # Alias the function that is to be tested
        func = ct.ensure_num_chosen_alts_equals_num_obs
        # Ensure that the function returns None when given data that is okay.
        args = [self.obs_id_col, self.choice_col, self.fake_df]
        self.assertIsNone(func(*args))

        # Make sure a ValueError is raised when one has more or less choices
        # than observations
        # Too many choice
        bad_df_1 = self.fake_df.copy()
        bad_df_1.loc[0, "choice"] = 1

        # Too few choices
        bad_df_2 = self.fake_df.copy()
        bad_df_2.loc[1, "choice"] = 0

        for bad_df in [bad_df_1, bad_df_2]:
            args[2] = bad_df
            self.assertRaises(ValueError, func, *args)

        return None

    def test_check_type_and_values_of_alt_name_dict(self):
        """
        Ensure that a TypeError is raised when alt_name_dict is not an instance
        of a dictionary, and ensure that a ValueError is raised when the keys
        of alt_name_dict are not actually in the alternative ID column of the
        passed dataframe.
        """
        # Alias the function that is to be tested
        func = ct.check_type_and_values_of_alt_name_dict
        # Ensure that the function returns None when given data that is okay.
        alt_name_dict = {1: "alternative 1",
                         2: "alternative 2",
                         3: "alternative 3"}
        args = [alt_name_dict, self.alt_id_col, self.fake_df]
        self.assertIsNone(func(*args))

        # Test both ways that the function of interest can raise errors.
        # Use a data structure that is not a dictionary.
        bad_dict_1 = alt_name_dict.items()
        # Use keys in the dictionary that are not valid alternative IDs.
        # Our alternative IDs are ints, not strings.
        bad_dict_2 = {'1': "alternative 1",
                      '2': "alternative 2",
                      '3': "alternative 3"}

        for bad_dict, error in [(bad_dict_1, TypeError),
                                (bad_dict_2, ValueError)]:
            args[0] = bad_dict
            self.assertRaises(error, func, *args)

        return None

    def test_convert_long_to_wide(self):
        """
        Test the basic functionality of convert_long_to_wide, ensuring correct
        outputs when given correct inputs.
        """
        # Create a long format dataframe with variables of all types (generic,
        # alternative-specific, and subset specific)

        # Add the alternative specific variable
        self.fake_df["y"] = np.array([12, 9, 0.90, 16, 4])
        # Add the subset specific variable (it only exists for a subset of
        # alternatives, 1 and 2)
        self.fake_df["z"] = np.array([2, 6, 0, 10, 0])
        # Add the individual specific variables
        self.fake_df["m"] = np.array([2, 2, 2, 6, 6])

        # Construct the wide format dataframe by hand
        wide_data = OrderedDict()
        wide_data["obs_id"] = [1, 2]
        wide_data["choice"] = [2, 3]
        wide_data["availability_1"] = [1, 1]
        wide_data["availability_2"] = [1, 0]
        wide_data["availability_3"] = [1, 1]
        # Add the individual specific variables
        wide_data["m"] = [2, 6]
        # Add the alternataive specific variables
        wide_data["y_1"] = [12.0, 16.0]
        wide_data["y_2"] = [9.0, np.nan]
        wide_data["y_3"] = [0.9, 4.0]
        # Add the subset specific variables
        wide_data["z_1"] = [2.0, 10.0]
        wide_data["z_2"] = [6, np.nan]

        expected = pd.DataFrame(wide_data)

        # Ensure the function's result matches our expectations
        ind_vars = ["m"]
        alt_specific_vars = ["y"]
        subset_specific_vars = {"z": [1, 2]}
        args = [self.fake_df,
                ind_vars,
                alt_specific_vars,
                subset_specific_vars,
                self.obs_id_col,
                self.alt_id_col,
                self.choice_col]
        alt_name_dict = {x: str(x) for x in range(1, 4)}
        func_results = ct.convert_long_to_wide(*args)
        func_results_2 = ct.convert_long_to_wide(*args,
                                                 alt_name_dict=alt_name_dict)

        npt.assert_allclose(func_results.values, expected.values)
        npt.assert_allclose(func_results_2.values, expected.values)

        return True

    def test_convert_wide_to_long(self):
        """
        Test the basic functionality of convert_wide_to_long, ensuring correct
        outputs when given correct inputs.
        """
        # Create a long format dataframe with variables of all types (generic,
        # alternative-specific, and subset specific)

        # Add another observation
        new_x = (self.fake_df["x"].tolist() +
                 self.fake_df["x"].iloc[-2:].tolist())
        new_choice = (self.fake_df["choice"].tolist() + [1, 0])
        new_alt_id = (self.fake_df["alt_id"].tolist() + [1, 3])
        new_obs_id = (self.fake_df["obs_id"].tolist() + [3, 3])
        new_intercept = [1 for x in range(7)]

        new_df = pd.DataFrame({"x": new_x,
                               "choice": new_choice,
                               "alt_id": new_alt_id,
                               "obs_id": new_obs_id,
                               "intercept": new_intercept})

        # Add the alternative specific variable
        new_df["y"] = np.array([12, 9, 0.90, 16, 4, 16, 4])
        # Add the subset specific variable (it only exists for a subset of
        # alternatives, 1 and 2)
        new_df["z"] = np.array([2, 6, 0, 10, 0, 10, 0])
        # Add the individual specific variables
        new_df["m"] = np.array([2, 2, 2, 6, 6, 6, 6])

        # Construct the wide format dataframe by hand
        wide_data = OrderedDict()
        wide_data["obs_id"] = [1, 2, 3]
        wide_data["choice"] = [2, 3, 1]
        wide_data["availability_1"] = [1, 1, 1]
        wide_data["availability_2"] = [1, 0, 0]
        wide_data["availability_3"] = [1, 1, 1]
        # Add the individual specific variables
        wide_data["m"] = [2, 6, 6]
        wide_data['intercept'] = [1, 1, 1]
        # Add the alternataive specific variables
        wide_data["x_1"] = new_df.loc[[0, 3, 5], "x"].tolist()
        wide_data["x_2"] = [new_df.at[1, "x"], np.nan, np.nan]
        wide_data["x_3"] = new_df.loc[[2, 4, 6], "x"].tolist()
        wide_data["y_1"] = [12.0, 16.0, 16.0]
        wide_data["y_2"] = [9.0, np.nan, np.nan]
        wide_data["y_3"] = [0.9, 4.0, 4.0]
        # Add the subset specific variables
        wide_data["z_1"] = [2.0, 10.0, 10.0]
        wide_data["z_2"] = [6, np.nan, np.nan]

        wide_df = pd.DataFrame(wide_data)

        # Ensure the function's result matches our expectations
        ind_vars = ["m", 'intercept']
        alt_specific_vars = {"x": {1: "x_1",
                                   2: "x_2",
                                   3: "x_3"},
                             "y": {1: "y_1",
                                   2: "y_2",
                                   3: "y_3"},
                             "z": {1: "z_1",
                                   2: "z_2"}}
        availability_vars = {1: "availability_1",
                             2: "availability_2",
                             3: "availability_3"}
        args = [wide_df, ind_vars, alt_specific_vars, availability_vars,
                self.obs_id_col, self.choice_col, self.alt_id_col]
        func_results = ct.convert_wide_to_long(*args).loc[:, new_df.columns]

        npt.assert_allclose(func_results.values, new_df.values)

        return None

    def test_check_wide_data_for_blank_choices(self):
        """
        Ensure that a ValueError is raised when and only when a dataframe has
        null values in its choice column.
        """
        # Create the dataframes used for testing
        good_df = self.fake_df
        bad_df = self.fake_df.copy()
        bad_df.loc[0, "choice"] = np.nan

        # Alias the function that will be tested
        func = ct.check_wide_data_for_blank_choices

        self.assertIsNone(func("choice", good_df))
        self.assertRaises(ValueError, func, "choice", bad_df)

        return None

    def test_ensure_unique_obs_ids_in_wide_data(self):
        """
        Ensure that a ValueError is raised when and only when a wide-format
        dataframe has fewer unique observations than rows.
        """
        good_df = pd.DataFrame({"obs_id": [1, 2, 3],
                                "choice": [1, 1, 2]})
        bad_df = pd.DataFrame({"obs_id": [1, 2, 1],
                               "choice": [1, 1, 2]})

        func = ct.ensure_unique_obs_ids_in_wide_data

        self.assertIsNone(func("obs_id", good_df))
        self.assertRaises(ValueError, func, "obs_id", bad_df)

        return None

    def test_ensure_chosen_alternatives_are_in_user_alt_ids(self):
        """
        Ensure that a ValueError is raised when and only when a wide-format
        dataframe has fewer unique observations than rows.
        """
        good_df = pd.DataFrame({"obs_id": [1, 2, 3],
                                "choice": [1, 1, 2]})
        bad_df = pd.DataFrame({"obs_id": [1, 2, 3],
                               "choice": [1, 1, 4]})
        availability_vars = {1: "availability_1",
                             2: "availability_2",
                             3: "availability_3"}

        func = ct.ensure_chosen_alternatives_are_in_user_alt_ids

        self.assertIsNone(func("choice", good_df, availability_vars))
        self.assertRaises(ValueError,
                          func,
                          "choice",
                          bad_df,
                          availability_vars)

        return None

    def test_ensure_each_wide_obs_chose_an_available_alternative(self):
        """
        Ensure a ValueError is raised when an individual in a wide format
        dataframe chooses an observation that was not available to him/her.
        """
        good_df = pd.DataFrame({"obs_id": [1, 2, 3],
                                "choice": [1, 1, 2],
                                "availability_1": [1, 1, 1],
                                "availability_2": [1, 1, 1],
                                "availability_3": [1, 1, 0]})

        bad_df = pd.DataFrame({"obs_id": [1, 2, 3],
                               "choice": [1, 1, 3],
                               "availability_1": [1, 1, 1],
                               "availability_2": [1, 1, 1],
                               "availability_3": [1, 1, 0]})

        availability_vars = {1: "availability_1",
                             2: "availability_2",
                             3: "availability_3"}

        func = ct.ensure_each_wide_obs_chose_an_available_alternative

        good_results = func("obs_id", "choice", availability_vars, good_df)
        self.assertIsNone(good_results)
        self.assertRaises(ValueError,
                          func,
                          "obs_id",
                          "choice",
                          availability_vars,
                          bad_df)

        return None

    def test_ensure_all_wide_alt_ids_are_chosen(self):
        """
        Ensure that ValueError is raised when a user-specified alternative id
        is not observed in the choices made by users.
        """
        # Construct the wide format dataframe by hand
        wide_data = OrderedDict()
        wide_data["obs_id"] = [1, 2, 3]
        wide_data["choice"] = [2, 3, 1]
        wide_data["availability_1"] = [1, 1, 1]
        wide_data["availability_2"] = [1, 0, 0]
        wide_data["availability_3"] = [1, 1, 1]
        # Add the individual specific variables
        wide_data["m"] = [2, 6, 6]
        wide_data['intercept'] = [1, 1, 1]
        # Add the alternataive specific variables
        wide_data["x_1"] = [1.5, 2.5, 3.5]
        wide_data["x_2"] = [0.4, np.nan, np.nan]
        wide_data["x_3"] = [2, 0.6, 1.3]
        wide_data["y_1"] = [12.0, 16.0, 16.0]
        wide_data["y_2"] = [9.0, np.nan, np.nan]
        wide_data["y_3"] = [0.9, 4.0, 4.0]
        # Add the subset specific variables
        wide_data["z_1"] = [2.0, 10.0, 10.0]
        wide_data["z_2"] = [6, np.nan, np.nan]

        wide_df = pd.DataFrame(wide_data)

        # Create needed arguments
        ind_vars = ["m", 'intercept']
        alt_specific_vars = {"x": {1: "x_1",
                                   2: "x_2",
                                   3: "x_3"},
                             "y": {1: "y_1",
                                   2: "y_2",
                                   3: "y_3"},
                             "z": {1: "z_1",
                                   2: "z_2"}}
        bad_alt_specific_vars = {"x": {1: "x_1",
                                       2: "x_2",
                                       4: "x_3"},
                                 "y": {1: "y_1",
                                       2: "y_2",
                                       3: "y_3"},
                                 "z": {1: "z_1",
                                       2: "z_2"}}
        availability_vars = {1: "availability_1",
                             2: "availability_2",
                             3: "availability_3"}
        bad_availability_vars = {1: "availability_1",
                                 2: "availability_2",
                                 4: "availability_3"}
        good_args = ["choice", alt_specific_vars, availability_vars, wide_df]
        bad_args_1 = ["choice", bad_alt_specific_vars,
                      availability_vars, wide_df]
        bad_args_2 = ["choice", alt_specific_vars,
                      bad_availability_vars, wide_df]

        # Alias the function being tested
        func = ct.ensure_all_wide_alt_ids_are_chosen

        self.assertIsNone(func(*good_args))

        for args in [bad_args_1, bad_args_2]:
            self.assertRaises(ValueError, func, *args)

        return None
