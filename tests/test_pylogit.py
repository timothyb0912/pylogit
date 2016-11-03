"""
Tests for the user-facing choice model constructor.
"""
import unittest
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pandas as pd

import pylogit
import pylogit.display_names as display_names


# Get the dictionary that maps the model type to the names of the model that
# are stored on the model object itself.
model_type_to_display_name = display_names.model_type_to_display_name


class ConstructorTests(unittest.TestCase):
    """
    Contains the tests of the choice model construction function.
    """
    def setUp(self):
        """
        Create the input data needed to test the choice model constructor.
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
        self.fake_shapes = np.array([-1, 0, 1])

        # Create names for the intercept parameters
        self.fake_shape_names = ["Shape 1", "Shape 2", "Shape 3"]

        # Create a shape ref position (used in the Asymmetric Logit Model)
        self.fake_shape_ref_pos = 2

        # # Create an array of all model parameters
        # self.fake_all_params = np.concatenate((self.fake_shapes,
        #                                        self.fake_intercepts,
        #                                        self.fake_betas))

        # # The mapping between rows and alternatives is given below.
        # self.fake_rows_to_alts = csr_matrix(np.array([[1, 0, 0],
        #                                               [0, 1, 0],
        #                                               [0, 0, 1],
        #                                               [1, 0, 0],
        #                                               [0, 0, 1]]))

        # Create the fake design matrix with columns denoting X
        # The intercepts are not included because they are kept outside the
        # index in the uneven model.
        self.fake_design = np.array([[1],
                                     [2],
                                     [3],
                                     [1.5],
                                     [3.5]])

        # Create the index array for this set of choice situations
        self.fake_index = self.fake_design.dot(self.fake_betas)

        # Create the needed dataframe for the choice model constructor
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

        # Create the nesting specification
        self.fake_nest_spec = OrderedDict()
        self.fake_nest_spec["Nest 1"] = [1, 2]
        self.fake_nest_spec["Nest 2"] = [3]

        # Bundle the args and kwargs used to construct the models.
        # Note that "MNL" is used as a model_type placeholder, and it will be
        # replaced as needed by each model
        self.constructor_args = [self.fake_df,
                                 self.alt_id_col,
                                 self.obs_id_col,
                                 self.choice_col,
                                 self.fake_specification,
                                 "MNL"]

        # Create a variable for the kwargs being passed to the constructor
        self.constructor_kwargs = {"intercept_ref_pos":
                                   self.fake_intercept_ref_pos,
                                   "names": self.fake_names,
                                   "intercept_names":
                                   self.fake_intercept_names,
                                   "shape_names": self.fake_shape_names}

    def test_constructor(self):
        """
        Construct the various choice models and make sure the constructed
        object has the necessary attributes.
        """
        # Record the model types of all the models to be created
        all_model_types = model_type_to_display_name.keys()

        # Record the attribute / value pairs that are common to all models.
        common_attr_value_dict = {"data": self.fake_df,
                                  "name_spec": self.fake_names,
                                  "design": self.fake_design,
                                  "ind_var_names": self.fake_names["x"],
                                  "alt_id_col": self.alt_id_col,
                                  "obs_id_col": self.obs_id_col,
                                  "choice_col": self.choice_col,
                                  "specification": self.fake_specification,
                                  "alt_IDs": self.fake_df["alt_id"].values,
                                  "choices": self.fake_df["choice"].values}

        # Create a shape name dictionary to relate the various models to the
        # names of their shape parameters.
        shape_name_dict = {"MNL": None,
                           "Asym": self.fake_shape_names[:2],
                           "Cloglog": None,
                           "Scobit": self.fake_shape_names,
                           "Uneven": self.fake_shape_names,
                           "Nested Logit": None,
                           "Mixed Logit": None}

        # Create a shape reference position dictionary to relate the various
        # models to their shape reference positions.
        shape_ref_dict = {}
        for key in shape_name_dict:
            shape_ref_dict[key] = (None if key != "Asym" else
                                   self.fake_shape_ref_pos)

        # Create an intercept_names and intercept_ref_position dictionary to
        # relate the various models to their respective kwargs.
        intercept_names_dict = {}
        intercept_ref_dict = {}
        for key in shape_name_dict:
            if key in ["MNL", "Nested Logit", "Mixed Logit"]:
                intercept_names_dict[key] = None
                intercept_ref_dict[key] = None
            else:
                intercept_names_dict[key] = self.fake_intercept_names
                intercept_ref_dict[key] = self.fake_intercept_ref_pos

        # Create a nest_names dictionary to relate the various models to their
        # nest_name attributes
        nest_name_dict = {}
        nest_spec_dict = {}
        for key in shape_name_dict:
            if key != "Nested Logit":
                nest_name_dict[key] = None
                nest_spec_dict[key] = None
            else:
                nest_name_dict[key] = self.fake_nest_spec.keys()
                nest_spec_dict[key] = self.fake_nest_spec

        # Create dictionaries for the mixing_id_col, mixing_vars, and
        # mixing_pos attributes
        mixing_id_col_dict = {}
        mixing_vars_dict = {}
        mixing_pos_dict = {}

        for key in shape_name_dict:
            if key != "Mixed Logit":
                mixing_id_col_dict[key] = None
                mixing_vars_dict[key] = None
                mixing_pos_dict[key] = None
            else:
                mixing_id_col_dict[key] = self.obs_id_col
                mixing_vars_dict[key] = self.fake_names["x"]
                mixing_pos_dict[key] = [0]

        # Record the attribute / value pairs that vary across models
        varying_attr_value_dict = {"model_type": model_type_to_display_name,
                                   "intercept_names": intercept_names_dict,
                                   "intercept_ref_position":
                                       intercept_ref_dict,
                                   "shape_names": shape_name_dict,
                                   "shape_ref_position": shape_ref_dict,
                                   "nest_names": nest_name_dict,
                                   "nest_spec": nest_spec_dict,
                                   "mixing_id_col": mixing_id_col_dict,
                                   "mixing_vars": mixing_vars_dict,
                                   "mixing_pos": mixing_pos_dict}

        # Set up the keyword arguments that are needed for each of the model
        # types
        variable_kwargs = {}
        for model_name in all_model_types:
            variable_kwargs[model_name] = {}
            variable_kwargs[model_name]["intercept_names"] =\
                intercept_names_dict[model_name]
            variable_kwargs[model_name]["intercept_ref_pos"] =\
                intercept_ref_dict[model_name]
            variable_kwargs[model_name]["shape_ref_pos"] =\
                shape_ref_dict[model_name]
            variable_kwargs[model_name]["shape_names"] =\
                shape_name_dict[model_name]
            variable_kwargs[model_name]["nest_spec"] =\
                nest_spec_dict[model_name]
            variable_kwargs[model_name]["mixing_id_col"] =\
                mixing_id_col_dict[model_name]
            variable_kwargs[model_name]["mixing_vars"] =\
                mixing_vars_dict[model_name]

        # Execute the test for each model type
        for model_name in all_model_types:
            # Update the model type in the list of constructor args
            self.constructor_args[-1] = model_name

            # Use this specific model's keyword arguments
            self.constructor_kwargs.update(variable_kwargs[model_name])

            # Construct the model object
            model_obj = pylogit.create_choice_model(*self.constructor_args,
                                                    **self.constructor_kwargs)

            # Make sure that the constructor has all of the required attributes
            for attr in common_attr_value_dict:
                value = common_attr_value_dict[attr]
                if isinstance(value, pd.DataFrame):
                    self.assertTrue(value.equals(model_obj.data))
                elif isinstance(value, np.ndarray):
                    npt.assert_allclose(value,
                                        model_obj.__getattribute__(attr))
                else:
                    self.assertEqual(value,
                                     model_obj.__getattribute__(attr))

            for attr in varying_attr_value_dict:
                value = varying_attr_value_dict[attr][model_name]

                self.assertEqual(value,
                                 model_obj.__getattribute__(attr))

        return None

    def test_ensure_valid_model_type(self):
        """
        Ensure that the desired message is raised when an invalid type is
        passed, and that None is returned otherwise
        """
        # Note the "valid" type strings for our test
        test_types = ["bar", "foo", "Sreeta", "Feras"]
        # Note a set of invalid type strings for the test
        bad_types = ["Tim", "Sam"]

        # Alias the function to be tested
        func = pylogit.pylogit.ensure_valid_model_type

        # Make note of part of the error message that should be raised
        partial_error_msg = "The specified model_type was not valid."

        # Perform the requisite tests
        for good_example in test_types:
            self.assertIsNone(func(good_example, test_types))
        for bad_example in bad_types:
            self.assertRaisesRegexp(ValueError,
                                    partial_error_msg,
                                    func,
                                    bad_example,
                                    test_types)

        return None
