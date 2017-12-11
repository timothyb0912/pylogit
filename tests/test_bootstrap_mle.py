"""
Tests for the bootstrap_mle.py file.
"""
import unittest
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix

import pylogit.bootstrap_mle as bmle
import pylogit.asym_logit as asym
from pylogit.conditional_logit import MNL


class HelperTests(unittest.TestCase):
    def test_get_model_abbrev(self):
        good_types = bmle.model_type_to_display_name.values()
        bad_type = "fakeType"

        # Alias the function to be tested.
        func = bmle.get_model_abbrev

        # Note the error message that should be raised.
        err_msg = "Model object has an unknown or incorrect model type."

        # Create a fake object class.
        class FakeModel(object):
            def __init__(self, fake_model_type):
                self.model_type = fake_model_type

        # Perform the tests that should pass.
        for model_type in good_types:
            current_obj = FakeModel(model_type)
            self.assertIn(func(current_obj), bmle.model_type_to_display_name)

        # Perform a test that should fail. Ensure the correct error is raised.
        current_obj = FakeModel(bad_type)
        self.assertRaisesRegexp(ValueError,
                                err_msg,
                                func,
                                current_obj)

        return None

    def test_get_model_creation_kwargs(self):
        # Create a fake object class.
        class FakeModel(object):
            def __init__(self, fake_model_type, fake_attr=None):
                self.model_type = fake_model_type
                if isinstance(fake_attr, dict):
                    for key in fake_attr:
                        setattr(self, key, fake_attr[key])

        # Create a fake model object for the tests
        fake_type = list(bmle.model_type_to_display_name.values())[0]
        fake_attr_dict = {"name_spec": OrderedDict([("intercept", "ASC 1")]),
                          "intercept_names": None,
                          "intercept_ref_position": None,
                          "shape_names": None,
                          "shape_ref_position": None,
                          "nest_spec": None,
                          "mixing_vars": None,
                          "mixing_id_col": None}
        fake_obj = FakeModel(fake_type, fake_attr=fake_attr_dict)

        # Alias the function being tested
        func = bmle.get_model_creation_kwargs

        # Perform the desired tests
        func_results = func(fake_obj)
        self.assertIsInstance(func_results, dict)

        expected_keys = ["model_type", "names", "intercept_names",
                         "intercept_ref_pos", "shape_names", "shape_ref_pos",
                         "nest_spec", "mixing_vars", "mixing_id_col"]
        self.assertTrue(all([x in func_results for x in expected_keys]))

        self.assertEqual(func_results["model_type"],
                         list(bmle.model_type_to_display_name.keys())[0])

        self.assertEqual(func_results["names"], fake_attr_dict["name_spec"])

        for key in expected_keys[2:]:
            self.assertIsNone(func_results[key])

        return None

    def test_extract_default_init_vals(self):
        # Set a desired number of parameters
        num_params = 9

        # Create a fake object class.
        class FakeModel(object):
            def __init__(self, fake_model_type, fake_attr=None):
                self.model_type = fake_model_type
                if isinstance(fake_attr, dict):
                    for key in fake_attr:
                        setattr(self, key, fake_attr[key])

        # Create an array of MNL parameter estimates
        mnl_point_array = np.arange(1, 6)
        mnl_param_names = ["ASC 1", "ASC 2", "ASC 3", "x1", "x2"]
        mnl_point_series =\
            pd.Series(mnl_point_array, index=mnl_param_names)

        # Create the fake model object
        fake_attr_dict = {"intercept_names": mnl_param_names[:3],
                          "ind_var_names": mnl_param_names[3:],
                          "mixing_vars": None}
        fake_obj = FakeModel(list(bmle.model_type_to_display_name.values())[0],
                             fake_attr=fake_attr_dict)

        # Create the array that we expect to be returned
        expected_array = np.concatenate([np.zeros(4), mnl_point_array], axis=0)

        # Alias the function being tested
        func = bmle.extract_default_init_vals

        # Test the function on a generic model with outside intercepts
        func_result = func(fake_obj, mnl_point_series, num_params)
        npt.assert_allclose(expected_array, func_result)

        # Test the function on a generic model without outside intercepts.
        new_fake_attrs = deepcopy(fake_attr_dict)
        new_fake_attrs["intercept_names"] = None
        new_fake_obj =\
            FakeModel(list(bmle.model_type_to_display_name.values())[0],
                      fake_attr=new_fake_attrs)
        new_results = func(new_fake_obj, mnl_point_series, num_params)
        npt.assert_allclose(expected_array, new_results)

        # Perform the desired tests on the Mixed Logit
        new_fake_attrs_2 = deepcopy(fake_attr_dict)
        new_fake_attrs_2["intercept_names"] = None
        new_fake_attrs_2["mixing_vars"] = ["ASC 1"]
        new_fake_obj =\
            FakeModel(list(bmle.model_type_to_display_name.values())[0],
                      fake_attr=new_fake_attrs_2)
        new_results_2 = func(new_fake_obj, mnl_point_series, num_params)

        new_expected_array =\
            np.concatenate([np.zeros(3), mnl_point_array, np.zeros(1)], axis=0)
        npt.assert_allclose(new_expected_array, new_results_2)

        # Perform the desired tests on the Asymmetric Logit
        num_params_3 = 8
        fake_attr_dict_3 = {"intercept_names": mnl_param_names[:3],
                            "ind_var_names": mnl_param_names[3:],
                            "mixing_vars": None,
                            "alt_IDs": np.arange(1, 5)}
        fake_obj_3 =\
            FakeModel(list(bmle.model_type_to_display_name.values())[1],
                      fake_attr=fake_attr_dict_3)
        new_results_3 = func(fake_obj_3, mnl_point_series, num_params_3)
        expected_array_3 =\
            np.concatenate([np.zeros(3),
                            mnl_point_array[:3],
                            mnl_point_array[3:] / np.log(4)], axis=0)
        npt.assert_allclose(expected_array_3, new_results_3)

        return None


class BootstrapEstimationTests(unittest.TestCase):
    """
    Make sure that we return expected results when using the various point
    estimation functions.
    """
    def setUp(self):
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

        # Calculate the 'natural' shape parameters
        self.natural_shapes = asym._convert_eta_to_c(self.fake_shapes,
                                                     self.fake_shape_ref_pos)

        # Create an array of all model parameters
        self.fake_all_params = np.concatenate((self.fake_shapes,
                                               self.fake_intercepts,
                                               self.fake_betas))

        # The mapping between rows and alternatives is given below.
        self.fake_rows_to_alts = csr_matrix(np.array([[1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1]]))

        # Get the mappping between rows and observations
        self.fake_rows_to_obs = csr_matrix(np.array([[1, 0, 0, 0, 0, 0],
                                                     [1, 0, 0, 0, 0, 0],
                                                     [1, 0, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0],
                                                     [0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 0, 1],
                                                     [0, 0, 0, 0, 0, 1],
                                                     [0, 0, 0, 0, 0, 1]]))

        # Create the fake design matrix with columns denoting X
        # The intercepts are not included because they are kept outside the
        # index in the scobit model.
        self.fake_design = np.array([[1],
                                     [2],
                                     [3],
                                     [1.5],
                                     [3.5],
                                     [0.78],
                                     [0.23],
                                     [1.04],
                                     [2.52],
                                     [1.49],
                                     [0.85],
                                     [1.37],
                                     [1.17],
                                     [2.03],
                                     [1.62],
                                     [1.94]])

        # Create the index array for this set of choice situations
        self.fake_index = self.fake_design.dot(self.fake_betas)

        # Create the needed dataframe for the Asymmetric Logit constructor
        nrows = self.fake_design.shape[0]
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2, 3, 3, 3,
                                                4, 4, 5, 5, 5, 6, 6, 6],
                                     "alt_id": [1, 2, 3, 1, 3, 1, 2, 3,
                                                2, 3, 1, 2, 3, 1, 2, 3],
                                     "choice": [0, 1, 0, 0, 1, 1, 0, 0,
                                                1, 0, 1, 0, 0, 0, 0, 1],
                                     "x": self.fake_design[:, 0],
                                     "intercept": np.ones(nrows)})

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

        # Initialize a basic Asymmetric Logit model whose coefficients will be
        # estimated.
        self.asym_model_obj = asym.MNAL(*self.constructor_args,
                                        **self.constructor_kwargs)

        self.asym_model_obj.coefs = pd.Series(self.fake_betas)
        self.asym_model_obj.intercepts =\
            pd.Series(self.fake_intercepts, index=self.fake_intercept_names)
        self.asym_model_obj.shapes =\
            pd.Series(self.fake_shapes, index=self.fake_shape_names)
        self.asym_model_obj.params =\
            pd.Series(np.concatenate([self.fake_shapes,
                                      self.fake_intercepts,
                                      self.fake_betas]),
                      index=(self.fake_shape_names +
                             self.fake_intercept_names +
                             self.fake_names["x"]))
        self.asym_model_obj.nests = None

        #####
        # Initialize a basic MNL model
        #####
        # Create the MNL specification and name dictionaries.
        self.mnl_spec, self.mnl_names = OrderedDict(), OrderedDict()
        self.mnl_spec["intercept"] = [1, 2]
        self.mnl_names["intercept"] = self.fake_intercept_names

        self.mnl_spec.update(self.fake_specification)
        self.mnl_names.update(self.fake_names)

        mnl_construct_args = self.constructor_args[:-1] + [self.mnl_spec]
        mnl_kwargs = {"names": self.mnl_names}
        self.mnl_model_obj = MNL(*mnl_construct_args, **mnl_kwargs)

        return None

    def test_get_mnl_point_est_using_mnl_model(self):
        """
        Ensure that the function returns the expected results when called using
        a Multinomial Logit Model object.
        """
        # Alias the function that is to be tested.
        func = bmle.get_mnl_point_est

        # Create the initial values for the mnl model object
        mnl_init_vals =\
            np.zeros(len(self.fake_intercept_names) +
                     sum([len(x) for x in self.fake_names.values()]))
        mnl_kwargs = {"ridge": 0.01,
                      "maxiter": 1200,
                      "method": "bfgs"}

        # Use arg list 1 to test the function using a regular MNL model
        arg_list_1 = [self.mnl_model_obj, self.fake_df,
                      self.mnl_model_obj.obs_id_col, mnl_init_vals.size,
                      self.mnl_spec, self.mnl_names, mnl_init_vals, mnl_kwargs]
        # Use arg list 2 to test the function when all arguments are not given.
        arg_list_2 = [self.mnl_model_obj, self.fake_df,
                      self.mnl_model_obj.obs_id_col, mnl_init_vals.size,
                      self.mnl_spec, self.mnl_names, None, None]
        # Combine all the argument lists for each series of tests.
        total_arg_lists = [arg_list_1, arg_list_2]
        # Iterate through the various sets of arguments to be tested
        for arg_list in total_arg_lists:
            # Get the function results
            point_result, obj_result = func(*arg_list)
            # Perform the desired tests
            self.assertIsInstance(point_result, dict)
            self.assertIn("x", point_result)
            self.assertIsInstance(point_result["x"], np.ndarray)
            self.assertEqual(point_result["x"].size, mnl_init_vals.size)
            self.assertIsInstance(obj_result, MNL)
        return None

    def test_get_mnl_point_est_using_asym_model(self):
        """
        Ensure that the function returns the expected results when called using
        a Multinomial Asymmetric Logit Model object.
        """
        # Alias the function that is to be tested.
        func = bmle.get_mnl_point_est

        # Create the initial values for the mnl model object
        mnl_init_vals =\
            np.zeros(len(self.fake_intercept_names) +
                     sum([len(x) for x in self.fake_names.values()]))
        mnl_kwargs = {"ridge": 0.01,
                      "maxiter": 1200,
                      "method": "bfgs"}

        # Use arg list to test the function using an Asymmetric Logit model
        arg_list = [self.asym_model_obj, self.fake_df,
                    self.asym_model_obj.obs_id_col, mnl_init_vals.size,
                    self.mnl_spec, self.mnl_names, mnl_init_vals, mnl_kwargs]
        # Get the function results
        point_result, obj_result = func(*arg_list)
        # Perform the desired tests
        self.assertIsInstance(point_result, dict)
        self.assertIn("x", point_result)
        self.assertIsInstance(point_result["x"], np.ndarray)
        self.assertEqual(point_result["x"].size, mnl_init_vals.size)
        self.assertIsInstance(obj_result, MNL)
        return None

    def test_retrieve_point_est_using_asym_model(self):
        """
        Ensure that the function returns the expected results when called using
        a Multinomial Asymmetric Logit Model object.
        """
        # Alias the function that is to be tested.
        func = bmle.retrieve_point_est

        # Determine the number of parameters for this model.
        num_params = self.asym_model_obj.params.size

        # Create the initial values for the mnl model object
        mnl_init_vals =\
            np.zeros(len(self.fake_intercept_names) +
                     sum([len(x) for x in self.fake_names.values()]))
        mnl_kwargs = {"ridge": 0.01,
                      "maxiter": 1200,
                      "method": "bfgs"}

        # Use arg list to test the function using an Asymmetric Logit model
        arg_list = [self.asym_model_obj, self.fake_df,
                    self.asym_model_obj.obs_id_col, num_params,
                    self.mnl_spec, self.mnl_names, mnl_init_vals, mnl_kwargs]

        # Use extract_func to create the array of initial values.
        def extract_func(model_obj, mnl_series, num_est_params):
            intercept_names = model_obj.intercept_names
            intercepts =\
                mnl_series.loc[intercept_names].values
            index_coefs =\
                mnl_series.loc[~mnl_series.index.isin(intercept_names)].values
            remaining_num_params =\
                num_est_params - intercepts.size - index_coefs.size
            num_alts = np.unique(model_obj.alt_IDs).size
            init_vals =\
                np.concatenate([np.zeros(remaining_num_params, dtype=float),
                                intercepts,
                                index_coefs / np.log(num_alts)],
                               axis=0)
            return init_vals

        # Get the kwargs for retrieve_point_est
        kwargs = deepcopy(mnl_kwargs)
        kwargs["extract_init_vals"] = extract_func
        # Get the function results
        point_result = func(*arg_list, **kwargs)
        # Perform the desired tests
        self.assertIsInstance(point_result, dict)
        self.assertIn("x", point_result)
        self.assertIsInstance(point_result["x"], np.ndarray)
        self.assertEqual(point_result["x"].size, num_params)

        # Test the function with the default extraction func.
        kwargs["extract_init_vals"] = None
        point_result = func(*arg_list, **kwargs)
        self.assertIsInstance(point_result, dict)
        self.assertIn("x", point_result)
        self.assertIsInstance(point_result["x"], np.ndarray)
        self.assertEqual(point_result["x"].size, num_params)

        return None

    def test_retrieve_point_est_using_mnl_model(self):
        """
        Ensure that the function returns the expected results when called using
        a Multinomial Logit Model object.
        """
        # Alias the function that is to be tested.
        func = bmle.retrieve_point_est

        # Create the initial values for the mnl model object
        mnl_init_vals =\
            np.zeros(len(self.fake_intercept_names) +
                     sum([len(x) for x in self.fake_names.values()]))
        mnl_kwargs = {"ridge": 0.01,
                      "maxiter": 1200,
                      "method": "bfgs"}

        # Determine the number of parameters for this model.
        num_params = mnl_init_vals.size

        # Use arg list to test the function using an Asymmetric Logit model
        arg_list = [self.mnl_model_obj, self.fake_df,
                    self.mnl_model_obj.obs_id_col, num_params,
                    self.mnl_spec, self.mnl_names, mnl_init_vals, mnl_kwargs]

        # Get the kwargs for retrieve_point_est
        kwargs = deepcopy(mnl_kwargs)
        # Get the function results
        point_result = func(*arg_list, **kwargs)
        # Perform the desired tests
        self.assertIsInstance(point_result, dict)
        self.assertIn("x", point_result)
        self.assertIsInstance(point_result["x"], np.ndarray)
        self.assertEqual(point_result["x"].size, num_params)

        return None
