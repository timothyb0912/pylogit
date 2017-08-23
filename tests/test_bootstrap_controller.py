"""
Tests for the bootstrap_controller.py file.
"""
import unittest
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix

import pylogit.bootstrap_controller as bc
import pylogit.asym_logit as asym
from pylogit.conditional_logit import MNL


class BootstrapTests(unittest.TestCase):
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
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2, 3, 3, 3,
                                                4, 4, 5, 5, 5, 6, 6, 6],
                                     "alt_id": [1, 2, 3, 1, 3, 1, 2, 3,
                                                2, 3, 1, 2, 3, 1, 2, 3],
                                     "choice": [0, 1, 0, 0, 1, 1, 0, 0,
                                                1, 0, 1, 0, 0, 0, 0, 1],
                                     "x": self.fake_design[:, 0],
                                     "intercept":
                                        np.ones(self.fake_design.shape[0])})

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
                      index=self.fake_shape_names +
                            self.fake_intercept_names +
                            self.fake_names["x"])
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

    def test_get_param_names(self):
        # Alias the function being tested.
        func = bc.get_param_names

        # Get the function results
        func_results = func(self.asym_model_obj)

        # Get the expected results
        expected_results = self.asym_model_obj.params.index.tolist()

        # Test the function results
        self.assertIsInstance(func_results, list)
        self.assertEqual(func_results, expected_results)

        # Set the nest names and re-test the function.
        self.asym_model_obj.nest_names = ["No Nest"]
        expected_results_2 = self.asym_model_obj.nest_names + expected_results
        func_results_2 = func(self.asym_model_obj)
        self.assertIsInstance(func_results_2, list)
        self.assertEqual(func_results_2, expected_results_2)

        return None

    def test_boot_initialization(self):
        # Create the bootstrap object
        boot_obj =\
            bc.Boot(self.asym_model_obj, self.asym_model_obj.params.values)

        # Test the bootstrap object.
        self.assertIsInstance(boot_obj, bc.Boot)
        self.assertEqual(id(boot_obj.model_obj), id(self.asym_model_obj))
        self.assertEqual(self.asym_model_obj.params.index.tolist(),
                         boot_obj.mle_params.index.tolist())
        expected_attrs =\
            ["point_samples", "conf_intervals", "conf_alpha", "summary"]
        for current_attr in expected_attrs:
            self.assertTrue(hasattr(boot_obj, current_attr))
            self.assertIsNone(getattr(boot_obj, current_attr))

        return None

    def test_bootstrap_params(self):
        # Create the bootstrap object.
        boot_obj =\
            bc.Boot(self.asym_model_obj, self.asym_model_obj.params.values)

        # Determine the number of bootstrap samples that we wish to take
        num_samples = 3

        # Create the necessary keyword arguments.
        mnl_init_vals =\
            np.zeros(len(self.fake_intercept_names) +
                     sum([len(x) for x in self.fake_names.values()]))

        mnl_kwargs = {"ridge": 0.01,
                      "maxiter": 1200,
                      "method": "bfgs"}

        bootstrap_kwargs = {"mnl_obj": self.mnl_model_obj,
                            "mnl_init_vals": mnl_init_vals,
                            "mnl_fit_kwargs": mnl_kwargs,
                            "constrained_pos": [0],
                            "boot_seed": 1988}

        # Get the function results
        func_results =\
            boot_obj.bootstrap_params(num_samples,
                                      mnl_obj=self.mnl_model_obj,
                                      mnl_init_vals=mnl_init_vals,
                                      mnl_fit_kwargs=mnl_kwargs,
                                      constrained_pos=[0],
                                      boot_seed=1988)

        # Perform the requisite tests
        self.assertIsNone(func_results)
        self.assertIsInstance(boot_obj.point_samples, pd.DataFrame)
        self.assertEqual(boot_obj.point_samples.ndim, 2)

        expected_shape = (3, self.asym_model_obj.params.size)
        self.assertEqual(boot_obj.point_samples.shape, expected_shape)
        self.assertEqual(boot_obj.point_samples.iloc[:, 0].unique().size, 1)
        self.assertEqual(boot_obj.point_samples.iloc[:, 0].unique()[0], 0)
        self.assertTrue(boot_obj.point_samples.iloc[:, 1].unique().size > 1)
        return None
