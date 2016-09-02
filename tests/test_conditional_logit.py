import sys
import warnings
import StringIO
import unittest
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import numpy.testing as npt

import pylogit.conditional_logit as mnl

class HelperFuncTests(unittest.TestCase):
    def setUp(self):
        # Set up the fake arguments
        self.fake_beta = np.arange(3)
        self.fake_args = ["foo", 1, None]
        self.fake_kwargs = {"fake_arg_1": "bar",
                            "fake_arg_2": 2,
                            "fake_arg_3": True}
        self.fake_design = np.arange(6).reshape((2,3))
        self.fake_index = self.fake_design.dot(self.fake_beta)

    def test_split_param_vec(self):
        # Store the results of split_param_vec()
        split_results = mnl.split_param_vec(self.fake_beta,
                                            *self.fake_args,
                                            **self.fake_kwargs)
        # Check for expected results.
        self.assertIsNone(split_results[0])
        self.assertIsNone(split_results[1])
        npt.assert_allclose(split_results[2], self.fake_beta)

        return None

    def test_mnl_utility_transform(self):
        # Get the results of _mnl_utiilty_transform()
        transform_results = mnl._mnl_utility_transform(self.fake_index,
                                                       *self.fake_args,
                                                       **self.fake_kwargs)

        # Check to make sure the results are as expected
        self.assertIsInstance(transform_results, np.ndarray)
        self.assertEqual(transform_results.shape, (2, 1))
        npt.assert_allclose(transform_results, self.fake_index[:, None])

        return None
        
    def test_mnl_transform_deriv_c(self):
        derivative_results = mnl._mnl_transform_deriv_c(self.fake_index,
                                                        *self.fake_args,
                                                        **self.fake_kwargs)
        self.assertIsNone(derivative_results)

        return None

    def test_mnl_transform_deriv_alpha(self):
        derivative_results = mnl._mnl_transform_deriv_alpha(self.fake_index,
                                                            *self.fake_args,
                                                            **self.fake_kwargs)
        self.assertIsNone(derivative_results)

        return None


class ChoiceObjectTests(unittest.TestCase):
    def setUp(self):
        # Create fake versions of the needed arguments for the MNL constructor
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 2, 2, 3, 3],
                                     "alt_id": [1, 2, 1, 2, 1, 2],
                                     "choice": [0, 1, 0, 1, 1, 0],
                                     "x": range(6)})
        self.fake_specification = OrderedDict()
        self.fake_specification["x"] = [[1, 2]]
        self.fake_names = OrderedDict()
        self.fake_names["x"] = ["x (generic coefficient)"]
        self.alt_id_col = "alt_id"
        self.obs_id_col = "obs_id"
        self.choice_col = "choice"
        self.fake_beta = np.array([1])

        return None

    def test_outside_intercept_error_in_constructor(self):
        # Create a variable for the standard arguments to this function.
        standard_args = [self.fake_df,
                         self.alt_id_col,
                         self.obs_id_col,
                         self.choice_col,
                         self.fake_specification]
        # Create a variable for the kwargs being passed to the constructor
        kwarg_map = {"intercept_ref_pos": 2}

        self.assertRaises(ValueError,
                          mnl.MNL,
                          *standard_args,
                          **kwarg_map)
        return None

    def test_shape_ignore_msg_in_constructor(self):
        # Create a variable for the standard arguments to this function.
        standard_args = [self.fake_df,
                         self.alt_id_col,
                         self.obs_id_col,
                         self.choice_col,
                         self.fake_specification]

        # Create a variable for the kwargs being passed to the constructor
        kwarg_map_1 = {"shape_ref_pos": 2}
        kwarg_map_2 = {"shape_names": OrderedDict([("x", ["foo"])])}

        # Test to make sure that the shape ignore message is printed when using
        # either of these two kwargs
        with warnings.catch_warnings(record=True) as w:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            mnl_1 = mnl.MNL(*standard_args, **kwarg_map_1)
            self.assertEqual(len(w), 1)
            self.assertIsInstance(w[-1].category, type(UserWarning))
            self.assertIn(mnl._shape_ignore_msg, str(w[-1].message))

            mnl_2 = mnl.MNL(*standard_args, **kwarg_map_2)
            self.assertEqual(len(w), 2)
            self.assertIsInstance(w[-1].category, type(UserWarning))
            self.assertIn(mnl._shape_ignore_msg, str(w[-1].message))

        return None

    def test_outside_intercept_error_in_fit_mle(self):
        # Create a variable for the standard arguments to the MNL constructor.
        standard_args = [self.fake_df,
                         self.alt_id_col,
                         self.obs_id_col,
                         self.choice_col,
                         self.fake_specification]

        # Create the mnl model object whose coefficients will be estimated.
        base_mnl = mnl.MNL(*standard_args)

        # Create a variable for the arguments to the fit_mle function.
        fit_args = [self.fake_beta]

        # Create variables for the incorrect kwargs.
        # The print_res = False arguments are to make sure strings aren't
        # printed to the console unnecessarily.
        kwarg_map_1 = {"init_shapes": np.array([1, 2]),
                       "print_res": False}
        kwarg_map_2 = {"init_intercepts": np.array([1]),
                       "print_res": False}
        kwarg_map_3 = {"init_coefs": np.array([1]),
                       "print_res": False}

        # Test to make sure that the kwarg ignore message is printed when using
        # any of these three incorrect kwargs
        for pos, kwargs in enumerate([kwarg_map_1, kwarg_map_2, kwarg_map_3]):
            self.assertRaises(ValueError, base_mnl.fit_mle, *fit_args, **kwargs)

        return None

    def test_ridge_warning_in_fit_mle(self):
        # Create a variable for the standard arguments to the MNL constructor.
        standard_args = [self.fake_df,
                         self.alt_id_col,
                         self.obs_id_col,
                         self.choice_col,
                         self.fake_specification]

        # Create the mnl model object whose coefficients will be estimated.
        base_mnl = mnl.MNL(*standard_args)

        # Create a variable for the arguments to the fit_mle function.
        fit_args = [self.fake_beta]

        # Create a variable for the kwargs being passed to the fit_mle function.
        # The print_res = False arguments are to make sure strings aren't
        # printed to the console unnecessarily.
        kwargs = {"ridge": 0.5,
                  "print_res": False}

        # Test to make sure that the ridge warning message is printed when using
        # the ridge keyword argument
        with warnings.catch_warnings(record=True) as w:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            base_mnl.fit_mle(self.fake_beta, **kwargs)
            self.assertGreaterEqual(len(w), 1)
            self.assertIsInstance(w[0].category, type(UserWarning))
            self.assertIn(mnl._ridge_warning_msg, str(w[0].message))
