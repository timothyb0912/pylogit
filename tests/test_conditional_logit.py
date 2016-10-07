"""
Tests for the conditional_logit.py file. These tests do not include tests of
the functions that perform the mathematical calculations necessary to estimate
the MNL model.
"""
import warnings
import unittest
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pandas as pd

import pylogit.conditional_logit as mnl


class HelperFuncTests(unittest.TestCase):
    """
    Defines the tests for the 'helper' functions for estimating the MNL model.
    """
    def setUp(self):
        # Set up the fake arguments
        self.fake_beta = np.arange(3)
        self.fake_args = ["foo", 1]
        self.fake_kwargs = {"fake_arg_1": "bar",
                            "fake_arg_2": 2,
                            "fake_arg_3": True}
        self.fake_design = np.arange(6).reshape((2, 3))
        self.fake_index = self.fake_design.dot(self.fake_beta)

    def test_split_param_vec(self):
        """
        Ensures that split_param_vec returns (None, None, index_coefs)
        when called from within conditional_logit.py.
        """
        # Store the results of split_param_vec()
        split_results = mnl.split_param_vec(self.fake_beta,
                                            return_all_types=False,
                                            *self.fake_args,
                                            **self.fake_kwargs)
        # Check for expected results.
        self.assertIsNone(split_results[0])
        self.assertIsNone(split_results[1])
        npt.assert_allclose(split_results[2], self.fake_beta)

        # Store the results of split_param_vec()
        split_results = mnl.split_param_vec(self.fake_beta,
                                            return_all_types=True,
                                            *self.fake_args,
                                            **self.fake_kwargs)
        # Check for expected results.
        self.assertIsNone(split_results[0])
        self.assertIsNone(split_results[1])
        self.assertIsNone(split_results[2])
        npt.assert_allclose(split_results[3], self.fake_beta)

        return None

    def test_mnl_utility_transform(self):
        """
        Ensures that mnl_utility_transform returns a 2D version of the 1D
        1D index array that is passed to it.
        """
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
        """
        Ensures that mnl_transform_deriv_c returns None.
        """
        derivative_results = mnl._mnl_transform_deriv_c(self.fake_index,
                                                        *self.fake_args,
                                                        **self.fake_kwargs)
        self.assertIsNone(derivative_results)

        return None

    def test_mnl_transform_deriv_alpha(self):
        """
        Ensures that mnl_transform_deriv_alpha returns None.
        """
        derivative_results = mnl._mnl_transform_deriv_alpha(self.fake_index,
                                                            *self.fake_args,
                                                            **self.fake_kwargs)
        self.assertIsNone(derivative_results)

        return None


class ChoiceObjectTests(unittest.TestCase):
    """
    Defines the tests for the MNL model object's `__init__` function and its
    other methods.
    """
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
        """
        Ensures that a ValueError is raised when the 'intercept_ref_pos' kwarg
        is passed to the MNL model constructor. This prevents people from
        expecting the use of outside intercept parameters to work with the MNL
        model.
        """
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
        """
        Ensures that a UserWarning is raised when the 'shape_ref_pos' or
        'shape_names' keyword arguments are passed to the MNL model
        constructor. This warns people against expecting the MNL to work with
        shape parameters, and alerts them to the fact they are using an MNL
        model when they might have been expecting to instantiate a different
        choice model.
        """
        # Create a variable for the standard arguments to this function.
        standard_args = [self.fake_df,
                         self.alt_id_col,
                         self.obs_id_col,
                         self.choice_col,
                         self.fake_specification]

        # Create a variable for the kwargs being passed to the constructor
        kwarg_map_1 = {"shape_ref_pos": 2}
        kwarg_map_2 = {"shape_names": OrderedDict([("x", ["foo"])])}

        # Test to ensure that the shape ignore message is printed when using
        # either of these two kwargs
        with warnings.catch_warnings(record=True) as context:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            for pos, bad_kwargs in enumerate([kwarg_map_1, kwarg_map_2]):
                # Create an MNL model object with the irrelevant kwargs.
                # This should trigger a UserWarning
                mnl_obj = mnl.MNL(*standard_args, **bad_kwargs)
                # Check that the warning has been created.
                self.assertEqual(len(context), pos + 1)
                self.assertIsInstance(context[-1].category, type(UserWarning))
                self.assertIn(mnl._shape_ignore_msg, str(context[-1].message))

        return None

    def test_outside_intercept_error_in_fit_mle(self):
        """
        Ensures that a ValueError is raised when users try to use any other
        type of initial value input methods other than the `init_vals`
        argument of `fit_mle()`. This prevents people from expecting the use
        of outside intercept or shape parameters to work with the MNL model.
        """
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

        # Test to ensure that the kwarg ignore message is printed when using
        # any of these three incorrect kwargs
        for kwargs in [kwarg_map_1, kwarg_map_2, kwarg_map_3]:
            self.assertRaises(ValueError, base_mnl.fit_mle,
                              *fit_args, **kwargs)

        return None

    def test_ridge_warning_in_fit_mle(self):
        """
        Ensure that a UserWarning is raised when one passes the ridge keyword
        argument to the `fit_mle` method of an MNL model object.
        """
        # Create a variable for the standard arguments to the MNL constructor.
        standard_args = [self.fake_df,
                         self.alt_id_col,
                         self.obs_id_col,
                         self.choice_col,
                         self.fake_specification]

        # Create the mnl model object whose coefficients will be estimated.
        base_mnl = mnl.MNL(*standard_args)

        # Create a variable for the fit_mle function's kwargs.
        # The print_res = False arguments are to make sure strings aren't
        # printed to the console unnecessarily.
        kwargs = {"ridge": 0.5,
                  "print_res": False}

        # Test to make sure that the ridge warning message is printed when
        # using the ridge keyword argument
        with warnings.catch_warnings(record=True) as w:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            base_mnl.fit_mle(self.fake_beta, **kwargs)
            self.assertGreaterEqual(len(w), 1)
            self.assertIsInstance(w[0].category, type(UserWarning))
            self.assertIn(mnl._ridge_warning_msg, str(w[0].message))

        return None

    def test_check_length_of_initial_values(self):
        """
        Ensure that a ValueError is raised when one passes an init_vals
        argument of the wrong length.
        """
        # Create a variable for the standard arguments to the MNL constructor.
        standard_args = [self.fake_df,
                         self.alt_id_col,
                         self.obs_id_col,
                         self.choice_col,
                         self.fake_specification]

        # Create the mnl model object whose coefficients will be estimated.
        base_mnl = mnl.MNL(*standard_args)

        # Create the EstimationObj
        mapping_res = base_mnl.get_mappings_for_fit()
        ridge = None
        zero_vector = np.zeros(1)
        split_params = mnl.split_param_vec
        mnl_estimator = mnl.MNLEstimator(base_mnl,
                                        mapping_res,
                                        ridge,
                                        zero_vector,
                                        split_params)

        # Alias the function to be checked
        func = mnl_estimator.check_length_of_initial_values

        for i in [2, 3]:
            init_vals = np.ones(i)
            self.assertRaises(ValueError, func, init_vals)

        self.assertIsNone(func(np.ones(1)))

        return None
