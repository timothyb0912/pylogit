"""
Tests for the nested_logit.py file. These tests do not include tests of
the functions that perform the mathematical calculations necessary to estimate
the Nested Logit model.
"""
import warnings
import unittest
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix

import pylogit.nested_logit as nl


class NestedLogitTests(unittest.TestCase):
    """
    Tests of the `split_params` function, the `NestedLogit` model
    constructor, and the `fit_mle()` method.
    """

    def setUp(self):
        # Create the betas to be used during the tests
        self.fake_betas = np.array([0.3, -0.6, 0.2])
        # Create the fake nest coefficients to be used during the tests
        self.fake_nest_coefs = np.array([1, 0.5])
        # Create an array of all model parameters
        self.fake_all_params = np.concatenate((self.fake_nest_coefs,
                                               self.fake_betas))
        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two.
        # The nest memberships of these alternatives are given below.
        self.fake_rows_to_nests = csr_matrix(np.array([[0, 1],
                                                       [0, 1],
                                                       [1, 0],
                                                       [0, 1],
                                                       [1, 0]]))

        # Create a sparse matrix that maps the rows of the design matrix to the
        # observatins
        self.fake_rows_to_obs = csr_matrix(np.array([[1, 0],
                                                     [1, 0],
                                                     [1, 0],
                                                     [0, 1],
                                                     [0, 1]]))

        # Create the fake design matrix with columns denoting ASC_1, ASC_2, X
        self.fake_design = np.array([[1, 0, 1],
                                     [0, 1, 2],
                                     [0, 0, 3],
                                     [1, 0, 1.5],
                                     [0, 0, 3.5]])

        # Create fake versions of the needed arguments for the MNL constructor
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                     "alt_id": [1, 2, 3, 1, 3],
                                     "choice": [0, 1, 0, 0, 1],
                                     "x": range(5),
                                     "intercept": [1 for i in range(5)]})

        # Record the various column names
        self.alt_id_col = "alt_id"
        self.obs_id_col = "obs_id"
        self.choice_col = "choice"

        # Create a sparse matrix that maps the chosen rows of the design
        # matrix to the observatins
        self.fake_chosen_rows_to_obs = csr_matrix(np.array([[0, 0],
                                                            [1, 0],
                                                            [0, 0],
                                                            [0, 0],
                                                            [0, 1]]))

        # Create the index specification  and name dictionaryfor the model
        self.fake_specification = OrderedDict()
        self.fake_specification["intercept"] = [1, 2]
        self.fake_specification["x"] = [[1, 2, 3]]
        self.fake_names = OrderedDict()
        self.fake_names["intercept"] = ["ASC 1", "ASC 2"]
        self.fake_names["x"] = ["x (generic coefficient)"]

        # Create the nesting specification
        self.fake_nest_spec = OrderedDict()
        self.fake_nest_spec["Nest 1"] = [1, 2]
        self.fake_nest_spec["Nest 2"] = [3]

        return None

    def test_split_params(self):
        """
        Ensures that split_params returns a tuple of nest coefficients and
        index coefficients.
        """
        split_results = nl.split_params(self.fake_all_params,
                                        self.fake_rows_to_nests)

        # Check that the results of split_params are as expected
        self.assertIsInstance(split_results, tuple)
        self.assertEqual(len(split_results), 2)
        for item in split_results:
            self.assertIsInstance(item, np.ndarray)
            self.assertEqual(len(item.shape), 1)
        npt.assert_allclose(self.fake_nest_coefs, split_results[0])
        npt.assert_allclose(self.fake_betas, split_results[1])

        return None

    def test_missing_nest_spec_error_in_constructor(self):
        """
        Ensure that the Nested Logit model cannot be constructed without the
        `nest_spec` keyword argument being passed a value other than `None`.
        """
        # Bundle the arguments used to construct the nested logit model
        constructor_args = [self.fake_df,
                            self.alt_id_col,
                            self.obs_id_col,
                            self.choice_col,
                            self.fake_specification,
                            self.fake_names]

        self.assertRaises(ValueError, nl.NestedLogit, *constructor_args)

        return None

    def test_ridge_warning_in_fit_mle(self):
        """
        Ensure that a UserWarning is raised when one passes the ridge keyword
        argument to the `fit_mle` method of a Nested Logit model object.
        """
        # Bundle the arguments used to construct the nested logit model
        constructor_args = [self.fake_df,
                            self.alt_id_col,
                            self.obs_id_col,
                            self.choice_col,
                            self.fake_specification,
                            self.fake_names]
        # Bundle the kwargs for constructing the nested_logit_model
        constructor_kwargs = {"nest_spec": self.fake_nest_spec}

        # Create the mnl model object whose coefficients will be estimated.
        base_nl = nl.NestedLogit(*constructor_args, **constructor_kwargs)

        # Create a variable for the fit_mle function's kwargs.
        # The print_res = False arguments are to make sure strings aren't
        # printed to the console unnecessarily.
        fit_kwargs = {"ridge": 0.5,
                      "print_res": False}

        # Test to make sure that the ridge warning message is printed when
        # using the ridge keyword argument
        with warnings.catch_warnings(record=True) as w:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            base_nl.fit_mle(self.fake_all_params, **fit_kwargs)
            self.assertGreaterEqual(len(w), 1)
            self.assertIsInstance(w[0].category, type(UserWarning))
            self.assertIn(nl._ridge_warning_msg, str(w[0].message))

        return None

    def test_invalid_init_kwargs_error_in_fit_mle(self):
        """
        Ensures that a ValueError is raised when users try to use any other
        type of initial value input methods other than the `init_vals`
        argument of `fit_mle()`. This prevents people from expecting the use
        of outside intercept or shape parameters to work with the Nested Logit
        model.
        """
        # Bundle the arguments used to construct the nested logit model
        constructor_args = [self.fake_df,
                            self.alt_id_col,
                            self.obs_id_col,
                            self.choice_col,
                            self.fake_specification]

        # Bundle the kwargs for constructing the nested_logit_model
        constructor_kwargs = {"names": self.fake_names,
                              "nest_spec": self.fake_nest_spec}

        # Create the mnl model object whose coefficients will be estimated.
        base_nl = nl.NestedLogit(*constructor_args, **constructor_kwargs)

        # Create a variable for the arguments to the fit_mle function.
        # this mimics the arguments passed when trying to use the shape_param
        # or outside intercepts kwargs with fit_mle.
        fit_args = [None]

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
            self.assertRaises(ValueError, base_nl.fit_mle,
                              *fit_args, **kwargs)

        return None

    def test_invalid_init_vals_length_in_estimate(self):
        """
        Ensure that when _estimate() is called, with an init_values argument
        that is of an incorrect length, a ValueError is raised.
        """
        # Specify the arguments that are needed for the _estimate() function
        estimate_args = [None,
                         self.fake_design,
                         self.fake_df[self.choice_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_nests,
                         self.fake_chosen_rows_to_obs,
                         None]

        # Test that the desired error is raised
        for i in [-1, 1]:
            init_values = np.arange(self.fake_betas.shape[0] + i)
            estimate_args[0] = init_values

            self.assertRaisesRegexp(ValueError,
                                    "values are of the wrong dimension",
                                    nl._estimate,
                                    *estimate_args)

        return None

    def test_identify_degenerate_nests(self):
        """
        Ensure that `identify_degenerate_nests` returns the correct list when
        using nest specifications that do and do not contain degenerate nests.
        """
        good_spec = OrderedDict()
        good_spec["Nest 1"] = [1, 2]
        good_spec["Nest 2"] = [3, 4]

        bad_spec = OrderedDict()
        bad_spec["Nest 1"] = [1]
        bad_spec["Nest 2"] = [2, 3]
        bad_spec["Nest 3"] = [4]

        # Alias the function being tested
        func = nl.identify_degenerate_nests

        # Test the function
        self.assertEqual([], func(good_spec))
        self.assertEqual([0, 2], func(bad_spec))

        return None

