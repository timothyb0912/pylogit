"""
Tests for the nested_choice_calcs.py file.
"""
import unittest
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import numpy.testing as npt

import pylogit.nested_choice_calcs as nlc
import pylogit.nested_logit as nested_logit


# Use the following to always show the warnings
np.seterr(all='warn')
warnings.simplefilter("always")


class ComputationalSetUp(unittest.TestCase):
    """
    Defines the common setUp method used for the different type of tests.
    """
    def setUp(self):
        # Create the betas to be used during the tests
        self.fake_betas = np.array([0.3, -0.6, 0.2])
        # Create the fake nest coefficients to be used during the tests
        # Note that these are the 'natural' nest coefficients, i.e. the
        # inverse of the scale parameters for each nest. They should be less
        # than or equal to 1.
        self.natural_nest_coefs = np.array([1, 0.5])
        # Create an array of all model parameters
        self.fake_all_params = np.concatenate((self.natural_nest_coefs,
                                               self.fake_betas))
        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two.
        # The nest memberships of these alternatives are given below.
        self.fake_rows_to_nests = csr_matrix(np.array([[1, 0],
                                                       [1, 0],
                                                       [0, 1],
                                                       [1, 0],
                                                       [0, 1]]))

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

        # Store the choice array
        self.choice_array = self.fake_df[self.choice_col].values

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

        # Create a nested logit object
        args = [self.fake_df,
                self.alt_id_col,
                self.obs_id_col,
                self.choice_col,
                self.fake_specification]
        kwargs = {"names": self.fake_names,
                  "nest_spec": self.fake_nest_spec}
        self.model_obj = nested_logit.NestedLogit(*args, **kwargs)

        # Store a ridge parameter
        self.ridge = 0.5

        return None

    def test_2d_error_in_calc_nested_probs(self):
        """
        Ensure that a NotImplementedError is raised whenever calc_nested_probs
        is called with a 2D array of nest coefficients or index coefficients.
        """
        # Create a 2D array of index coefficients
        index_2d = np.concatenate([self.fake_betas[:, None],
                                   self.fake_betas[:, None]], axis=1)

        # Create a 2D array of nest coefficients
        nest_coef_2d = np.concatenate([self.natural_nest_coefs[:, None],
                                       self.natural_nest_coefs[:, None]],
                                      axis=1)

        # Alias the function being tested
        func = nlc.calc_nested_probs

        # Get the arguments needed for the function. These are not the
        # legitimate arguments, but we just need a set of arrays to get to the
        # first argument check. 
        args = [np.arange(5) for x in range(5)]

        # Note the error message that should be raised.
        msg = "Support for 2D index_coefs or nest_coefs not yet implemented."

        for pos, array_2d in enumerate([nest_coef_2d, index_2d]):
            # Set the argument to the given 2D array.
            args[pos] = array_2d

            # Ensure that the appropriate error is raised.
            self.assertRaisesRegexp(NotImplementedError,
                                    msg,
                                    func,
                                    *args)

            # Set the argument back to None.
            args[pos] = None

        return None

    def test_return_type_error_in_calc_nested_probs(self):
        """
        Ensure that a ValueError is raised if return_type is not one of a
        handful of accepted values.
        """
        # Alias the function being tested
        func = nlc.calc_nested_probs

        # Get the arguments needed for the function. These are not the
        # legitimate arguments, but we just need a set of arrays to get to the
        # second argument check.
        args = [np.arange(5) for x in range(5)]

        # Note the error message that should be raised.
        msg = "return_type must be one of the following values: "

        # Create the kwargs to be tested
        bad_return_types = ["foo", 5, None]

        # Perform the tests
        kwargs = {"return_type": "long_probs"}
        for return_string in bad_return_types:
            kwargs["return_type"] = return_string
            self.assertRaisesRegexp(ValueError,
                                    msg,
                                    func,
                                    *args,
                                    **kwargs)

        return None

    def test_return_type_mismatch_error_in_calc_nested_probs(self):
        """
        Ensure that a ValueError is raised if return_type includes chosen_probs
        but chosen_row_to_obs is None.
        """
        # Alias the function being tested
        func = nlc.calc_nested_probs

        # Get the arguments needed for the function. These are not the
        # legitimate arguments, but we just need a set of arrays to get to the
        # third argument check.
        args = [np.arange(5) for x in range(5)]

        # Note the error message that should be raised.
        msg = "chosen_row_to_obs is None AND return_type in"

        # Create the kwargs to be tested
        bad_return_types = ['chosen_probs', 'long_and_chosen_probs']

        # Perform the tests
        kwargs = {"return_type": "long_probs",
                  "chosen_row_to_obs": None}
        for return_string in bad_return_types:
            kwargs["return_type"] = return_string
            self.assertRaisesRegexp(ValueError,
                                    msg,
                                    func,
                                    *args,
                                    **kwargs)

        return None

    def test_calc_probabilities(self):
        """
        Ensure that the calc_probabilities function returns correct results
        when executed.
        """
        # Calculate the index values, i.e. the systematic utilities for each
        # person.
        index_array = self.model_obj.design.dot(self.fake_betas)

        # Scale the index array by the nest coefficients
        long_nests = self.fake_rows_to_nests.dot(self.natural_nest_coefs)
        scaled_index_array = index_array / long_nests

        # Exponentiate the scaled index array
        exp_scaled_index = np.exp(scaled_index_array)

        # Calculate the sum of the exponentiated scaled index values, per nest
        nest_1_sum = np.array([exp_scaled_index[[0, 1]].sum(),
                               exp_scaled_index[3]])
        nest_2_sum = np.array([exp_scaled_index[2], exp_scaled_index[4]])

        # Raise the nest sums to the power of the nest coefficients
        # There will be one element for each person.
        powered_nest_1_sum = nest_1_sum**self.natural_nest_coefs[0]
        powered_nest_2_sum = nest_2_sum**self.natural_nest_coefs[1]

        # Create 'long-format' versions of the nest sums and the powered nest
        # sums
        long_nest_sums = np.array([nest_1_sum[0],
                                   nest_1_sum[0],
                                   nest_2_sum[0],
                                   nest_1_sum[1],
                                   nest_2_sum[1]])

        long_powered_nest_sums = np.array([powered_nest_1_sum[0],
                                           powered_nest_1_sum[0],
                                           powered_nest_2_sum[0],
                                           powered_nest_1_sum[1],
                                           powered_nest_2_sum[1]])

        # Calculate the powered-denominators
        sum_powered_nests = powered_nest_1_sum + powered_nest_2_sum

        long_powered_denoms = self.fake_rows_to_obs.dot(sum_powered_nests)
        # print long_powered_denoms

        # Calculate the probability
        probs = ((exp_scaled_index / long_nest_sums) *
                 (long_powered_nest_sums / long_powered_denoms))

        # Isolate the chosen probabilities
        condition = self.fake_df[self.choice_col].values == 1
        expected_chosen_probs = probs[np.where(condition)]

        # Alias the function being tested
        func = nlc.calc_nested_probs

        # Gather the arguments needed for the function
        args = [self.natural_nest_coefs,
                self.fake_betas,
                self.model_obj.design,
                self.fake_rows_to_obs,
                self.fake_rows_to_nests]
        kwargs = {"return_type": "long_probs"}

        # Get and test the function results
        function_results = func(*args, **kwargs)
        self.assertIsInstance(function_results, np.ndarray)
        self.assertEqual(function_results.shape, probs.shape)
        npt.assert_allclose(function_results, probs)

        # Check the function results again, this time looking for chosen probs
        kwargs["return_type"] = "chosen_probs"
        kwargs["chosen_row_to_obs"] = self.fake_chosen_rows_to_obs
        function_results_2 = func(*args, **kwargs)
        self.assertIsInstance(function_results_2, np.ndarray)
        self.assertEqual(function_results_2.shape, expected_chosen_probs.shape)
        npt.assert_allclose(function_results_2, expected_chosen_probs)

        # Check the function result when we return long_and_chosen_probs
        kwargs["return_type"] = "long_and_chosen_probs"
        function_results_3 = func(*args, **kwargs)
        self.assertIsInstance(function_results_3, tuple)
        self.assertTrue([all(isinstance(x, np.ndarray)
                         for x in function_results_3)])
        self.assertEqual(function_results_3[0].shape,
                         expected_chosen_probs.shape)
        self.assertEqual(function_results_3[1].shape, probs.shape)
        npt.assert_allclose(function_results_3[0], expected_chosen_probs)
        npt.assert_allclose(function_results_3[1], probs)

        return None

    def test_calc_log_likelihood(self):
        """
        Ensure that calc_log_likelihood returns the expected results.
        """
        # Gather the arguments needed for the calc_probabilities function
        args = [self.natural_nest_coefs,
                self.fake_betas,
                self.model_obj.design,
                self.fake_rows_to_obs,
                self.fake_rows_to_nests]
        kwargs = {"return_type": "chosen_probs",
                  "chosen_row_to_obs": self.fake_chosen_rows_to_obs}
        chosen_prob_array = nlc.calc_nested_probs(*args, **kwargs)

        # Calculate the expected log-likelihood
        expected_log_likelihood = np.log(chosen_prob_array).sum()
        penalized_log_likelihood = (expected_log_likelihood -
                                    self.ridge *
                                    ((self.natural_nest_coefs - 1)**2).sum() -
                                    self.ridge *
                                    (self.fake_betas**2).sum())

        # Alias the function being tested
        func = nlc.calc_nested_log_likelihood

        # Gather the arguments for the function being tested
        likelihood_args = [self.natural_nest_coefs,
                           self.fake_betas,
                           self.model_obj.design,
                           self.fake_rows_to_obs,
                           self.fake_rows_to_nests,
                           self.choice_array]
        likelihood_kwargs = {"ridge": self.ridge}

        # Get and test the function results.
        function_results = func(*likelihood_args)
        self.assertAlmostEqual(expected_log_likelihood, function_results)

        # Repeat the test with the ridge penalty
        function_results_2 = func(*likelihood_args, **likelihood_kwargs)
        self.assertAlmostEqual(penalized_log_likelihood, function_results_2)

        return None
