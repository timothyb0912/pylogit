"""
Tests for the nested_choice_calcs.py file.
"""
import unittest
import warnings
from collections import OrderedDict

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
        self.natural_nest_coefs = np.array([0.995, 0.5])
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

    def test_naturalize_nest_coefs(self):
        """
        Ensure that we return expected results when using
        naturalize_nest_coefs.
        """
        # Create a set of reparametrized nest coefficients
        orig_nest_coefs = np.array([-800, -5, -1, 0, 1, 5])

        # Calculate what the results should be
        expected_coefs = (1.0 + np.exp(-1 * orig_nest_coefs))**-1
        expected_coefs[0] = nlc.min_comp_value

        # Get and test the results of the naturalize_nest_coefs function
        function_results = nlc.naturalize_nest_coefs(orig_nest_coefs)

        self.assertIsInstance(function_results, np.ndarray)
        self.assertEqual(function_results.shape, expected_coefs.shape)
        npt.assert_allclose(function_results, expected_coefs)

        return None
        return None

    def test_calc_nested_gradient(self):
        """
        Ensure that we return the correct gradient when passing correct
        arguments to calc_nested_gradient(). For formulas used to
        'hand'-calculate the gradient, see page 34 of "Estimation of
        multinomial logit models in R : The mlogit Packages"
        """
        # Get the logit of the natural nest coefficients
        nest_coefs = np.log(self.natural_nest_coefs /
                            (1 - self.natural_nest_coefs))

        #####
        # Calculate what the gradient should be for the observations in the
        # test case.
        #####
        # Create the index array for each alternative
        index_array = self.fake_design.dot(self.fake_betas)
        # Create an array of long, natural nest parameters
        long_nests = self.fake_rows_to_nests.dot(self.natural_nest_coefs)
        # Exponentiate the index array
        exp_scaled_index = np.exp(index_array / long_nests)

        # Calculate the sum of exp_scaled_index by obs by nest
        # Note the resulting array will be num_obs by num_nests
        exp_scaled_index_2d = exp_scaled_index[:, None]
        interim_array = self.fake_rows_to_nests.multiply(exp_scaled_index_2d)
        nest_sum = self.fake_rows_to_obs.T.dot(interim_array)
        # Create a 1D array that notes the nest-sum for the given nest and
        # observation that corresponds to a given row
        long_nest_sums = self.fake_rows_to_obs.dot(nest_sum)
        long_nest_sums = (self.fake_rows_to_nests
                              .multiply(long_nest_sums)
                              .sum(axis=1))
        long_nest_sums = np.asarray(long_nest_sums).ravel()

        # Get the probability of each individual choosing each available
        # alternative, given the alternative's nest.
        prob_alt_given_nest = exp_scaled_index / long_nest_sums

        # Get the probability of each individual choosing a given nest
        # Note that this array will be num_obs by num_nests
        nest_probs_numerator = np.power(nest_sum,
                                        self.natural_nest_coefs[None, :])
        nest_probs_denominator = nest_probs_numerator.sum(axis=1)
        nest_probs = nest_probs_numerator / nest_probs_denominator

        # Get the "average" value of the design matrix, in the chosen nests for
        # each observation. Note that observation 1 chosen nest 1 and
        # observation 2 chose nest 2.
        prob_by_design = prob_alt_given_nest[:, None] * self.fake_design
        x_bar_obs_1_nest_1 = prob_by_design[0:2, :].sum(axis=0)
        x_bar_obs_1_nest_2 = prob_by_design[2, :]
        x_bar_array = np.concatenate([x_bar_obs_1_nest_1[None, :],
                                      x_bar_obs_1_nest_2[None, :]],
                                     axis=0)
        x_bar_obs_1 = (nest_probs[0, :][:, None] * x_bar_array)

        x_bar_obs_2_nest_1 = prob_by_design[3, :]
        x_bar_obs_2_nest_2 = prob_by_design[4, :]
        x_bar_array_2 = np.concatenate([x_bar_obs_2_nest_1[None, :],
                                        x_bar_obs_2_nest_2[None, :]],
                                       axis=0)
        x_bar_obs_2 = (nest_probs[1, :][:, None] * x_bar_array_2)

        index_bar_obs_1_nest_1 = (prob_alt_given_nest * index_array)[:2].sum()
        index_bar_obs_1_nest_2 = index_array[2]
        index_bar_obs_2_nest_1 = index_array[3]
        index_bar_obs_2_nest_2 = index_array[4]

        # Note that the order of the gradient will be nest coef 1, nest coef 2,
        # then the index coefficients.
        obs_1_gradient = np.zeros(self.fake_all_params.shape[0])
        obs_2_gradient = np.zeros(self.fake_all_params.shape[0])

        # Calculate the gradient for observation 1
        term_1 = index_array[1]
        term_2 = (self.natural_nest_coefs[0]**2 *
                  (1 - nest_probs[0, 0]) *
                  np.log(nest_sum[0, 0]))
        term_3 = ((1 - self.natural_nest_coefs[0] * (1 - nest_probs[0, 0])) *
                  index_bar_obs_1_nest_1)
        obs_1_gradient[0] = (-1 * self.natural_nest_coefs[0]**-2 *
                             (term_1 - term_2 - term_3))

        term_4 = nest_probs[0, 1] / self.natural_nest_coefs[1]
        term_5 = index_bar_obs_1_nest_2
        term_6 = self.natural_nest_coefs[1] * np.log(nest_sum[0, 1])
        obs_1_gradient[1] = term_4 * (term_5 - term_6)

        term_7 = 1.0 / self.natural_nest_coefs[0]
        term_8 = self.fake_design[1]
        term_9 = (1 - self.natural_nest_coefs[0]) * x_bar_obs_1_nest_1
        term_10 = x_bar_obs_1
        obs_1_gradient[2:] = term_7 * (term_8 - term_9) - term_10

        # Calculate the gradient for observation 2
        term_1 = index_array[4]
        term_2 = (self.natural_nest_coefs[1]**2 *
                  (1 - nest_probs[1, 1]) *
                  np.log(nest_sum[1, 1]))
        term_3 = ((1 - self.natural_nest_coefs[1] * (1 - nest_probs[1, 1])) *
                  index_bar_obs_2_nest_2)
        # Note the calculates above are for the chosen nest which is nest 2
        # for this observation
        obs_2_gradient[1] = (-1 * self.natural_nest_coefs[1]**-2 *
                             (term_1 - term_2 - term_3))

        term_4 = nest_probs[1, 0] / self.natural_nest_coefs[0]
        term_5 = index_bar_obs_2_nest_1
        term_6 = self.natural_nest_coefs[0] * np.log(nest_sum[1, 0])
        obs_2_gradient[0] = term_4 * (term_5 - term_6)

        term_7 = 1.0 / self.natural_nest_coefs[1]
        term_8 = self.fake_design[4]
        term_9 = (1 - self.natural_nest_coefs[1]) * x_bar_obs_2_nest_2
        term_10 = x_bar_obs_2
        obs_2_gradient[2:] = term_7 * (term_8 - term_9) - term_10

        # Calculate the overall gradient
        expected_gradient = obs_1_gradient + obs_2_gradient
        # Don't forget to account for the jacobian
        jacobian = self.natural_nest_coefs * (1.0 - self.natural_nest_coefs)
        expected_gradient[:2] *= jacobian

        # Get the arguments necessary for the nested gradient function
        args = [nest_coefs,
                self.fake_betas,
                self.fake_design,
                self.choice_array,
                self.fake_rows_to_obs,
                self.fake_rows_to_nests]

        # Alias the function being tested
        func = nlc.calc_nested_gradient

        # Get the function results
        func_results = func(*args)

        # Test the returned results
        self.assertIsInstance(func_results, np.ndarray)
        self.assertEqual(len(func_results.shape), 1)
        self.assertEqual(func_results.shape, expected_gradient.shape)
        npt.assert_allclose(func_results, expected_gradient)

        # Ensure the function works when using a ridge penalty
        # Note we have to create an adjusted array for penalization because we
        # have reparameterized the nest coefficients
        params_for_penalty = np.concatenate([(20 - nest_coefs),
                                             self.fake_betas], axis=0)
        ridge_penalty = 2 * self.ridge * params_for_penalty
        penalized_gradient = expected_gradient - ridge_penalty

        kwargs = {"ridge": self.ridge}
        new_func_results = func(*args, **kwargs)

        # Test the returned results
        self.assertIsInstance(new_func_results, np.ndarray)
        self.assertEqual(len(new_func_results.shape), 1)
        self.assertEqual(new_func_results.shape, penalized_gradient.shape)
        npt.assert_allclose(new_func_results, penalized_gradient)

        return None

    def test_prep_vectors_for_gradient(self):
        """
        Ensure that the dictionary returned by this function contains the
        desired arrays.
        """
        # Calculate the arrays that should be returned for our test case.
        # Create the index array for each alternative
        index_array = self.model_obj.design.dot(self.fake_betas)
        # Create an array of long, natural nest parameters
        long_nests = self.fake_rows_to_nests.dot(self.natural_nest_coefs)
        # Exponentiate the index array
        exp_scaled_index = np.exp(index_array / long_nests)

        # Calculate the sum of exp_scaled_index by obs by nest
        # Note the resulting array will be num_obs by num_nests
        exp_scaled_index_2d = exp_scaled_index[:, None]
        interim_array = self.fake_rows_to_nests.multiply(exp_scaled_index_2d)
        nest_sum = self.fake_rows_to_obs.T.dot(interim_array)
        # Create a 1D array that notes the nest-sum for the given nest and
        # observation that corresponds to a given row
        long_nest_sums = self.fake_rows_to_obs.dot(nest_sum)
        long_nest_sums = (self.fake_rows_to_nests
                              .multiply(long_nest_sums)
                              .sum(axis=1))
        long_nest_sums = np.asarray(long_nest_sums).ravel()

        # Get the probability of each individual choosing each available
        # alternative, given the alternative's nest.
        prob_alt_given_nest = exp_scaled_index / long_nest_sums

        # Get the probability of each individual choosing a given nest
        # Note that this array will be num_obs by num_nests
        nest_probs_numerator = np.power(nest_sum,
                                        self.natural_nest_coefs[None, :])
        nest_probs_denominator = nest_probs_numerator.sum(axis=1)
        nest_probs = nest_probs_numerator / nest_probs_denominator

        # Get the probability of each alternative being chosen
        args = [self.natural_nest_coefs,
                self.fake_betas,
                self.model_obj.design,
                self.fake_rows_to_obs,
                self.fake_rows_to_nests]
        kwargs = {"return_type": "long_probs"}
        long_probs = nlc.calc_nested_probs(*args, **kwargs)

        # Create an expected dictionary that containing the same keys and
        # hopefully the same falues ans the function results.
        expected_dict = {}
        expected_dict["long_nest_params"] = long_nests
        expected_dict["scaled_y"] = self.choice_array / long_nests
        long_chosen_nest = np.array([1, 1, 0, 0, 1])
        expected_dict["long_chosen_nest"] = long_chosen_nest
        obs_to_chosen_nests = np.array([[1, 0], [0, 1]])
        expected_dict["obs_to_chosen_nests"] = obs_to_chosen_nests
        expected_dict["prob_given_nest"] = prob_alt_given_nest
        expected_dict["nest_choice_probs"] = nest_probs
        expected_dict["ind_sums_per_nest"] = nest_sum.A
        expected_dict["long_probs"] = long_probs

        expected_dict["p_tilde_given_nest"] = (prob_alt_given_nest *
                                               long_chosen_nest /
                                               long_nests)

        # Alias the function being tested
        func = nlc.prep_vectors_for_gradient

        # Gather the necessary function arguments
        args = [self.natural_nest_coefs,
                self.fake_betas,
                self.model_obj.design,
                self.choice_array,
                self.fake_rows_to_obs,
                self.fake_rows_to_nests]
        function_results = func(*args)

        # Perform the desired tests
        for key in expected_dict:
            self.assertTrue(key in function_results)
            self.assertIsInstance(function_results[key], np.ndarray)
            npt.assert_allclose(function_results[key], expected_dict[key])

        return None

    def test_calc_bhhh_hessian_approximation(self):
        """
        Ensure that we return the correct BHHH matrix when passing correct
        arguments to calc_bhhh_hessian_approximation(). For formulas used to
        'hand'-calculate the gradient of each observation, see page 34 of
        "Estimation of multinomial logit models in R : The mlogit Packages"
        """
        # Get the logit of the natural nest coefficients
        nest_coefs = np.log(self.natural_nest_coefs /
                            (1 - self.natural_nest_coefs))

        #####
        # Calculate what the gradient should be for the observations in the
        # test case.
        #####
        # Create the index array for each alternative
        index_array = self.fake_design.dot(self.fake_betas)
        # Create an array of long, natural nest parameters
        long_nests = self.fake_rows_to_nests.dot(self.natural_nest_coefs)
        # Exponentiate the index array
        exp_scaled_index = np.exp(index_array / long_nests)

        # Calculate the sum of exp_scaled_index by obs by nest
        # Note the resulting array will be num_obs by num_nests
        exp_scaled_index_2d = exp_scaled_index[:, None]
        interim_array = self.fake_rows_to_nests.multiply(exp_scaled_index_2d)
        nest_sum = self.fake_rows_to_obs.T.dot(interim_array)
        # Create a 1D array that notes the nest-sum for the given nest and
        # observation that corresponds to a given row
        long_nest_sums = self.fake_rows_to_obs.dot(nest_sum)
        long_nest_sums = (self.fake_rows_to_nests
                              .multiply(long_nest_sums)
                              .sum(axis=1))
        long_nest_sums = np.asarray(long_nest_sums).ravel()

        # Get the probability of each individual choosing each available
        # alternative, given the alternative's nest.
        prob_alt_given_nest = exp_scaled_index / long_nest_sums

        # Get the probability of each individual choosing a given nest
        # Note that this array will be num_obs by num_nests
        nest_probs_numerator = np.power(nest_sum,
                                        self.natural_nest_coefs[None, :])
        nest_probs_denominator = nest_probs_numerator.sum(axis=1)
        nest_probs = nest_probs_numerator / nest_probs_denominator

        # Get the "average" value of the design matrix, in the chosen nests for
        # each observation. Note that observation 1 chosen nest 1 and
        # observation 2 chose nest 2.
        prob_by_design = prob_alt_given_nest[:, None] * self.fake_design
        x_bar_obs_1_nest_1 = prob_by_design[0:2, :].sum(axis=0)
        x_bar_obs_1_nest_2 = prob_by_design[2, :]
        x_bar_array = np.concatenate([x_bar_obs_1_nest_1[None, :],
                                      x_bar_obs_1_nest_2[None, :]],
                                     axis=0)
        x_bar_obs_1 = (nest_probs[0, :][:, None] * x_bar_array)

        x_bar_obs_2_nest_1 = prob_by_design[3, :]
        x_bar_obs_2_nest_2 = prob_by_design[4, :]
        x_bar_array_2 = np.concatenate([x_bar_obs_2_nest_1[None, :],
                                        x_bar_obs_2_nest_2[None, :]],
                                       axis=0)
        x_bar_obs_2 = (nest_probs[1, :][:, None] * x_bar_array_2)

        index_bar_obs_1_nest_1 = (prob_alt_given_nest * index_array)[:2].sum()
        index_bar_obs_1_nest_2 = index_array[2]
        index_bar_obs_2_nest_1 = index_array[3]
        index_bar_obs_2_nest_2 = index_array[4]

        # Note that the order of the gradient will be nest coef 1, nest coef 2,
        # then the index coefficients.
        obs_1_gradient = np.zeros(self.fake_all_params.shape[0])
        obs_2_gradient = np.zeros(self.fake_all_params.shape[0])

        # Calculate the gradient for observation 1
        term_1 = index_array[1]
        term_2 = (self.natural_nest_coefs[0]**2 *
                  (1 - nest_probs[0, 0]) *
                  np.log(nest_sum[0, 0]))
        term_3 = ((1 - self.natural_nest_coefs[0] * (1 - nest_probs[0, 0])) *
                  index_bar_obs_1_nest_1)
        obs_1_gradient[0] = (-1 * self.natural_nest_coefs[0]**-2 *
                             (term_1 - term_2 - term_3))

        term_4 = nest_probs[0, 1] / self.natural_nest_coefs[1]
        term_5 = index_bar_obs_1_nest_2
        term_6 = self.natural_nest_coefs[1] * np.log(nest_sum[0, 1])
        obs_1_gradient[1] = term_4 * (term_5 - term_6)

        term_7 = 1.0 / self.natural_nest_coefs[0]
        term_8 = self.fake_design[1]
        term_9 = (1 - self.natural_nest_coefs[0]) * x_bar_obs_1_nest_1
        term_10 = x_bar_obs_1
        obs_1_gradient[2:] = term_7 * (term_8 - term_9) - term_10

        # Calculate the gradient for observation 2
        term_1 = index_array[4]
        term_2 = (self.natural_nest_coefs[1]**2 *
                  (1 - nest_probs[1, 1]) *
                  np.log(nest_sum[1, 1]))
        term_3 = ((1 - self.natural_nest_coefs[1] * (1 - nest_probs[1, 1])) *
                  index_bar_obs_2_nest_2)
        # Note the calculates above are for the chosen nest which is nest 2
        # for this observation
        obs_2_gradient[1] = (-1 * self.natural_nest_coefs[1]**-2 *
                             (term_1 - term_2 - term_3))

        term_4 = nest_probs[1, 0] / self.natural_nest_coefs[0]
        term_5 = index_bar_obs_2_nest_1
        term_6 = self.natural_nest_coefs[0] * np.log(nest_sum[1, 0])
        obs_2_gradient[0] = term_4 * (term_5 - term_6)

        term_7 = 1.0 / self.natural_nest_coefs[1]
        term_8 = self.fake_design[4]
        term_9 = (1 - self.natural_nest_coefs[1]) * x_bar_obs_2_nest_2
        term_10 = x_bar_obs_2
        obs_2_gradient[2:] = term_7 * (term_8 - term_9) - term_10

        # Calculate the overall gradient
        stacked_gradient = np.concatenate([obs_1_gradient[None, :],
                                           obs_2_gradient[None, :]], axis=0)
        # Don't forget to account for the jacobian
        jacobian = self.natural_nest_coefs * (1.0 - self.natural_nest_coefs)
        stacked_gradient[:, :2] *= jacobian[None, :]

        # Calculate the BHHH matrix that we expect to be returned
        # Note the -1 is because the bhhh should approximate the hessian, and
        # the hessian should be negative (think downward opening parabola) in
        # order for the log-likelihood to achieve a maximum.
        expected_bhhh = -1 * (np.outer(stacked_gradient[0, :],
                                       stacked_gradient[0, :]) +
                              np.outer(stacked_gradient[1, :],
                                       stacked_gradient[1, :]))

        # Get the arguments necessary for the nested gradient function
        args = [nest_coefs,
                self.fake_betas,
                self.fake_design,
                self.choice_array,
                self.fake_rows_to_obs,
                self.fake_rows_to_nests]

        # Alias the function being tested
        func = nlc.calc_bhhh_hessian_approximation

        # Get the function results
        func_results = func(*args)

        # Test the returned results
        self.assertIsInstance(func_results, np.ndarray)
        self.assertEqual(len(func_results.shape), 2)
        self.assertEqual(func_results.shape, expected_bhhh.shape)
        npt.assert_allclose(func_results, expected_bhhh)

        # Ensure the function works when using a ridge penalty
        # Note we have to create an adjusted array for penalization because we
        # have reparameterized the nest coefficients
        ridge_penalty = 2 * self.ridge
        penalized_bhhh = expected_bhhh - ridge_penalty

        kwargs = {"ridge": self.ridge}
        new_func_results = func(*args, **kwargs)

        # Test the returned results
        self.assertIsInstance(new_func_results, np.ndarray)
        self.assertEqual(len(new_func_results.shape), 2)
        self.assertEqual(new_func_results.shape, penalized_bhhh.shape)
        npt.assert_allclose(new_func_results, penalized_bhhh)

        return None
