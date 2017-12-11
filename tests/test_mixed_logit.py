# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 15:59:25 2016

@author: timothyb0912
"""

import unittest
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import numpy.testing as npt
import pylogit.mixed_logit_calcs as mlc
import pylogit.mixed_logit as mixed_logit

try:
    # in Python 3 range returns an iterator instead of list
    # to maintain backwards compatibility use "old" version of range
    from past.builtins import xrange
except ImportError:
    pass


# Use the following to always show the warnings
np.seterr(all='warn')
warnings.simplefilter("always")


def temp_utility_transform(sys_utility_array, *args, **kwargs):
    """
    Parameters
    ----------
    sys_utility_array : numpy array.
        Should have 1D or 2D. Should have been created by the dot product of a
        design matrix and an array  of index coefficients.

    Returns
    -------
    2D numpy array.
        The returned array will contain a representation of the
        `sys_utility_array`. If `sys_utility_array` is 2D, then
        `sys_utility_array` will be returned unaltered. Else, the function will
        return `sys_utility_array[:, None]`.
    """
    # Return a 2D array of systematic utility values
    if len(sys_utility_array.shape) == 1:
        systematic_utilities = sys_utility_array[:, np.newaxis]
    else:
        systematic_utilities = sys_utility_array

    return systematic_utilities


class NormalDrawsTests(unittest.TestCase):

    def test_return_format(self):
        n_obs = 10
        n_draws = 5
        n_vars = 3
        random_draws = mlc.get_normal_draws(n_obs, n_draws, n_vars)

        self.assertIsInstance(random_draws, list)
        self.assertEqual(len(random_draws), n_vars)
        for draws in random_draws:
            self.assertIsInstance(draws, np.ndarray)
            self.assertAlmostEqual(draws.shape, (n_obs, n_draws))

        return None


class MixingNamesToPositions(unittest.TestCase):

    def test_convert_mixing_names_to_positions(self):
        fake_index_vars = ["foo", "bar", "cake", "cereal"]
        fake_mixing_vars = ["bar", "cereal"]
        args = (fake_mixing_vars, fake_index_vars)
        mix_pos = mlc.convert_mixing_names_to_positions(*args)

        self.assertIsInstance(mix_pos, list)
        self.assertEqual(len(mix_pos), len(fake_mixing_vars))
        for pos, idx_val in enumerate(mix_pos):
            current_var = fake_mixing_vars[pos]
            self.assertEqual(idx_val, fake_index_vars.index(current_var))
        return None


class MixedLogitCalculations(unittest.TestCase):

    # Note that for this set up, we will consider a situation with the
    # following parameters:
    # 3 Alternatives per individual
    # 2 Individuals
    # Individual 1 has 2 observed choice situations
    # Individual 2 has 1 observed choice situation
    # The true systematic utility depends on ASC_1, ASC_2, and a single X
    # The X coefficient is randomly distributed
    def setUp(self):
        # Fake random draws where Row 1 is for observation 1 and row 2 is
        # for observation 2. Column 1 is for draw 1 and column 2 is for draw 2
        self.fake_draws = np.array([[0.4, 0.8], [0.6, 0.2]])
        # Create the betas to be used during the tests
        self.fake_betas = np.array([0.3, -0.6, 0.2])
        self.fake_std = 1
        self.fake_betas_ext = np.concatenate((self.fake_betas,
                                              np.array([self.fake_std])),
                                             axis=0)

        # Create the fake design matrix with columns denoting ASC_1, ASC_2, X
        self.fake_design = np.array([[1, 0, 1],
                                     [0, 1, 2],
                                     [0, 0, 3],
                                     [1, 0, 1.5],
                                     [0, 1, 2.5],
                                     [0, 0, 3.5],
                                     [1, 0, 0.5],
                                     [0, 1, 1.0],
                                     [0, 0, 1.5]])
        # Record what positions in the design matrix are being mixed over
        self.mixing_pos = [2]

        # Create the arrays that specify the choice situation, individual id
        # and alternative ids
        self.situation_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.individual_ids = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2])
        self.alternative_ids = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        # Create a fake array of choices
        self.choice_array = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])

        # Create the 'rows_to_mixers' sparse array for this dataset
        # Denote the rows that correspond to observation 1 and observation 2
        self.obs_1_rows = np.ones(self.fake_design.shape[0])
        # Make sure the rows for observation 2 are given a zero in obs_1_rows
        self.obs_1_rows[-3:] = 0
        self.obs_2_rows = 1 - self.obs_1_rows
        # Create the row_to_mixers scipy.sparse matrix
        self.fake_rows_to_mixers = csr_matrix(self.obs_1_rows[:, None] ==
                                              np.array([1, 0])[None, :])
        # Create the rows_to_obs scipy.sparse matrix
        self.fake_rows_to_obs = csr_matrix(self.situation_ids[:, None] ==
                                           np.arange(1, 4)[None, :])
        # Create the rows_to_alts scipy.sparse matrix
        self.fake_rows_to_alts = csr_matrix(self.alternative_ids[:, None] ==
                                            np.arange(1, 4)[None, :])

        # Create the design matrix that we should see for draw 1 and draw 2
        arrays_to_join = (self.fake_design.copy(),
                          self.fake_design.copy()[:, -1][:, None])
        self.fake_design_draw_1 = np.concatenate(arrays_to_join, axis=1)
        self.fake_design_draw_2 = self.fake_design_draw_1.copy()

        # Multiply the 'random' coefficient draws by the corresponding variable
        self.fake_design_draw_1[:, -1] *= (self.obs_1_rows *
                                           self.fake_draws[0, 0] +
                                           self.obs_2_rows *
                                           self.fake_draws[1, 0])
        self.fake_design_draw_2[:, -1] *= (self.obs_1_rows *
                                           self.fake_draws[0, 1] +
                                           self.obs_2_rows *
                                           self.fake_draws[1, 1])
        extended_design_draw_1 = self.fake_design_draw_1[:, None, :]
        extended_design_draw_2 = self.fake_design_draw_2[:, None, :]
        self.fake_design_3d = np.concatenate((extended_design_draw_1,
                                              extended_design_draw_2),
                                             axis=1)

        # Create the fake systematic utility values
        self.sys_utilities_draw_1 = (self.fake_design_draw_1
                                         .dot(self.fake_betas_ext))
        self.sys_utilities_draw_2 = (self.fake_design_draw_2
                                         .dot(self.fake_betas_ext))

        #####
        # Calculate the probabilities of each alternatve in each choice
        # situation
        #####
        long_exp_draw_1 = np.exp(self.sys_utilities_draw_1)
        long_exp_draw_2 = np.exp(self.sys_utilities_draw_2)
        ind_exp_sums_draw_1 = self.fake_rows_to_obs.T.dot(long_exp_draw_1)
        ind_exp_sums_draw_2 = self.fake_rows_to_obs.T.dot(long_exp_draw_2)
        long_exp_sum_draw_1 = self.fake_rows_to_obs.dot(ind_exp_sums_draw_1)
        long_exp_sum_draw_2 = self.fake_rows_to_obs.dot(ind_exp_sums_draw_2)
        long_probs_draw_1 = long_exp_draw_1 / long_exp_sum_draw_1
        long_probs_draw_2 = long_exp_draw_2 / long_exp_sum_draw_2
        self.prob_array = np.concatenate((long_probs_draw_1[:, None],
                                          long_probs_draw_2[:, None]),
                                         axis=1)

        ###########
        # Create a mixed logit object for later use.
        ##########
        # Create a fake old long format dataframe for mixed logit model object
        self.alt_id_column = "alt_id"
        self.situation_id_column = "situation_id"
        self.obs_id_column = "observation_id"
        self.choice_column = "choice"

        data = {"x": self.fake_design[:, 2],
                self.alt_id_column: self.alternative_ids,
                self.situation_id_column: self.situation_ids,
                self.obs_id_column: self.individual_ids,
                self.choice_column: self.choice_array}
        self.fake_old_df = pd.DataFrame(data)
        self.fake_old_df["intercept"] = 1

        # Create a fake specification
        self.fake_spec = OrderedDict()
        self.fake_names = OrderedDict()

        self.fake_spec["intercept"] = [1, 2]
        self.fake_names["intercept"] = ["ASC 1", "ASC 2"]

        self.fake_spec["x"] = [[1, 2, 3]]
        self.fake_names["x"] = ["beta_x"]

        # Specify the mixing variable
        self.fake_mixing_vars = ["beta_x"]

        # Create a fake version of a mixed logit model object
        args = [self.fake_old_df,
                self.alt_id_column,
                self.situation_id_column,
                self.choice_column,
                self.fake_spec]
        kwargs = {"names": self.fake_names,
                  "mixing_id_col": self.obs_id_column,
                  "mixing_vars": self.fake_mixing_vars}
        self.mixl_obj = mixed_logit.MixedLogit(*args, **kwargs)

        # Set all the necessary attributes for prediction:
        # design_3d, coefs, intercepts, shapes, nests, mixing_pos
        self.mixl_obj.design_3d = self.fake_design_3d
        self.mixl_obj.coefs = pd.Series(self.fake_betas_ext)
        self.mixl_obj.intercepts = None
        self.mixl_obj.shapes = None
        self.mixl_obj.nests = None

        # Create a mixed logit estimator object for testing.
        self.ridge = 0.5
        zero_vector = np.zeros(self.fake_betas_ext.shape[0])
        args = [self.mixl_obj,
                self.mixl_obj.get_mappings_for_fit(),
                self.ridge,
                zero_vector,
                mixed_logit.split_param_vec]

        self.estimator = mixed_logit.MixedEstimator(*args)

        return None

    def test_split_param_vec(self):
        """
        Ensures that split_param_vec returns (None, None, index_coefs)
        when called from within mixed_logit.py. Also ensures that the
        return_all_types keyword arguments work as expected.
        """
        # Store the results of split_param_vec()
        split_results = mixed_logit.split_param_vec(self.fake_betas,
                                                    return_all_types=False)
        # Check for expected results.
        self.assertIsNone(split_results[0])
        self.assertIsNone(split_results[1])
        npt.assert_allclose(split_results[2], self.fake_betas)

        # Store the results of split_param_vec()
        split_results = mixed_logit.split_param_vec(self.fake_betas,
                                                    return_all_types=True)
        # Check for expected results.
        self.assertIsNone(split_results[0])
        self.assertIsNone(split_results[1])
        self.assertIsNone(split_results[2])
        npt.assert_allclose(split_results[3], self.fake_betas)

        return None

    def test_mnl_utility_transform(self):
        """
        Ensure that the mnl_utility_transform works as expected, returning the
        input systematic utilities in a form with 2D arrays.
        """
        array_1 = np.arange(3)
        array_2 = np.arange(6).reshape((3, 2))

        for array in [array_1, array_2]:
            results = mixed_logit.mnl_utility_transform(array)
            self.assertEqual(len(results.shape), 2)
            self.assertEqual(results.shape,
                             (array.shape[0], min(len(array.shape), 2)))
            if len(array.shape) == 1:
                npt.assert_allclose(array, results[:, 0])
            else:
                npt.assert_allclose(array, results)

        return None

    def test_create_expanded_design_for_mixing(self):
        # Create the 3d design matrix using the mixed logit functions
        # Note the [2] denotes the fact that the column at position 2 of the
        # fake design matrix is being treated as having random coefficients
        args = [self.fake_design,
                [self.fake_draws],
                [2],
                self.fake_rows_to_mixers]
        actual_3d_design = mlc.create_expanded_design_for_mixing(*args)
        # Actually perform the tests
        npt.assert_allclose(actual_3d_design[:, 0, :], self.fake_design_draw_1)
        npt.assert_allclose(actual_3d_design[:, 1, :], self.fake_design_draw_2)

        # Ensre that a ValueError is raised if we execute
        # mlc.create_expanded_design_for_mixing with the wrong arguments.
        args[2] = [2, 3, 4]
        self.assertRaisesRegexp(ValueError,
                                "mixing_pos",
                                mlc.create_expanded_design_for_mixing,
                                *args)

        return None

    def test_check_length_of_initial_values(self):
        """
        Ensure that a ValueError is raised when one passes an init_vals
        argument of the wrong length.
        """
        # Alias the functions to be checked
        func = mixed_logit.check_length_of_init_values
        func_2 = self.estimator.check_length_of_initial_values

        for i in [-1, 1]:
            init_vals = np.ones(self.fake_design_3d.shape[2] + i)
            self.assertRaisesRegexp(ValueError,
                                    "wrong dimension",
                                    func,
                                    self.fake_design_3d,
                                    init_vals)

            self.assertRaisesRegexp(ValueError,
                                    "wrong dimension",
                                    func_2,
                                    init_vals)

        self.assertIsNone(func(self.fake_design_3d,
                               np.ones(self.fake_design_3d.shape[2])))
        self.assertIsNone(func_2(np.ones(self.fake_design_3d.shape[2])))

        return None

    def test_shape_ignore_msg_in_constructor(self):
        """
        Ensures that a UserWarning is raised when the 'shape_ref_pos' or
        'shape_names' keyword arguments are passed to the Mixed Logit model
        constructor. This warns people against expecting the MNL to work with
        shape parameters, and alerts them to the fact they are using an Mixed
        Logit model when they might have been expecting to instantiate a
        different choice model.
        """
        # Create a variable for the standard arguments to this function.
        fake_specification = OrderedDict()
        fake_specification["intercept"] = [1, 2]
        fake_specification["x"] = [[1, 2, 3]]

        fake_names = OrderedDict()
        fake_names["intercept"] = ["ASC 1", "ASC 2"]
        fake_names["x"] = ["Generic x"]

        fake_df = pd.DataFrame({"x": self.fake_design[:, 2],
                                "alt_id": self.alternative_ids,
                                "situation_id": self.situation_ids,
                                "obs_id": self.individual_ids,
                                "choice": self.choice_array})
        fake_df["intercept"] = 1

        standard_args = [fake_df,
                         "alt_id",
                         "situation_id",
                         "choice",
                         fake_specification]
        standard_kwargs = {"names": fake_names,
                           "mixing_id_col": "obs_id",
                           "mixing_vars": ["Generic x"]}

        # Create a variable for the kwargs being passed to the constructor
        kwarg_map_1 = deepcopy(standard_kwargs)
        kwarg_map_1["shape_ref_pos"] = 2

        kwarg_map_2 = deepcopy(standard_kwargs)
        kwarg_map_2["shape_names"] = OrderedDict([("x", ["foo"])])

        # Test to ensure that the shape ignore message is printed when using
        # either of these two kwargs
        with warnings.catch_warnings(record=True) as context:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            for pos, bad_kwargs in enumerate([kwarg_map_1, kwarg_map_2]):
                # Create an MNL model object with the irrelevant kwargs.
                # This should trigger a UserWarning
                mixl_obj = mixed_logit.MixedLogit(*standard_args, **bad_kwargs)
                # Check that the warning has been created.
                self.assertEqual(len(context), pos + 1)
                self.assertIsInstance(context[-1].category, type(UserWarning))
                self.assertIn(mixed_logit._shape_ignore_msg,
                              str(context[-1].message))

        return None

    def test_calc_choice_sequence_probs(self):
        # Get the array of probabilities for each alternative under each draw
        # of the random coefficients.
        fake_prob_array = self.prob_array

        # Calculate the average probability of correctly prediicting each
        # person's sequence of choices. Note the 1, 5, 6 are the locations of
        # the "ones" in the self.choice_array. Also, 1 and 5 are grouped
        # because individual 1 has two observed choice situations.
        ind_1_sequence_probs = (fake_prob_array[1, :] *
                                fake_prob_array[5, :]).mean()
        ind_2_sequence_probs = (fake_prob_array[6, :]).mean()
        fake_sequence_probs = np.array([ind_1_sequence_probs,
                                        ind_2_sequence_probs])

        # Calculate the actual, simulated sequence probabilities
        args = [fake_prob_array,
                self.choice_array,
                self.fake_rows_to_mixers,
                "all"]
        prob_results = mlc.calc_choice_sequence_probs(*args)
        actual_sequence_probs = prob_results[0]
        sequence_probs_given_draws = prob_results[1]

        # Perform the desired testing
        self.assertEqual(len(prob_results), 2)
        self.assertIsInstance(actual_sequence_probs, np.ndarray)
        self.assertIsInstance(sequence_probs_given_draws, np.ndarray)
        self.assertEqual(len(actual_sequence_probs.shape), 1)
        self.assertEqual(len(sequence_probs_given_draws.shape), 2)
        npt.assert_allclose(actual_sequence_probs, fake_sequence_probs)

        # Ensure that the approrpriate error is raised if we execute
        # calc_choice_sequence_probs() with incorrect arguments.
        args[-1] = "foo"
        self.assertRaisesRegexp(ValueError,
                                "return_type",
                                mlc.calc_choice_sequence_probs,
                                *args)

        return None

    def test_calc_mixed_log_likelihood(self):
        # Calculate the 'true' log-likelihood
        args_1 = (self.prob_array, self.choice_array, self.fake_rows_to_mixers)
        func_sequence_probs = mlc.calc_choice_sequence_probs(*args_1)
        true_log_likelihood = np.log(func_sequence_probs).sum()

        # Calculate the log-likelihood according to the function being tested
        args_2 = [self.fake_betas_ext,
                  self.fake_design_3d,
                  self.alternative_ids,
                  self.fake_rows_to_obs,
                  self.fake_rows_to_alts,
                  self.fake_rows_to_mixers,
                  self.choice_array,
                  temp_utility_transform]

        function_log_likelihood = mlc.calc_mixed_log_likelihood(*args_2)

        # Perform the required test. AmostEqual used to avoid any issues with
        # floating point representations of numbers.
        self.assertAlmostEqual(true_log_likelihood, function_log_likelihood)

        # Test the function with the ridge penalty
        args_2.append(self.ridge)
        true_log_like_w_ridge = (true_log_likelihood -
                                 self.ridge * (self.fake_betas_ext**2).sum())
        new_func_log_like = mlc.calc_mixed_log_likelihood(*args_2)
        self.assertAlmostEqual(true_log_like_w_ridge, new_func_log_like)

        # Use the convenience function for the mixed logit estimator to test
        # the same functionality
        convenience_func = self.estimator.convenience_calc_log_likelihood
        self.assertAlmostEqual(convenience_func(self.fake_betas_ext),
                               true_log_like_w_ridge)

        # Repeat the tests with observation weights
        weights = 2 * np.ones(self.fake_design_3d.shape[0])
        args_2.append(weights)
        weighted_ridge_log_like = (2 * true_log_likelihood -
                                   self.ridge * (self.fake_betas_ext**2).sum())
        new_func_log_like = mlc.calc_mixed_log_likelihood(*args_2)
        self.assertAlmostEqual(weighted_ridge_log_like, new_func_log_like)

        # Use the convenience function with the mixed logit estimator.
        self.estimator.weights = weights
        self.assertAlmostEqual(convenience_func(self.fake_betas_ext),
                               weighted_ridge_log_like)
        self.estimator.weights = None

        return None

    def test_calc_mixed_logit_gradient(self):
        # Get the simulated probabilities for each individual and get the
        # array of probabilities given the random draws
        # Calculate the actual, simulated sequence probabilities
        args = (self.prob_array,
                self.choice_array,
                self.fake_rows_to_mixers,
                "all")
        prob_results = mlc.calc_choice_sequence_probs(*args)
        simulated_probs = prob_results[0]
        sequence_probs_given_draws = prob_results[1]

        s_twidle = sequence_probs_given_draws / simulated_probs[:, None]
        long_s_twidle = self.fake_rows_to_mixers.dot(s_twidle)
        error_twidle = ((self.choice_array[:, None] -
                         self.prob_array) *
                        long_s_twidle)

        # Initialize the true gradient
        gradient = np.zeros(self.fake_design_3d.shape[2])

        # Calculate the true gradient in an inefficient but clearly correct way
        for i in xrange(self.fake_design.shape[0]):
            for d in xrange(self.prob_array.shape[1]):
                gradient += (error_twidle[i, d] *
                             self.fake_design_3d[i, d, :])
        gradient *= 1.0 / self.prob_array.shape[1]

        # Get the gradient from the function being tested
        args = [self.fake_betas_ext,
                self.fake_design_3d,
                self.alternative_ids,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.fake_rows_to_mixers,
                self.choice_array,
                temp_utility_transform]
        function_gradient = mlc.calc_mixed_logit_gradient(*args)

        # Perform the test.
        self.assertIsInstance(function_gradient, np.ndarray)
        self.assertEqual(len(function_gradient.shape), 1)
        self.assertEqual(function_gradient.shape[0],
                         self.fake_design_3d.shape[2])
        npt.assert_allclose(gradient, function_gradient)

        # Test the function with a ridge penalty
        ridge_penalty = 2 * self.ridge * self.fake_betas_ext
        new_gradient = gradient - ridge_penalty
        args.append(self.ridge)
        new_func_gradient = mlc.calc_mixed_logit_gradient(*args)
        npt.assert_allclose(new_gradient, new_func_gradient)

        # Use the convenience function for the mixed logit estimator to test
        # the same functionality
        convenience_func = self.estimator.convenience_calc_gradient
        npt.assert_allclose(new_gradient,
                            convenience_func(self.fake_betas_ext))

        # Repeat the tests with observation weights
        weights = 2 * np.ones(self.fake_design_3d.shape[0])
        args.append(weights)
        weighted_ridge_gradient = (2 * gradient - ridge_penalty)
        new_func_gradient = mlc.calc_mixed_logit_gradient(*args)
        npt.assert_allclose(weighted_ridge_gradient, new_func_gradient)

        # Use the convenience function with the mixed logit estimator.
        self.estimator.weights = weights
        npt.assert_allclose(convenience_func(self.fake_betas_ext),
                            weighted_ridge_gradient)
        self.estimator.weights = None

        return None

    def test_calc_bhhh_hessian_approximation_mixed_logit(self):
        # Get the simulated probabilities for each individual and get the
        # array of probabilities given the random draws
        # Calculate the actual, simulated sequence probabilities
        args = (self.prob_array,
                self.choice_array,
                self.fake_rows_to_mixers,
                "all")
        prob_results = mlc.calc_choice_sequence_probs(*args)
        simulated_probs = prob_results[0]
        sequence_probs_given_draws = prob_results[1]

        s_twidle = sequence_probs_given_draws / simulated_probs[:, None]
        long_s_twidle = self.fake_rows_to_mixers.dot(s_twidle)
        error_twidle = ((self.choice_array[:, None] -
                         self.prob_array) *
                        long_s_twidle)

        # Initialize the true gradient, with one row per individual
        gradient = np.zeros((simulated_probs.shape[0],
                             self.fake_design_3d.shape[2]))

        # Calculate the true gradient in an inefficient but clearly correct way
        for pos, i in enumerate(self.individual_ids):
            for d in xrange(self.prob_array.shape[1]):
                gradient[i - 1, :] += (error_twidle[pos, d] *
                                       self.fake_design_3d[pos, d, :])
        gradient *= 1.0 / self.prob_array.shape[1]

        # Calculate the bhhh matrix
        bhhh_matrix = np.zeros((self.fake_design_3d.shape[2],
                                self.fake_design_3d.shape[2]))
        for i in xrange(gradient.shape[0]):
            bhhh_matrix += np.outer(gradient[i, :], gradient[i, :])
        # Multiply by negative one to account for the fact that we're
        # approximating the Fisher Information Matrix
        bhhh_matrix *= -1

        # Get the bhhh matrix from the function being tested
        args = [self.fake_betas_ext,
                self.fake_design_3d,
                self.alternative_ids,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.fake_rows_to_mixers,
                self.choice_array,
                temp_utility_transform]
        function_bhhh = mlc.calc_bhhh_hessian_approximation_mixed_logit(*args)

        # Perform the test.
        self.assertIsInstance(function_bhhh, np.ndarray)
        self.assertEqual(len(function_bhhh.shape), 2)
        self.assertEqual(function_bhhh.shape[0],
                         self.fake_design_3d.shape[2])
        self.assertEqual(function_bhhh.shape[1],
                         self.fake_design_3d.shape[2])
        npt.assert_allclose(bhhh_matrix, function_bhhh)

        # Perform the test with the ridge coefficient
        args.append(self.ridge)
        func_res2 = mlc.calc_bhhh_hessian_approximation_mixed_logit(*args)
        new_neg_bhhh = bhhh_matrix + 2 * self.ridge

        npt.assert_allclose(new_neg_bhhh, func_res2)

        # Perform the same test using the convenience function in the estimator
        # object.
        convenience_func = self.estimator.convenience_calc_hessian
        npt.assert_allclose(new_neg_bhhh,
                            convenience_func(self.fake_betas_ext))

        # Repeat the tests with observation weights
        weights = 2 * np.ones(self.fake_design_3d.shape[0])
        args.append(weights)
        weighted_ridge_bhhh = (2 * bhhh_matrix + 2 * self.ridge)
        new_func_bhhh = mlc.calc_bhhh_hessian_approximation_mixed_logit(*args)
        npt.assert_allclose(weighted_ridge_bhhh, new_func_bhhh)

        # Use the convenience function with the mixed logit estimator.
        self.estimator.weights = weights
        npt.assert_allclose(convenience_func(self.fake_betas_ext),
                            weighted_ridge_bhhh)
        self.estimator.weights = None

        return None

    def test_panel_predict(self):
        # Specify settings for the test (including seed for reproducibility)
        chosen_seed = 912
        num_test_draws = 3
        num_test_mixing_vars = 1

        # Create the new design matrix for testing
        # There should be two observations. One of which was in in the old
        # data, and one of which was not. One of the observations should have
        # multiple situations being predicted
        new_design = np.array([[1, 0, 1],
                               [0, 1, 2],
                               [0, 0, 1],
                               [1, 0, 0.75],
                               [0, 1, 0.37],
                               [0, 0, 1.5],
                               [1, 0, 2.3],
                               [0, 1, 1.2],
                               [0, 0, 1.1]])
        # Create new arrays that speficy the situation, observation, and
        # alternative ids.
        new_alt_ids = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        new_situation_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        new_obs_ids = np.array([1, 1, 1, 3, 3, 3, 3, 3, 3])
        new_choices = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])

        # Take the chosen number of draws from the normal distribution for each
        # unique observation who has choice situations being predicted.
        new_draw_list = mlc.get_normal_draws(len(np.unique(new_obs_ids)),
                                             num_test_draws,
                                             num_test_mixing_vars,
                                             seed=chosen_seed)

        # Create the new rows_to_mixers for the observations being predicted
        new_row_to_mixer = csr_matrix(new_obs_ids[:, None] ==
                                      np.array([1, 3])[None, :])
        # Create the new rows_to_obs for the observations being predicted
        new_rows_to_obs = csr_matrix(new_situation_ids[:, None] ==
                                     np.array([1, 2, 3])[None, :])
        # Create the new rows_to_alts for the observations being predicted
        new_rows_to_alts = csr_matrix(new_alt_ids[:, None] ==
                                      np.array([1, 2, 3])[None, :])

        # Create the new 3D design matrix
        new_design_3d = mlc.create_expanded_design_for_mixing(new_design,
                                                              new_draw_list,
                                                              self.mixing_pos,
                                                              new_row_to_mixer)

        # Get the array of kernel probabilities for each individual for each
        # choice situation
        prob_args = (self.fake_betas_ext,
                     new_design_3d,
                     new_alt_ids,
                     new_rows_to_obs,
                     new_rows_to_alts,
                     temp_utility_transform)
        prob_kwargs = {"return_long_probs": True}
        new_kernel_probs = mlc.general_calc_probabilities(*prob_args,
                                                          **prob_kwargs)

        # Initialize and calculate the weights needed for prediction with
        # "individualized" coefficient distributions. Should have shape
        # (new_row_to_mixer.shape[1], num_test_draws) == (2, 3)
        weights_per_ind_per_draw = (1.0 / num_test_draws *
                                    np.ones((new_row_to_mixer.shape[1],
                                             num_test_draws)))

        ##########
        # Create the 3D design matrix for the one individual whom we have
        # previously recorded choices for.
        ##########
        # Note rel_old_idx should be np.array([T, T, T, T, T, T, F, F, F])
        rel_old_idx = np.in1d(self.individual_ids, new_obs_ids)
        # rel_old_matrix_2d should have shape (6, 3)
        rel_old_matrix_2d = self.fake_design[rel_old_idx, :]
        rel_old_mixing_var = rel_old_matrix_2d[:, -1][:, None]
        # rel_old_matrix_ext_2d should have shape (6, 4)
        rel_old_matrix_ext_2d = np.concatenate((rel_old_matrix_2d,
                                                rel_old_mixing_var), axis=1)
        # rel_old_matrix_3d should have shape (6, 3, 4)
        rel_old_matrix_3d = np.tile(rel_old_matrix_ext_2d[:, None, :],
                                    (1, num_test_draws, 1))
        # random_vals should have shape(6, 3)
        random_vals = np.tile(new_draw_list[0][0, :][None, :],
                              (rel_old_matrix_3d.shape[0], 1))
        rel_old_matrix_3d[:, :, -1] *= random_vals

        ##########
        # Get the array of kernel probabilities for each individual for whom we
        # have previously recorded choices, for each previously recorded choice
        # situation.
        ##########
        # Get the identifying arrays for the relevant but old observations
        rel_old_alt_ids = self.alternative_ids[rel_old_idx]
        rel_old_rows_to_situations = csr_matrix(np.array([[1, 0],
                                                          [1, 0],
                                                          [1, 0],
                                                          [0, 1],
                                                          [0, 1],
                                                          [0, 1]]))
        rel_old_rows_to_alts = self.fake_rows_to_alts[rel_old_idx, :]
        rel_old_rows_to_mixers = csr_matrix(np.array([[1],
                                                      [1],
                                                      [1],
                                                      [1],
                                                      [1],
                                                      [1]]))

        # Calclulate the desired kernel probabilities for the previously
        # recorded choice situations of those individuals for whom we are
        # predicting future choice situations
        prob_args = (self.fake_betas_ext,
                     rel_old_matrix_3d,
                     rel_old_alt_ids,
                     rel_old_rows_to_situations,
                     rel_old_rows_to_alts,
                     temp_utility_transform)
        prob_kwargs = {"return_long_probs": True}
        rel_old_kernel_probs = mlc.general_calc_probabilities(*prob_args,
                                                              **prob_kwargs)

        ##########
        # Calculate the old sequence probabilities of all the individual's
        # for whom we have recorded observations and for whom we are predicting
        # future choice situations
        ##########
        rel_old_choices = self.choice_array[rel_old_idx]
        sequence_args = (rel_old_kernel_probs,
                         rel_old_choices,
                         rel_old_rows_to_mixers)
        seq_kwargs = {"return_type": 'all'}
        old_sequence_results = mlc.calc_choice_sequence_probs(*sequence_args,
                                                              **seq_kwargs)
        # Note sequence_probs_per_draw should have shape (1, 3)
        sequence_probs_per_draw = old_sequence_results[1]
        # Note rel_old_weights should have shape (1, 3)
        rel_old_weights = (sequence_probs_per_draw /
                           sequence_probs_per_draw.sum(axis=1)[:, None])

        ##########
        # Finish creating the weights for the individualized coefficient
        # distributions.
        ##########
        # Given that this is a test and we know which row corresponds to the
        # individual that has previously recorded observations, we hardcode the
        # assignment to the array of weights
        weights_per_ind_per_draw[0, :] = rel_old_weights

        # Create a 'long' format version of the weights array. This version
        # should have the same number of rows as the new kernel probs but the
        # same number of columns as the weights array (aka the number of draws)
        weights_per_draw = new_row_to_mixer.dot(weights_per_ind_per_draw)

        ##########
        # Calcluate the final probabilities per situation using the
        # individualized coefficients
        ##########
        true_pred_probs = (weights_per_draw * new_kernel_probs).sum(axis=1)
        # Calculate the probabilities per situation without individualized
        # coefficients
        wrong_pred_probs = new_kernel_probs.mean(axis=1)

        ##########
        # Calcluate the predicted probabilities using the function being tested
        ##########
        # Create a fake long format dataframe of the data to be predicted
        predictive_df = pd.DataFrame({"x": new_design[:, 2],
                                      self.alt_id_column: new_alt_ids,
                                      self.situation_id_column:
                                          new_situation_ids,
                                      self.obs_id_column: new_obs_ids,
                                      self.choice_column: new_choices})
        predictive_df["intercept"] = 1

        # Calculate the probabilities of each alternative being chosen in
        # each choice situation being predictied
        kwargs = {"choice_col": self.choice_column,
                  "seed": chosen_seed}
        results = self.mixl_obj.panel_predict(predictive_df,
                                              num_test_draws,
                                              **kwargs)
        function_chosen_probs, _ = results

        # Get the 'true' chosen probabilities
        true_chosen_probs = true_pred_probs[np.where(new_choices == 1)]

        # Repeat the prediction process without the choice column
        results = self.mixl_obj.panel_predict(predictive_df,
                                              num_test_draws,
                                              seed=chosen_seed)
        function_pred_probs = results

        # Ensure that the chosen probabilities, by themselves, can be returned
        kwargs = {"choice_col": self.choice_column,
                  "return_long_probs": False,
                  "seed": chosen_seed}
        second_chosen_probs = self.mixl_obj.panel_predict(predictive_df,
                                                          num_test_draws,
                                                          **kwargs)

        ##########
        # Perform the actual tests.
        ##########
        # Test for desired return types and equality of probability arrays
        self.assertIsInstance(function_pred_probs, np.ndarray)
        self.assertEqual(len(function_pred_probs.shape), 1)
        self.assertEqual(function_pred_probs.shape[0],
                         new_design.shape[0])
        assert not np.allclose(wrong_pred_probs, function_pred_probs)
        npt.assert_allclose(true_pred_probs, function_pred_probs)

        # Test for desired return types and equality of probability arrays
        self.assertIsInstance(function_chosen_probs, np.ndarray)
        self.assertEqual(len(function_chosen_probs.shape), 1)
        self.assertEqual(function_chosen_probs.shape[0],
                         true_chosen_probs.shape[0])
        npt.assert_allclose(true_chosen_probs, function_chosen_probs)

        # Ensure that all variations of the returned chosen probs are correct
        npt.assert_allclose(second_chosen_probs, function_chosen_probs)

        # Make sure the appropriate errors are raised when the input dataframe
        # is missing needed columns
        new_predictive_df = predictive_df.copy()
        for col in [self.alt_id_column,
                    self.situation_id_column,
                    self.obs_id_column]:
            # Delete the column from the new dataframe
            del new_predictive_df[col]

            # Ensure the panel_predict function raises an error
            self.assertRaisesRegexp(ValueError,
                                    "not in data.columns",
                                    self.mixl_obj.panel_predict,
                                    new_predictive_df,
                                    num_test_draws,
                                    seed=chosen_seed)

            # Puth the column back in new_predictive_df
            new_predictive_df[col] = predictive_df[col]

        return None

    def test_value_error_in_panel_predict_for_incorrect_args(self):
        """
        Ensure that a ValueError is raised when `return_long_probs  == False`
        and `choice_col is None`.
        """
        func = self.mixl_obj.panel_predict
        args = [None, 20]

        msg = "choice_col is None AND return_long_probs == False"
        self.assertRaisesRegexp(ValueError,
                                msg,
                                func,
                                *args,
                                return_long_probs=False,
                                choice_col=None)

        return None

    def test_calc_neg_log_likelihood_and_neg_gradient(self):
        """
        Ensure that the constrained_pos arguement works, and that we can
        correctly calculate the negative log-likelihood and negative gradient.
        """
        # Get the gradient from the function being tested
        args = [self.fake_betas_ext,
                self.fake_design_3d,
                self.alternative_ids,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.fake_rows_to_mixers,
                self.choice_array,
                temp_utility_transform]

        # Specify a constrained position argument
        constrained_pos = [0]

        # Get the actual gradient and actual log-likelihood
        original_gradient = mlc.calc_mixed_logit_gradient(*args)
        for idx in constrained_pos:
            original_gradient[idx] = 0
        actual_log_likelihood = mlc.calc_mixed_log_likelihood(*args)

        expected_neg_gradient = -1 * original_gradient
        expected_log_likelihood = -1 * actual_log_likelihood

        # Get the results from the function of interest
        args_2 = [x for x in args]
        args_2.append(constrained_pos)
        func_results = mlc.calc_neg_log_likelihood_and_neg_gradient(*args_2)
        func_neg_log_like, func_neg_gradient = func_results
        self.assertAlmostEqual(func_neg_log_like, expected_log_likelihood)
        npt.assert_allclose(func_neg_gradient, expected_neg_gradient)

        return None

    def test_outside_intercept_error_in_fit_mle(self):
        """
        Ensures that a ValueError is raised when users try to use any other
        type of initial value input methods other than the `init_vals`
        argument of `fit_mle()`. This prevents people from expecting the use
        of outside intercept or shape parameters to work with the Mixed Logit
        model.
        """
        # Create a variable for the arguments to the fit_mle function.
        fit_args = [self.fake_betas, 800]

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
            self.assertRaises(ValueError, self.mixl_obj.fit_mle,
                              *fit_args, **kwargs)

        return None

    def test_ridge_warning_in_fit_mle(self):
        """
        Ensure that a UserWarning is raised when one passes the ridge keyword
        argument to the `fit_mle` method of an Mixed Logit model object.
        """
        # Create the mnl model object whose coefficients will be estimated.
        base_mixl = self.mixl_obj

        # Create a variable for the arguments to the fit_mle function.
        fit_args = [self.fake_betas_ext, 2]

        # Create a variable for the fit_mle function's kwargs.
        # The print_res = False arguments are to make sure strings aren't
        # printed to the console unnecessarily.
        kwargs = {"ridge": "foo",
                  "print_res": False,
                  "seed": 1}

        # Test to make sure that the ridge warning message is printed when
        # using the ridge keyword argument
        with warnings.catch_warnings(record=True) as w:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            self.assertRaises(TypeError,
                              base_mixl.fit_mle,
                              *fit_args,
                              **kwargs)
            self.assertGreaterEqual(len(w), 1)
            self.assertIsInstance(w[0].category, type(UserWarning))
            self.assertIn(mixed_logit._ridge_warning_msg, str(w[0].message))

        return None

    def test_just_point_kwarg(self):
        """
        Ensure that calling `fit_mle` with `just_point = True` returns a
        dictionary with a 'x' key and a corresponding value that is an ndarray.
        """
        # Create a variable for the arguments to the fit_mle function.
        fit_args = [self.fake_betas_ext, 2]
        # Alias the function being tested
        func = self.mixl_obj.fit_mle
        # Get the necessary kwargs
        kwargs = {"just_point": True}
        # Get the function results
        func_result = func(*fit_args, **kwargs)
        # Perform the desired tests to make sure we get back a dictionary with
        # an "x" key in it and a value that is a ndarray.
        self.assertIsInstance(func_result, dict)
        self.assertIn("x", func_result)
        self.assertIsInstance(func_result["x"], np.ndarray)
        return None

    def test_outside_intercept_error_in_constructor(self):
        """
        Ensures that a ValueError is raised when the 'intercept_ref_pos' kwarg
        is passed to the MNL model constructor. This prevents people from
        expecting the use of outside intercept parameters to work with the MNL
        model.
        """
        # Create a variable for the standard arguments to this function.
        standard_args = [self.fake_old_df,
                         self.alt_id_column,
                         self.situation_id_column,
                         self.choice_column,
                         self.fake_spec]
        # Create a variable for the kwargs being passed to the constructor
        kwarg_map = {"names": self.fake_names,
                     "mixing_id_col": self.obs_id_column,
                     "mixing_vars": self.fake_mixing_vars,
                     "intercept_ref_pos": 2}

        self.assertRaises(ValueError,
                          mixed_logit.MixedLogit,
                          *standard_args,
                          **kwarg_map)
        return None

    def test_mixl_estimator_constructor(self):
        """
        Ensure that we can instantiate the mixed logit estimator object.
        """
        # Create various parameters needed to create the mixed logit estimator
        ridge = 0.5
        zero_vector = np.zeros(self.fake_betas_ext.shape[0])

        # Get the arguments needed for the object constructor
        args = [self.mixl_obj,
                self.mixl_obj.get_mappings_for_fit(),
                ridge,
                zero_vector,
                mixed_logit.split_param_vec]

        mixed_logit_etimator = mixed_logit.MixedEstimator(*args)

        npt.assert_allclose(mixed_logit_etimator.design_3d,
                            self.mixl_obj.design_3d)

        return None

    def test_add_mixl_specific_results_to_estimation_res(self):
        """
        Ensure that the desired key value pairs are added to the results
        dictionary.
        """
        # Create a fake results dictionary
        results_dict = {}
        results_dict["long_probs"] = self.prob_array

        # Get the probability of each sequence of choices, given the draws
        args = [results_dict["long_probs"],
                self.estimator.choice_vector,
                self.estimator.rows_to_mixers]
        kwargs = {"return_type": 'all'}
        prob_res = mlc.calc_choice_sequence_probs(*args, **kwargs)

        # Alias the function to be tested
        func = mixed_logit.add_mixl_specific_results_to_estimation_res

        # Map the new keys that should be added to results dict to the values
        # those keys should have.
        expected_results = {}
        expected_results["simulated_sequence_probs"] = prob_res[0]
        expected_results["expanded_sequence_probs"] = prob_res[1]

        # Test the function
        results_dict = func(self.estimator, results_dict)
        self.assertTrue([x in results_dict for x in expected_results.keys()])
        for key in expected_results:
            npt.assert_allclose(results_dict[key], expected_results[key])

        return None

    def test_convenience_calc_hessian(self):
        """
        Ensure that the constraints are correctly accounted for.
        test_calc_bhhh_hessian_approximation_mixed_logit() already tested the
        main functionality of the convenience_calc_hessian method.
        """
        # Set a constrained position on the estimator object
        constrained_idx = 1
        self.estimator.constrained_pos = [constrained_idx]

        # Alias the function that is being tested
        func = self.estimator.convenience_calc_hessian

        # Retrieve the BHHH matrix for this dataset
        bhhh_approx = func(self.fake_betas_ext)

        # Ensure that the second row and second column look like they came
        # from an identity matrix. The only non-zero value in the row should
        # be -1.
        self.assertEqual(bhhh_approx[constrained_idx, :].tolist(),
                         [0, -1, 0, 0])
        self.assertEqual(bhhh_approx[:, constrained_idx].tolist(),
                         [0, -1, 0, 0])

        return None

    def test_convenience_calc_fisher_approx(self):
        """
        Ensure that the desired placeholder is returned since we don't
        calculate the analytic hessian, and the BHHH approximation is already
        used to approximate the hessian.
        """
        expected_matrix = np.diag(-1 * np.ones(self.fake_betas_ext.shape[0]))

        func = self.estimator.convenience_calc_fisher_approx
        function_results = func(self.fake_betas_ext)

        npt.assert_allclose(expected_matrix, function_results)

        return None

    def test_execution_of_fit_mle(self):
        """
        This function simply tests whether or not the fit_mle function can
        be run without throwing an error.
        """
        init_vals = np.zeros(self.fake_betas_ext.shape[0])
        num_draws = 2
        seed = 1

        self.mixl_obj.fit_mle(init_vals, num_draws, seed=seed)

        return None
