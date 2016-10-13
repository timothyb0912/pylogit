"""
Tests for the choice_calcs.py file.
"""
import unittest
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import numpy.testing as npt

import pylogit.asym_logit as asym
import pylogit.choice_calcs as cc


# Use the following to always show the warnings
np.seterr(all='warn')
warnings.simplefilter("always")

class GenericTestCase(unittest.TestCase):
    """
    Defines the common setUp method used for the different type of tests.
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
                                                      [0, 0, 1]]))
        # Create the mapping between rows and individuals
        self.fake_rows_to_obs = csr_matrix(np.array([[1, 0],
                                                     [1, 0],
                                                     [1, 0],
                                                     [0, 1],
                                                     [0, 1]]))

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

        # Create the needed dataframe for the Asymmetric Logit constructor
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                     "alt_id": [1, 2, 3, 1, 3],
                                     "choice": [0, 1, 0, 0, 1],
                                     "x": self.fake_design[:, 0],
                                     "intercept": [1 for i in range(5)]})

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
        self.model_obj = asym.MNAL(*self.constructor_args,
                                   **self.constructor_kwargs)

        # Store a ridge penalty for use in calculations.
        self.ridge = 0.5

        return None


class ComputationalTests(GenericTestCase):
    """
    Tests the computational functions to make sure that they return the
    expected results.
    """
    def test_calc_asymptotic_covariance(self):
        """
        Ensure that the correct Huber-White covariance matrix is calculated.
        """
        ones_array = np.ones(5)
        # Create the hessian matrix for testing. It will be a 5 by 5 matrix.
        test_hessian = np.diag(2 * ones_array)
        # Create the approximation of the Fisher Information Matrix
        test_fisher_matrix = np.diag(ones_array)
        # Create the inverse of the hessian matrix.
        test_hess_inverse = np.diag(0.5 * ones_array)
        # Calculated the expected result
        expected_result = np.dot(test_hess_inverse,
                                 np.dot(test_fisher_matrix, test_hess_inverse))
        # Alias the function being tested
        func = cc.calc_asymptotic_covariance
        # Perform the test.
        function_results = func(test_hessian, test_fisher_matrix)
        self.assertIsInstance(function_results, np.ndarray)
        self.assertEqual(function_results.shape, test_hessian.shape)
        npt.assert_allclose(expected_result, function_results)

        return None

    def test_log_likelihood(self):
        """
        Ensure that we correctly calculate the log-likelihood, both with and
        without ridge penalties, and both with and without shape and intercept
        parameters.
        """
        # Create a utility transformation function for testing
        test_utility_transform = lambda x, *args: x
        # Calculate the index for each alternative for each individual
        test_index = self.fake_design.dot(self.fake_betas)
        # Exponentiate each index value
        exp_test_index  = np.exp(test_index)
        # Calculate the denominator for each probability
        interim_dot_product = self.fake_rows_to_obs.T.dot(exp_test_index)
        test_denoms = self.fake_rows_to_obs.dot(interim_dot_product)
        # Calculate the probabilities for each individual
        prob_array = exp_test_index / test_denoms
        # Calculate what the log-likelihood should be
        choices = self.fake_df[self.choice_col].values
        expected_log_likelihood = np.dot(choices, np.log(prob_array))
        # Create a set of intercepts, that are all zeros
        intercepts = np.zeros(2)
        # Combine all the 'parameters'
        test_all_params = np.concatenate([intercepts, self.fake_betas], axis=0)
        # Calculate what the log-likelihood should be with a ridge penalty
        penalty = self.ridge * (test_all_params**2).sum()
        expected_log_likelihood_penalized = expected_log_likelihood - penalty

        # Alias the function being tested
        func = cc.calc_log_likelihood
        # Create the arguments for the function being tested
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                choices,
                test_utility_transform]
        kwargs = {"intercept_params": intercepts,
                  "shape_params": None}

        # Perform the tests
        function_results = func(*args, **kwargs)
        self.assertAlmostEqual(expected_log_likelihood, function_results)
        kwargs["ridge"] = self.ridge
        function_results_2 = func(*args, **kwargs)
        self.assertAlmostEqual(expected_log_likelihood_penalized,
                               function_results_2)


        return None

    def test_array_size_error_in_calc_probabilities(self):
        """
        Ensure that a helpful ValueError is raised when a person tries to
        calculate probabilities using BOTH a  2D coefficient array and a 3D
        design matrix.
        """
        # Alias the function being tested
        func = cc.calc_probabilities

        # Create fake arguments for the function being tested.
        # Note these arguments are not valid in general, but suffice for
        # testing the functionality we care about in this function.
        args = [np.arange(9).reshape((3, 3)),
                np.arange(27).reshape((3, 3, 3)),
                None,
                None,
                None,
                None]

        # Note the error message that should be shown.
        msg_1 = "Cannot calculate probabilities with both 3D design matrix AND"
        msg_2 = " 2D coefficient array."
        msg = msg_1 + msg_2

        self.assertRaisesRegexp(ValueError,
                                msg,
                                func,
                                *args)

        return None

    def test_return_argument_error_in_calc_probabilities(self):
        """
        Ensure that a helpful ValueError is raised when a person tries to
        calculate probabilities using BOTH a return_long_probs == False and
        chosen_row_to_obs being None.
        """
        # Alias the function being tested
        func = cc.calc_probabilities

        # Create fake arguments for the function being tested.
        # Note these arguments are not valid in general, but suffice for
        # testing the functionality we care about in this function.
        args = [np.arange(9).reshape((3, 3)),
                np.arange(9).reshape((3, 3)),
                None,
                None,
                None,
                None]

        # Note the error message that should be shown.
        msg = "chosen_row_to_obs is None AND return_long_probs == False"

        self.assertRaisesRegexp(ValueError,
                                msg,
                                func,
                                *args)

        return None
