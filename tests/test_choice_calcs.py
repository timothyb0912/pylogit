"""
Tests for the choice_calcs.py file.
"""
import unittest
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import block_diag

import pylogit.asym_logit as asym
import pylogit.conditional_logit as mnl
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

        # Store the choices as their own array
        self.choice_array = self.fake_df[self.choice_col].values

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
    # Store a utility transformation function for the tests
    def utility_transform(self,
                          sys_utilities,
                          alt_IDs,
                          rows_to_alts,
                          shape_params,
                          intercept_params):
        return sys_utilities[:, None]

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

        # Test the function again, this time without intercepts
        kwargs["intercept_params"] = None
        function_results_3 = func(*args, **kwargs)
        self.assertAlmostEqual(expected_log_likelihood_penalized,
                               function_results_3)

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
        msg = "chosen_row_to_obs is None AND return_long_probs is False"

        self.assertRaisesRegexp(ValueError,
                                msg,
                                func,
                                *args)

        return None

    def test_1D_calc_probabilities(self):
        """
        Ensure that when using a 2D design matrix and 1D vector of parameters,
        that the calc_probabilities function returns the correct values. Note
        that this test will only verify the functionality under 'normal'
        conditions, where the values of the exponentiated indices do not go
        to zero nor to infinity.
        """
        # Calculate the index vector
        expected_index = self.fake_design.dot(self.fake_betas)
        # Calculate exp(index)
        expected_exp_index = np.exp(expected_index)
        # Calculate the sum of exp(index) for each individual
        denoms = self.fake_rows_to_obs.T.dot(expected_exp_index)
        # Calculate the expected probabilities
        expected_probs = expected_exp_index / self.fake_rows_to_obs.dot(denoms)

        # Alias the function to be tested
        func = cc.calc_probabilities

        # Collect the arguments needed for this function
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        function_results = func(*args, **kwargs)

        # Perform the tests
        self.assertIsInstance(function_results, np.ndarray)
        self.assertEqual(len(function_results.shape), 1)
        self.assertEqual(function_results.shape, (self.fake_design.shape[0],))
        npt.assert_allclose(function_results, expected_probs)

        return None

    def test_return_values_of_calc_probabilities(self):
        """
        Ensure that the various configuration of return values can all be
        returned.
        """
        # Calculate the index vector
        expected_index = self.fake_design.dot(self.fake_betas)
        # Calculate exp(index)
        expected_exp_index = np.exp(expected_index)
        # Calculate the sum of exp(index) for each individual
        denoms = self.fake_rows_to_obs.T.dot(expected_exp_index)
        # Calculate the expected probabilities
        expected_probs = expected_exp_index / self.fake_rows_to_obs.dot(denoms)
        # Extract the probabilities of the chosen alternatives for each
        # observaation
        chosen_indices = np.where(self.choice_array == 1)
        expected_chosen_probs = expected_probs[chosen_indices]

        # Alias the function to be tested
        func = cc.calc_probabilities

        # Create the chosen_row_to_obs mapping matrix
        choices_2d = self.choice_array[:, None]
        chosen_row_to_obs = self.fake_rows_to_obs.multiply(choices_2d)

        # Collect the arguments needed for this function
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]

        # kwargs_1 should result in long_probs being returned.
        kwargs_1 = {"intercept_params": self.fake_intercepts,
                    "shape_params": self.fake_shapes,
                    "return_long_probs": True}

        # kwargs_2 should result in (chosen_probs, long_probs being returned)
        kwargs_2 = {"intercept_params": self.fake_intercepts,
                    "shape_params": self.fake_shapes,
                    "chosen_row_to_obs": chosen_row_to_obs,
                    "return_long_probs": True}

        # kwargs_3 should result in chosen_probs being returned.
        kwargs_3 = {"intercept_params": self.fake_intercepts,
                    "shape_params": self.fake_shapes,
                    "chosen_row_to_obs": chosen_row_to_obs,
                    "return_long_probs": False}

        # Collect the expected results
        expected_results = [expected_probs,
                            (expected_chosen_probs, expected_probs),
                            expected_chosen_probs]

        # Perform the tests
        for pos, kwargs in enumerate([kwargs_1, kwargs_2, kwargs_3]):
            function_results = func(*args, **kwargs)
            if isinstance(function_results, tuple):
                expected_arrays = expected_results[pos]
                for array_pos, array in enumerate(function_results):
                    current_expected_array = expected_arrays[array_pos]
                    self.assertIsInstance(array, np.ndarray)
                    self.assertEqual(array.shape, current_expected_array.shape)
                    npt.assert_allclose(array, current_expected_array)
            else:
                expected_array = expected_results[pos]
                self.assertIsInstance(function_results, np.ndarray)
                self.assertEqual(function_results.shape, expected_array.shape)
                npt.assert_allclose(function_results, expected_array)

        return None

    def test_2D_calc_probabilities(self):
        """
        Ensure that when using either a 2D design matrix and 2D vector of
        parameters, or a 3D design matrix and 1D vector of parameters,
        that the calc_probabilities function returns the correct values. Note
        that this test will only verify the functionality under 'normal'
        conditions, where the values of the exponentiated indices do not go
        to zero nor to infinity.
        """
        # Designate a utility transform for this test
        utility_transform = mnl._mnl_utility_transform

        # Calculate the index vector
        expected_index = self.fake_design.dot(self.fake_betas)
        # Calculate exp(index)
        expected_exp_index = np.exp(expected_index)
        # Calculate the sum of exp(index) for each individual
        denoms = self.fake_rows_to_obs.T.dot(expected_exp_index)
        # Calculate the expected probabilities
        expected_probs = expected_exp_index / self.fake_rows_to_obs.dot(denoms)
        # Create the 2D vector of expected probs
        expected_probs_2d = np.concatenate([expected_probs[:, None],
                                            expected_probs[:, None]], axis=1)
        # Create the 2D coefficient vector
        betas_2d = np.concatenate([self.fake_betas[:, None],
                                   self.fake_betas[:, None]], axis=1)
        assert self.fake_design.dot(betas_2d).shape[1] > 1

        # Create the 3D design matrix
        design_3d = np.concatenate([self.fake_design[:, None, :],
                                    self.fake_design[:, None, :]], axis=1)

        # Alias the function to be tested
        func = cc.calc_probabilities

        # Collect the arguments needed for this function
        args = [betas_2d,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                utility_transform]
        # The kwargs below mean that only the long format probabilities will
        # be returned.
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "chosen_row_to_obs": None,
                  "return_long_probs": True}
        function_results_1 = func(*args, **kwargs)

        # Now test the functions with the various multidimensional argumnts
        args[0] = self.fake_betas
        args[1] = design_3d
        function_results_2 = func(*args, **kwargs)

        # Now try the results when calling for chosen_probs as well
        chosen_row_to_obs = self.fake_rows_to_obs.multiply(
            self.choice_array[:, None])
        kwargs["chosen_row_to_obs"] = chosen_row_to_obs
        chosen_probs, function_results_3 = func(*args, **kwargs)

        # Perform the tests using a 2d coefficient array
        for function_results in [function_results_1,
                                 function_results_2,
                                 function_results_3]:
            self.assertIsInstance(function_results, np.ndarray)
            self.assertEqual(len(function_results.shape), 2)
            self.assertEqual(function_results.shape,
                             (self.fake_design.shape[0], 2))
            npt.assert_allclose(function_results, expected_probs_2d)

        chosen_idx = np.where(self.choice_array == 1)[0]
        self.assertIsInstance(chosen_probs, np.ndarray)
        self.assertEqual(len(chosen_probs.shape), 2)
        npt.assert_allclose(chosen_probs, expected_probs_2d[chosen_idx, :])

        return None

    def test_calc_probabilities_robustness_to_under_overflow(self):
        """
        Ensure that the calc_probabilities function correctly handles under-
        and overflow in the exponential of the systematic utilities.
        """
        # Create a design array that will test the under- and over-flow
        # capabilities of the calc_probabilities function.
        extreme_design = np.array([[1],
                                   [800 / self.fake_betas[0]],
                                   [-800 / self.fake_betas[0]],
                                   [-800 / self.fake_betas[0]],
                                   [3]])
        # Calculate the index vector
        expected_index = extreme_design.dot(self.fake_betas)
        # Calculate exp(index)
        expected_exp_index = np.exp(expected_index)
        # Guard against over and underflow
        expected_exp_index[1] = np.exp(cc.max_exponent_val)
        expected_exp_index[[2, 3]] = np.exp(cc.min_exponent_val)
        # Calculate the sum of exp(index) for each individual
        denoms = self.fake_rows_to_obs.T.dot(expected_exp_index)
        # Calculate the expected probabilities
        expected_probs = expected_exp_index / self.fake_rows_to_obs.dot(denoms)
        # Guard against underflow
        expected_probs[expected_probs == 0.0] = cc.min_comp_value

        # Alias the function to be tested
        func = cc.calc_probabilities

        # Collect the arguments needed for this function
        args = [self.fake_betas,
                extreme_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        function_results = func(*args, **kwargs)

        # Perform the tests
        self.assertIsInstance(function_results, np.ndarray)
        self.assertEqual(len(function_results.shape), 1)
        self.assertEqual(function_results.shape, (self.fake_design.shape[0],))
        npt.assert_allclose(function_results, expected_probs)

        return None

    def test_calc_gradient_no_shapes_no_intercepts(self):
        """
        Ensure that calc_gradient returns the correct values when there are no
        shape parameters and no intercept parameters.
        """
        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        dh_dv = diags(np.ones(self.fake_design.shape[0]), 0, format='csr')
        def transform_deriv_v(*args):
            return dh_dv

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(*args):
            return None

        def transform_deriv_shapes(*args):
            return None

        # Collect the arguments needed to calculate the probabilities
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        # Calculate the required probabilities
        probs = cc.calc_probabilities(*args, **kwargs)
        # In this scenario, the gradient should be (Y- P)'(dh_dv * dv_db)
        # which simplifies to (Y- P)'(dh_dv * X)
        error_vec = (self.choice_array - probs)[None, :]
        expected_gradient = error_vec.dot(dh_dv.dot(self.fake_design)).ravel()

        # Alias the function being tested
        func = cc.calc_gradient

        # Collect the arguments for the function being tested
        gradient_args = [self.fake_betas,
                         self.fake_design,
                         self.fake_df[self.alt_id_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_alts,
                         self.choice_array,
                         self.utility_transform,
                         transform_deriv_shapes,
                         transform_deriv_v,
                         transform_deriv_intercepts,
                         None,
                         None,
                         None]
        function_gradient = func(*gradient_args)

        # Perform the required tests
        self.assertIsInstance(function_gradient, np.ndarray)
        self.assertEqual(function_gradient.shape, (self.fake_betas.shape[0],))
        npt.assert_allclose(function_gradient, expected_gradient)

        # Test the gradient function with the ridge argument
        gradient_args[-1] = self.ridge
        new_expected_gradient = (expected_gradient -
                                 2 * self.ridge * self.fake_betas[0])
        function_gradient_penalized = func(*gradient_args)

        self.assertIsInstance(function_gradient, np.ndarray)
        self.assertEqual(function_gradient.shape, (self.fake_betas.shape[0],))
        npt.assert_allclose(function_gradient_penalized, new_expected_gradient)

        return None

    def test_calc_gradient_no_shapes(self):
        """
        Ensure that calc_gradient returns the correct values when there are no
        shape parameters but there are intercept parameters.
        """
        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        dh_dv = diags(np.ones(self.fake_design.shape[0]), 0, format='csr')
        def transform_deriv_v(*args):
            return dh_dv

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        dh_d_intercept = self.fake_rows_to_alts[:, 1:]
        def transform_deriv_intercepts(*args):
            return dh_d_intercept

        def transform_deriv_shapes(*args):
            return None

        # Collect the arguments needed to calculate the probabilities
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        # Calculate the required probabilities
        probs = cc.calc_probabilities(*args, **kwargs)
        # In this scenario, the gradient should be (Y- P)'(dh_d_theta)
        # which simplifies to (Y- P)'[dh_d_intercept | dh_d_beta]
        # and finally to (Y- P)'[dh_d_intercept | dh_dv * X]
        error_vec = (self.choice_array - probs)[None, :]
        dh_d_beta = dh_dv.dot(self.fake_design)
        dh_d_theta = np.concatenate((dh_d_intercept.A, dh_d_beta), axis=1)
        expected_gradient = error_vec.dot(dh_d_theta).ravel()

        # Alias the function being tested
        func = cc.calc_gradient

        # Collect the arguments for the function being tested
        gradient_args = [self.fake_betas,
                         self.fake_design,
                         self.fake_df[self.alt_id_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_alts,
                         self.choice_array,
                         self.utility_transform,
                         transform_deriv_shapes,
                         transform_deriv_v,
                         transform_deriv_intercepts,
                         self.fake_intercepts,
                         None,
                         None]
        function_gradient = func(*gradient_args)

        # Perform the required tests
        self.assertIsInstance(function_gradient, np.ndarray)
        self.assertEqual(function_gradient.shape,
                         (self.fake_betas.shape[0] +
                          self.fake_intercepts.shape[0],))
        npt.assert_allclose(function_gradient, expected_gradient)

        return None

    def test_calc_gradient_no_intercepts(self):
        """
        Ensure that calc_gradient returns the correct values when there are no
        intercept parameters but there are shape parameters.
        """
        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        dh_dv = diags(np.ones(self.fake_design.shape[0]), 0, format='csr')
        def transform_deriv_v(*args):
            return dh_dv

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(*args):
            return None

        fake_deriv = np.exp(self.fake_shapes)[None, :]
        dh_d_shape = self.fake_rows_to_alts[:, 1:].multiply(fake_deriv)
        def transform_deriv_shapes(*args):
            return dh_d_shape

        # Collect the arguments needed to calculate the probabilities
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        # Calculate the required probabilities
        probs = cc.calc_probabilities(*args, **kwargs)
        # In this scenario, the gradient should be (Y- P)'(dh_d_theta)
        # which simplifies to (Y- P)'[dh_d_shape | dh_d_beta]
        # and finally to (Y- P)'[dh_d_shape | dh_dv * X]
        error_vec = (self.choice_array - probs)[None, :]
        dh_d_beta = dh_dv.dot(self.fake_design)
        dh_d_theta = np.concatenate((dh_d_shape.A, dh_d_beta), axis=1)
        expected_gradient = error_vec.dot(dh_d_theta).ravel()

        # Alias the function being tested
        func = cc.calc_gradient

        # Collect the arguments for the function being tested
        gradient_args = [self.fake_betas,
                         self.fake_design,
                         self.fake_df[self.alt_id_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_alts,
                         self.choice_array,
                         self.utility_transform,
                         transform_deriv_shapes,
                         transform_deriv_v,
                         transform_deriv_intercepts,
                         None,
                         self.fake_shapes,
                         None]
        function_gradient = func(*gradient_args)

        # Perform the required tests
        self.assertIsInstance(function_gradient, np.ndarray)
        self.assertEqual(function_gradient.shape,
                         (self.fake_betas.shape[0] +
                          self.fake_intercepts.shape[0],))
        npt.assert_allclose(function_gradient, expected_gradient)

        return None

    def test_calc_gradient_shapes_and_intercepts(self):
        """
        Ensure that calc_gradient returns the correct values when there are
        shape and intercept parameters.
        """
        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        dh_dv = diags(np.ones(self.fake_design.shape[0]), 0, format='csr')
        def transform_deriv_v(*args):
            return dh_dv

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        dh_d_intercept = self.fake_rows_to_alts[:, 1:]
        def transform_deriv_intercepts(*args):
            return dh_d_intercept

        fake_deriv = np.exp(self.fake_shapes)[None, :]
        dh_d_shape = self.fake_rows_to_alts[:, 1:].multiply(fake_deriv)
        def transform_deriv_shapes(*args):
            return dh_d_shape

        # Collect the arguments needed to calculate the probabilities
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        # Calculate the required probabilities
        probs = cc.calc_probabilities(*args, **kwargs)
        # In this scenario, the gradient should be (Y- P)'(dh_d_theta)
        # which simplifies to (Y- P)'[dh_d_shape | dh_d_intercept | dh_d_beta]
        # and finally to (Y- P)'[dh_d_shape | dh_d_intercept | dh_dv * X]
        error_vec = (self.choice_array - probs)[None, :]
        dh_d_beta = dh_dv.dot(self.fake_design)
        dh_d_theta = np.concatenate((dh_d_shape.A,
                                     dh_d_intercept.A,
                                     dh_d_beta), axis=1)
        expected_gradient = error_vec.dot(dh_d_theta).ravel()

        # Alias the function being tested
        func = cc.calc_gradient

        # Collect the arguments for the function being tested
        gradient_args = [self.fake_betas,
                         self.fake_design,
                         self.fake_df[self.alt_id_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_alts,
                         self.choice_array,
                         self.utility_transform,
                         transform_deriv_shapes,
                         transform_deriv_v,
                         transform_deriv_intercepts,
                         self.fake_intercepts,
                         self.fake_shapes,
                         None]
        function_gradient = func(*gradient_args)

        # Perform the required tests
        self.assertIsInstance(function_gradient, np.ndarray)
        self.assertEqual(function_gradient.shape,
                         (self.fake_betas.shape[0] +
                          self.fake_intercepts.shape[0] +
                          self.fake_shapes.shape[0],))
        npt.assert_allclose(function_gradient, expected_gradient)

        return None

    def test_create_matrix_block_indices(self):
        """
        Ensure that create_matrix_block_indices returns the expected results.
        """
        # Note that we have two observations, the first with three alternatives
        # and the second with two alternatives.
        expected_results = [np.array([0, 1, 2]), np.array([3, 4])]

        # Get the results of the function being tested
        results = cc.create_matrix_block_indices(self.fake_rows_to_obs)

        # Test that the two sets of results are equal
        self.assertIsInstance(results, list)
        self.assertTrue(all([isinstance(x, np.ndarray) for x in results]))
        npt.assert_allclose(expected_results[0], results[0])
        npt.assert_allclose(expected_results[1], results[1])

        return None

    def test_robust_outer_product(self):
        """
        Ensure that robust_outer_product returns the expected results.
        Unfortunately, I cannot find a good case now where using the regular
        outer product gives incorrect results. However without a compelling
        reason to remove the function, I'll trust my earlier judgement in
        creating it in the first place.
        """
        # Define a vector whose outer product we want to take
        x = np.array([1e-100, 0.01])
        outer_product = np.outer(x, x)
        robust_outer_product = cc.robust_outer_product(x, x)

        # Perform the desired tests
        self.assertIsInstance(robust_outer_product, np.ndarray)
        self.assertEqual(robust_outer_product.shape, outer_product.shape)
        npt.assert_allclose(outer_product, robust_outer_product)

        return None

    def test_create_matrix_blocks(self):
        """
        Ensure that create_matrix_blocks returns expected results when not
        having to correct for underflow.
        """
        # Designate a utility transform for this test
        utility_transform = mnl._mnl_utility_transform

        # Collect the arguments needed for this function
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        # Get the long-format probabilities
        long_probs = cc.calc_probabilities(*args, **kwargs)

        # Get the matrix-block indices
        matrix_indices = cc.create_matrix_block_indices(self.fake_rows_to_obs)

        # Create the matrix block for individual 1.
        matrix_block_1 = (np.diag(long_probs[:3]) -
                          np.outer(long_probs[:3], long_probs[:3]))
        matrix_block_2 = (np.diag(long_probs[3:]) -
                          np.outer(long_probs[3:], long_probs[3:]))
        # Create a list of the expected results
        expected_results = [matrix_block_1, matrix_block_2]

        # Get the function results
        func_results = cc.create_matrix_blocks(long_probs, matrix_indices)
        for pos, result in enumerate(func_results):
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, expected_results[pos].shape)
            npt.assert_allclose(result, expected_results[pos])

        return None

    def test_create_matrix_blocks_with_underflow(self):
        """
        Ensure that create_matrix_blocks returns expected results when also
        having to correct for underflow.
        """
        # Get the long-format probabilities
        long_probs = np.array([cc.min_comp_value,
                               cc.min_comp_value,
                               1.0,
                               cc.min_comp_value,
                               1.0])

        # Get the matrix-block indices
        matrix_indices = cc.create_matrix_block_indices(self.fake_rows_to_obs)

        # Create the matrix block for individual 1.
        row_1 = [cc.min_comp_value, -cc.min_comp_value, -cc.min_comp_value]
        row_2 = [-cc.min_comp_value, cc.min_comp_value, -cc.min_comp_value]
        row_3 = [-cc.min_comp_value, -cc.min_comp_value, cc.min_comp_value]
        matrix_block_1 = np.array([row_1, row_2, row_3])

        matrix_block_2 = (np.diag(long_probs[3:]) -
                          np.outer(long_probs[3:], long_probs[3:]))
        # Assuming that no probabilities should actually be zero or one,
        # the underflow guard would set the last value to a very small,
        # positive number 
        matrix_block_2[-1, -1] = cc.min_comp_value

        # Create a list of the expected results
        expected_results = [matrix_block_1, matrix_block_2]

        # Get the function results
        func_results = cc.create_matrix_blocks(long_probs, matrix_indices)
        for pos, result in enumerate(func_results):
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, expected_results[pos].shape)
            npt.assert_allclose(result, expected_results[pos])

        return None

    def test_calc_fisher_info_matrix_no_shapes_no_intercepts(self):
        """
        Ensure that calc_fisher_info_matrix returns the expected values when
        there are no shape or intercept parameters.
        """
        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        def transform_deriv_v(sys_utilities,
                              alt_IDs,
                              rows_to_alts,
                              shape_params):
            return diags(np.ones(sys_utilities.shape[0]), 0, format='csr')

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(*args):
            return None

        def transform_deriv_shapes(*args):
            return None

        # Collect the arguments for the function being tested
        gradient_args = [self.fake_betas,
                         self.fake_design,
                         self.fake_df[self.alt_id_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_alts,
                         self.choice_array,
                         self.utility_transform,
                         transform_deriv_shapes,
                         transform_deriv_v,
                         transform_deriv_intercepts,
                         None,
                         None,
                         None]

        gradient_args[1] = self.fake_design[:3, :]
        gradient_args[2] = self.fake_df[self.alt_id_col].values[:3]
        gradient_args[3] = self.fake_rows_to_obs[:3, :]
        gradient_args[4] = self.fake_rows_to_alts[:3, :]
        gradient_args[5] = self.choice_array[:3]
        gradient_1 = cc.calc_gradient(*gradient_args)

        gradient_args[1] = self.fake_design[3:, :]
        gradient_args[2] = self.fake_df[self.alt_id_col].values[3:]
        gradient_args[3] = self.fake_rows_to_obs[3:, :]
        gradient_args[4] = self.fake_rows_to_alts[3:, :]
        gradient_args[5] = self.choice_array[3:]
        gradient_2 = cc.calc_gradient(*gradient_args)

        # Calcuate the BHHH approximation to the Fisher Info Matrix
        expected_result = (np.outer(gradient_1, gradient_1) +
                           np.outer(gradient_2, gradient_2))

        # Alias the function being tested
        func = cc.calc_fisher_info_matrix
        # Get the results of the function being tested
        gradient_args[1] = self.fake_design
        gradient_args[2] = self.fake_df[self.alt_id_col].values
        gradient_args[3] = self.fake_rows_to_obs
        gradient_args[4] = self.fake_rows_to_alts
        gradient_args[5] = self.choice_array
        function_result = func(*gradient_args)

        # Perform the required tests
        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        npt.assert_allclose(function_result, expected_result)

        # Test the function with the ridge penalty
        expected_result -= 2 * self.ridge
        gradient_args[-1] = self.ridge
        function_result = func(*gradient_args)

        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        npt.assert_allclose(function_result, expected_result)

        return None

    def test_calc_fisher_info_matrix_no_intercepts(self):
        """
        Ensure that calc_fisher_info_matrix returns the expected values when
        there are no intercept parameters.
        """
        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        def transform_deriv_v(sys_utilities,
                              alt_IDs,
                              rows_to_alts,
                              shape_params):
            return diags(np.ones(sys_utilities.shape[0]), 0, format='csr')

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(*args):
            return None

        fake_deriv = np.exp(self.fake_shapes)[None, :]
        def transform_deriv_shapes(sys_utilities,
                                   alt_IDs,
                                   rows_to_alts,
                                   shape_params):
            return rows_to_alts[:, 1:].multiply(fake_deriv)

        # Collect the arguments needed to calculate the gradients
        gradient_args = [self.fake_betas,
                         self.fake_design,
                         self.fake_df[self.alt_id_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_alts,
                         self.choice_array,
                         self.utility_transform,
                         transform_deriv_shapes,
                         transform_deriv_v,
                         transform_deriv_intercepts,
                         None,
                         self.fake_shapes,
                         None]

        # Get the gradient, for each observation, separately.
        # Note this test expects that we only have two observations.
        gradient_args[1] = self.fake_design[:3, :]
        gradient_args[2] = self.fake_df[self.alt_id_col].values[:3]
        gradient_args[3] = self.fake_rows_to_obs[:3, :]
        gradient_args[4] = self.fake_rows_to_alts[:3, :]
        gradient_args[5] = self.choice_array[:3]
        gradient_1 = cc.calc_gradient(*gradient_args)

        gradient_args[1] = self.fake_design[3:, :]
        gradient_args[2] = self.fake_df[self.alt_id_col].values[3:]
        gradient_args[3] = self.fake_rows_to_obs[3:, :]
        gradient_args[4] = self.fake_rows_to_alts[3:, :]
        gradient_args[5] = self.choice_array[3:]
        gradient_2 = cc.calc_gradient(*gradient_args)

        # Calcuate the BHHH approximation to the Fisher Info Matrix
        expected_result = (np.outer(gradient_1, gradient_1) +
                           np.outer(gradient_2, gradient_2))

        # Alias the function being tested
        func = cc.calc_fisher_info_matrix
        # Get the results of the function being tested
        gradient_args[1] = self.fake_design
        gradient_args[2] = self.fake_df[self.alt_id_col].values
        gradient_args[3] = self.fake_rows_to_obs
        gradient_args[4] = self.fake_rows_to_alts
        gradient_args[5] = self.choice_array
        function_result = func(*gradient_args)

        # Perform the required tests
        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        npt.assert_allclose(function_result, expected_result)

        return None

    def test_calc_fisher_info_matrix_no_shapes(self):
        """
        Ensure that calc_fisher_info_matrix returns the expected values when
        there are no shape parameters.
        """
        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        def transform_deriv_v(sys_utilities,
                              alt_IDs,
                              rows_to_alts,
                              shape_params):
            return diags(np.ones(sys_utilities.shape[0]), 0, format='csr')

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(sys_utilities,
                                       alt_IDs,
                                       rows_to_alts,
                                       shape_params):
            return rows_to_alts[:, 1:]

        def transform_deriv_shapes(*args):
            return None

        # Collect the arguments needed to calculate the gradients
        gradient_args = [self.fake_betas,
                         self.fake_design,
                         self.fake_df[self.alt_id_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_alts,
                         self.choice_array,
                         self.utility_transform,
                         transform_deriv_shapes,
                         transform_deriv_v,
                         transform_deriv_intercepts,
                         self.fake_intercepts,
                         None,
                         None]

        # Get the gradient, for each observation, separately.
        # Note this test expects that we only have two observations.
        gradient_args[1] = self.fake_design[:3, :]
        gradient_args[2] = self.fake_df[self.alt_id_col].values[:3]
        gradient_args[3] = self.fake_rows_to_obs[:3, :]
        gradient_args[4] = self.fake_rows_to_alts[:3, :]
        gradient_args[5] = self.choice_array[:3]
        gradient_1 = cc.calc_gradient(*gradient_args)

        gradient_args[1] = self.fake_design[3:, :]
        gradient_args[2] = self.fake_df[self.alt_id_col].values[3:]
        gradient_args[3] = self.fake_rows_to_obs[3:, :]
        gradient_args[4] = self.fake_rows_to_alts[3:, :]
        gradient_args[5] = self.choice_array[3:]
        gradient_2 = cc.calc_gradient(*gradient_args)

        # Calcuate the BHHH approximation to the Fisher Info Matrix
        expected_result = (np.outer(gradient_1, gradient_1) +
                           np.outer(gradient_2, gradient_2))

        # Alias the function being tested
        func = cc.calc_fisher_info_matrix
        # Get the results of the function being tested
        gradient_args[1] = self.fake_design
        gradient_args[2] = self.fake_df[self.alt_id_col].values
        gradient_args[3] = self.fake_rows_to_obs
        gradient_args[4] = self.fake_rows_to_alts
        gradient_args[5] = self.choice_array
        function_result = func(*gradient_args)

        # Perform the required tests
        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        npt.assert_allclose(function_result, expected_result)

        return None

    def test_calc_fisher_info_matrix(self):
        """
        Ensure that calc_fisher_info_matrix returns the expected values.
        """
        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        def transform_deriv_v(sys_utilities,
                              alt_IDs,
                              rows_to_alts,
                              shape_params):
            return diags(np.ones(sys_utilities.shape[0]), 0, format='csr')

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(sys_utilities,
                                       alt_IDs,
                                       rows_to_alts,
                                       shape_params):
            return rows_to_alts[:, 1:]

        fake_deriv = np.exp(self.fake_shapes)[None, :]
        def transform_deriv_shapes(sys_utilities,
                                   alt_IDs,
                                   rows_to_alts,
                                   shape_params):
            return rows_to_alts[:, 1:].multiply(fake_deriv)

        # Collect the arguments needed to calculate the gradients
        gradient_args = [self.fake_betas,
                         self.fake_design,
                         self.fake_df[self.alt_id_col].values,
                         self.fake_rows_to_obs,
                         self.fake_rows_to_alts,
                         self.choice_array,
                         self.utility_transform,
                         transform_deriv_shapes,
                         transform_deriv_v,
                         transform_deriv_intercepts,
                         self.fake_intercepts,
                         self.fake_shapes,
                         None]

        # Get the gradient, for each observation, separately.
        # Note this test expects that we only have two observations.
        gradient_args[1] = self.fake_design[:3, :]
        gradient_args[2] = self.fake_df[self.alt_id_col].values[:3]
        gradient_args[3] = self.fake_rows_to_obs[:3, :]
        gradient_args[4] = self.fake_rows_to_alts[:3, :]
        gradient_args[5] = self.choice_array[:3]
        gradient_1 = cc.calc_gradient(*gradient_args)

        gradient_args[1] = self.fake_design[3:, :]
        gradient_args[2] = self.fake_df[self.alt_id_col].values[3:]
        gradient_args[3] = self.fake_rows_to_obs[3:, :]
        gradient_args[4] = self.fake_rows_to_alts[3:, :]
        gradient_args[5] = self.choice_array[3:]
        gradient_2 = cc.calc_gradient(*gradient_args)

        # Calcuate the BHHH approximation to the Fisher Info Matrix
        expected_result = (np.outer(gradient_1, gradient_1) +
                           np.outer(gradient_2, gradient_2))

        # Alias the function being tested
        func = cc.calc_fisher_info_matrix
        # Get the results of the function being tested
        gradient_args[1] = self.fake_design
        gradient_args[2] = self.fake_df[self.alt_id_col].values
        gradient_args[3] = self.fake_rows_to_obs
        gradient_args[4] = self.fake_rows_to_alts
        gradient_args[5] = self.choice_array
        function_result = func(*gradient_args)

        # Perform the required tests
        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        npt.assert_allclose(function_result, expected_result)

        return None

    def test_calc_hessian_no_shapes_no_intercept(self):
        """
        Ensure that the calc_hessian function returns expected results when
        there are no shape parameters and no intercept parameters.
        """
        # Alias the design matrix
        design = self.fake_design
        # Get the matrix block indices for the test
        matrix_indices = cc.create_matrix_block_indices(self.fake_rows_to_obs)
        # Calculate the probabilities for this test.
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        probs = cc.calc_probabilities(*args, **kwargs)
        # Get the matrix blocks for dP_i_dH_i
        matrix_blocks = cc.create_matrix_blocks(probs, matrix_indices)
        # Create the dP_dH matrix that represents the derivative of the
        # long probabilities with respect to the array of transformed index
        # values / systematic utilities
        dP_dH = block_diag(matrix_blocks)

        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        def transform_deriv_v(sys_utilities,
                              alt_IDs,
                              rows_to_alts,
                              shape_params):
            return diags(np.ones(sys_utilities.shape[0]), 0, format='csr')

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(*args):
            return None

        def transform_deriv_shapes(*args):
            return None

        # Collect the arguments for the hessian function being tested
        hessian_args = [self.fake_betas,
                        self.fake_design,
                        self.fake_df[self.alt_id_col].values,
                        self.fake_rows_to_obs,
                        self.fake_rows_to_alts,
                        self.utility_transform,
                        transform_deriv_shapes,
                        transform_deriv_v,
                        transform_deriv_intercepts,
                        matrix_indices,
                        None,
                        None,
                        None]

        # Calculate the expected result
        # Since we're essentially dealing with an MNL model in this test,
        # the expected answer is -X^T * dP_dH * X

        expected_result = (-1 * design.T.dot(dP_dH.dot(design)))

        # Alias the function being tested
        func = cc.calc_hessian
        # Get the results of the function being tested
        function_result = func(*hessian_args)

        # Perform the required tests
        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        npt.assert_allclose(function_result, expected_result)

        # Test the function with the ridge penalty
        expected_result -= 2 * self.ridge
        hessian_args[-1] = self.ridge
        function_result = func(*hessian_args)

        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        npt.assert_allclose(function_result, expected_result)

        return None

    def test_calc_hessian(self):
        """
        Ensure that the calc_hessian function returns expected results when
        there are both shape parameters and intercept parameters.
        """
        # Alias the design matrix
        design = self.fake_design
        # Get the matrix block indices for the test
        matrix_indices = cc.create_matrix_block_indices(self.fake_rows_to_obs)
        # Calculate the probabilities for this test.
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        probs = cc.calc_probabilities(*args, **kwargs)
        # Get the matrix blocks for dP_i_dH_i
        matrix_blocks = cc.create_matrix_blocks(probs, matrix_indices)
        # Create the dP_dH matrix that represents the derivative of the
        # long probabilities with respect to the array of transformed index
        # values / systematic utilities
        dP_dH = block_diag(matrix_blocks)

        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        def transform_deriv_v(sys_utilities,
                              alt_IDs,
                              rows_to_alts,
                              shape_params):
            return diags(np.ones(sys_utilities.shape[0]), 0, format='csr')

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(sys_utilities,
                                       alt_IDs,
                                       rows_to_alts,
                                       intercept_params):
            return rows_to_alts[:, 1:]

        fake_deriv = np.exp(self.fake_shapes)[None, :]
        def transform_deriv_shapes(sys_utilities,
                                   alt_IDs,
                                   rows_to_alts,
                                   shape_params):
            return rows_to_alts[:, 1:].multiply(fake_deriv)


        # Collect the arguments for the hessian function being tested
        hessian_args = [self.fake_betas,
                        self.fake_design,
                        self.fake_df[self.alt_id_col].values,
                        self.fake_rows_to_obs,
                        self.fake_rows_to_alts,
                        self.utility_transform,
                        transform_deriv_shapes,
                        transform_deriv_v,
                        transform_deriv_intercepts,
                        matrix_indices,
                        self.fake_intercepts,
                        self.fake_shapes,
                        None]

        # Calculate the derivative of the transformation vector with respect
        # to the shape parameters.
        args = (design.dot(self.fake_betas),
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_alts,
                self.fake_shapes)
        dh_d_shape = transform_deriv_shapes(*args)

        # Calculate the derivative of the transformation vector with respect
        # to the intercept parameters
        dh_d_intercept = self.fake_rows_to_alts[:, 1:]

        # Calculate the various matrices needed for the expected result
        # Note dH_dV is the Identity matrix in this test.
        # See the documentation pdf for a description of what each of these
        # matrixes are.
        # h_33 is -X^T * dP_dH * X. This is the hessian in the standard MNL
        h_33 = np.asarray(-1 * design.T.dot(dP_dH.dot(design)))
        # h_32 is -X^T * dH_dV^T * dP_dH * dH_d_intercept
        h_32 = np.asarray(-1 * design.T.dot(dP_dH.dot(dh_d_intercept.A)))
        # h_31 is -X^T * dH_dV^T * dP_dH * dH_d_shape
        h_31 = np.asarray(-1 * design.T.dot(dP_dH.dot(dh_d_shape.A)))
        # h_21 = -dH_d_intercept^T * dP_dH * dH_d_shape
        h_21 = np.asarray(-1 * dh_d_intercept.T.dot(dP_dH.dot(dh_d_shape.A)))
        # h_22 = -dH_d_intercept^T * dP_dH * dH_d_intercept
        h_22 = np.asarray(-1 *
                          dh_d_intercept.T.dot(dP_dH.dot(dh_d_intercept.A)))
        # h_11 = -dH_d_shape^T * dP_dH * dH_d_shape
        h_11 = np.asarray(-1 * dh_d_shape.T.dot(dP_dH.dot(dh_d_shape.A)))

        # Create the final hessian
        top_row = np.concatenate((h_11, h_21.T, h_31.T), axis=1)
        middle_row = np.concatenate((h_21, h_22, h_32.T), axis=1)
        bottom_row = np.concatenate((h_31, h_32, h_33), axis=1)
        expected_result = np.concatenate((top_row, middle_row, bottom_row),
                                         axis=0)

        # Alias the function being tested
        func = cc.calc_hessian
        # Get the results of the function being tested
        function_result = func(*hessian_args)

        # Perform the required tests
        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        self.assertFalse(isinstance(function_result,
                                    np.matrixlib.defmatrix.matrix))
        npt.assert_allclose(function_result, expected_result)

        # Test the function with the ridge penalty
        expected_result -= 2 * self.ridge
        hessian_args[-1] = self.ridge
        function_result = func(*hessian_args)

        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        self.assertFalse(isinstance(function_result,
                                    np.matrixlib.defmatrix.matrix))
        npt.assert_allclose(function_result, expected_result)

        return None

    def test_calc_hessian_no_shapes(self):
        """
        Ensure that the calc_hessian function returns expected results when
        there are no shape parameters.
        """
        # Alias the design matrix
        design = self.fake_design
        # Get the matrix block indices for the test
        matrix_indices = cc.create_matrix_block_indices(self.fake_rows_to_obs)
        # Calculate the probabilities for this test.
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        probs = cc.calc_probabilities(*args, **kwargs)
        # Get the matrix blocks for dP_i_dH_i
        matrix_blocks = cc.create_matrix_blocks(probs, matrix_indices)
        # Create the dP_dH matrix that represents the derivative of the
        # long probabilities with respect to the array of transformed index
        # values / systematic utilities
        dP_dH = block_diag(matrix_blocks)

        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        def transform_deriv_v(sys_utilities,
                              alt_IDs,
                              rows_to_alts,
                              shape_params):
            return diags(np.ones(sys_utilities.shape[0]), 0, format='csr')

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(sys_utilities,
                                       alt_IDs,
                                       rows_to_alts,
                                       intercept_params):
            return rows_to_alts[:, 1:]

        def transform_deriv_shapes(sys_utilities,
                                   alt_IDs,
                                   rows_to_alts,
                                   shape_params):
            return None


        # Collect the arguments for the hessian function being tested
        hessian_args = [self.fake_betas,
                        self.fake_design,
                        self.fake_df[self.alt_id_col].values,
                        self.fake_rows_to_obs,
                        self.fake_rows_to_alts,
                        self.utility_transform,
                        transform_deriv_shapes,
                        transform_deriv_v,
                        transform_deriv_intercepts,
                        matrix_indices,
                        self.fake_intercepts,
                        None,
                        None]

        # Calculate the derivative of the transformation vector with respect
        # to the intercept parameters
        dh_d_intercept = self.fake_rows_to_alts[:, 1:]

        # Calculate the various matrices needed for the expected result
        # Note dH_dV is the Identity matrix in this test.
        # See the documentation pdf for a description of what each of these
        # matrixes are.
        # h_33 is -X^T * dP_dH * X. This is the hessian in the standard MNL
        h_33 = np.asarray(-1 * design.T.dot(dP_dH.dot(design)))
        # h_32 is -X^T * dH_dV^T * dP_dH * dH_d_intercept
        h_32 = np.asarray(-1 * design.T.dot(dP_dH.dot(dh_d_intercept.A)))
        # h_22 = -dH_d_intercept^T * dP_dH * dH_d_intercept
        h_22 = np.asarray(-1 *
                          dh_d_intercept.T.dot(dP_dH.dot(dh_d_intercept.A)))

        # Create the final hessian
        middle_row = np.concatenate((h_22, h_32.T), axis=1)
        bottom_row = np.concatenate((h_32, h_33), axis=1)
        expected_result = np.concatenate((middle_row, bottom_row), axis=0)

        # Alias the function being tested
        func = cc.calc_hessian
        # Get the results of the function being tested
        function_result = func(*hessian_args)

        # Perform the required tests
        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        self.assertFalse(isinstance(function_result,
                                    np.matrixlib.defmatrix.matrix))
        npt.assert_allclose(function_result, expected_result)

        return None

    def test_calc_hessian_no_intercepts(self):
        """
        Ensure that the calc_hessian function returns expected results when
        there are no intercept parameters.
        """
        # Alias the design matrix
        design = self.fake_design
        # Get the matrix block indices for the test
        matrix_indices = cc.create_matrix_block_indices(self.fake_rows_to_obs)
        # Calculate the probabilities for this test.
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        probs = cc.calc_probabilities(*args, **kwargs)
        # Get the matrix blocks for dP_i_dH_i
        matrix_blocks = cc.create_matrix_blocks(probs, matrix_indices)
        # Create the dP_dH matrix that represents the derivative of the
        # long probabilities with respect to the array of transformed index
        # values / systematic utilities
        dP_dH = block_diag(matrix_blocks)

        # Designate a function that calculates the parital derivative of the
        # transformed index values, with respect to the index.
        def transform_deriv_v(sys_utilities,
                              alt_IDs,
                              rows_to_alts,
                              shape_params):
            return diags(np.ones(sys_utilities.shape[0]), 0, format='csr')

        # Designate functions that calculate the partial derivative of the
        # transformed index values, with respect to shape and index parameters
        def transform_deriv_intercepts(sys_utilities,
                                       alt_IDs,
                                       rows_to_alts,
                                       intercept_params):
            return None

        fake_deriv = np.exp(self.fake_shapes)[None, :]
        def transform_deriv_shapes(sys_utilities,
                                   alt_IDs,
                                   rows_to_alts,
                                   shape_params):
            return rows_to_alts[:, 1:].multiply(fake_deriv)


        # Collect the arguments for the hessian function being tested
        hessian_args = [self.fake_betas,
                        self.fake_design,
                        self.fake_df[self.alt_id_col].values,
                        self.fake_rows_to_obs,
                        self.fake_rows_to_alts,
                        self.utility_transform,
                        transform_deriv_shapes,
                        transform_deriv_v,
                        transform_deriv_intercepts,
                        matrix_indices,
                        None,
                        self.fake_shapes,
                        None]

        # Calculate the derivative of the transformation vector with respect
        # to the shape parameters.
        args = (design.dot(self.fake_betas),
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_alts,
                self.fake_shapes)
        dh_d_shape = transform_deriv_shapes(*args)

        # Calculate the various matrices needed for the expected result
        # Note dH_dV is the Identity matrix in this test.
        # See the documentation pdf for a description of what each of these
        # matrixes are.
        # h_33 is -X^T * dP_dH * X. This is the hessian in the standard MNL
        h_33 = np.asarray(-1 * design.T.dot(dP_dH.dot(design)))
        # h_31 is -X^T * dH_dV^T * dP_dH * dH_d_shape
        h_31 = np.asarray(-1 * design.T.dot(dP_dH.dot(dh_d_shape.A)))
        # h_11 = -dH_d_shape^T * dP_dH * dH_d_shape
        h_11 = np.asarray(-1 * dh_d_shape.T.dot(dP_dH.dot(dh_d_shape.A)))

        # Create the final hessian
        top_row = np.concatenate((h_11, h_31.T), axis=1)
        bottom_row = np.concatenate((h_31, h_33), axis=1)
        expected_result = np.concatenate((top_row, bottom_row), axis=0)

        # Alias the function being tested
        func = cc.calc_hessian
        # Get the results of the function being tested
        function_result = func(*hessian_args)

        # Perform the required tests
        self.assertIsInstance(function_result, np.ndarray)
        self.assertEqual(function_result.shape, expected_result.shape)
        npt.assert_allclose(function_result, expected_result)

        return None
