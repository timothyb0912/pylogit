"""
Tests for the bootstrap_abc.py file.
"""
import unittest
from collections import Iterable

import numpy as np
import numpy.testing as npt
import pandas as pd
import scipy.stats
import scipy.sparse

import pylogit.bootstrap_abc as abc


class HelperFuncTests(unittest.TestCase):
    def test_ensure_model_obj_has_mapping_constructor(self):
        class GoodModel(object):
            def get_mappings_for_fit(self):
                return None

        class BadModel(object):
            def nonsense_func(self):
                return None

        # Alias the function being tested
        func = abc.ensure_model_obj_has_mapping_constructor

        # Create the arguments for the test
        good_obj = GoodModel()
        bad_obj = BadModel()
        err_msg = "model_obj MUST have a 'get_mappings_for_fit' method."

        # Perform the desired tests
        self.assertIsNone(func(good_obj))
        self.assertRaisesRegexp(ValueError, err_msg, func, bad_obj)
        return None

    def test_ensure_rows_to_obs_validity(self):
        # Alias the function being tested
        func = abc.ensure_rows_to_obs_validity

        # Create the arguments for the test
        good_objects = [scipy.sparse.eye(3, format='csr', dtype=int), None]
        bad_objects = [scipy.sparse.eye(3, format='csc', dtype=int), np.eye(3)]
        err_msg = "rows_to_obs MUST be a 2D scipy sparse row matrix."

        # Perform the desired tests
        for good_obj in good_objects:
            self.assertIsNone(func(good_obj))
        for bad_obj in bad_objects:
            self.assertRaisesRegexp(ValueError, err_msg, func, bad_obj)
        return None

    def test_ensure_wide_weights_is_1D_or_2D_ndarray(self):
        # Alias the function being tested
        func = abc.ensure_wide_weights_is_1D_or_2D_ndarray

        # Create the arguments for the test
        good_objects =\
            [np.arange(4).reshape((2,2)), np.arange(3)]
        bad_objects = [None, np.arange(3)[None, None, :, None]]
        err_msgs = ["wide_weights MUST be a ndarray.",
                    "wide_weights MUST be a 1D or 2D ndarray."]

        # Perform the desired tests
        for good_obj in good_objects:
            self.assertIsNone(func(good_obj))
        for pos, bad_obj in enumerate(bad_objects):
            err_msg = err_msgs[pos]
            self.assertRaisesRegexp(ValueError, err_msg, func, bad_obj)
        return None


class ComputationalTests(unittest.TestCase):
    # Create various attributes on the ComputationalTests instance.
    # Store the spatial test data from Efron and Tibshirani (1994)
    test_data =\
        np.array([48, 36, 20, 29, 42, 42, 20, 42, 22, 41, 45, 14, 6,
                  0, 33, 28, 34, 4, 32, 24, 47, 41, 24, 26, 30, 41],
                 dtype=float)
    test_data_mean = test_data.mean()
    rows_to_obs = scipy.sparse.eye(test_data.size, format='csr', dtype=int)
    num_obs = test_data.size

    def calc_theta(self, weight):
        """
        See Equation 14.22 of Efron and Tibshirani (1994).
        """
        a_mean = weight.dot(self.test_data)
        differences = (self.test_data - a_mean)
        squared_diffs = differences**2
        return weight.dot(squared_diffs)

    def test_create_long_form_weights(self):
        # Create fake arguments for the test
        fake_rows_to_obs =\
            scipy.sparse.csr_matrix([[1, 0, 0],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 1, 0],
                                     [0, 0, 1],
                                     [0, 0, 1]])
        class FakeModel(object):
            def get_mappings_for_fit(self):
                return {'rows_to_obs': fake_rows_to_obs}

        fake_model_obj = FakeModel()

        fake_weights_1D = np.arange(1, 4)
        fake_weights_2D = np.array([[1, 2],
                                    [2, 4],
                                    [3, 6]])
        fake_weights = [fake_weights_1D, fake_weights_2D]

        # Create the expected results
        expected_result_1D = np.array([1, 1, 2, 2, 3, 3])
        expected_result_2D = np.array([[1, 2],
                                       [1, 2],
                                       [2, 4],
                                       [2, 4],
                                       [3, 6],
                                       [3, 6]])
        expected_results = [expected_result_1D, expected_result_2D]

        # Alias the function being tested
        func = abc.create_long_form_weights

        # Perform the desired tests
        for pos, weights in enumerate(fake_weights):
            func_array = func(fake_model_obj, weights)
            expected_array = expected_results[pos]
            self.assertIsInstance(func_array, np.ndarray)
            self.assertTrue(func_array.shape, expected_array.shape)
            npt.assert_allclose(func_array, expected_array)

        new_result =\
            func(fake_model_obj, fake_weights_1D, rows_to_obs=fake_rows_to_obs)
        self.assertIsInstance(new_result, np.ndarray)
        self.assertTrue(new_result.shape, expected_results[0].shape)
        npt.assert_allclose(new_result, expected_results[0])
        return None

    def test_calc_finite_diff_terms_for_abc(self):
        # Determine how many observations and how many parameters there will be
        num_obs = 3
        num_params = 4

        # Create a fake rows_to_obs mapping matrix for this test
        fake_rows_to_obs = scipy.sparse.eye(num_obs, format='csr', dtype=int)

        # Create an array that will be used in our implementation of a fake
        # T(P) function.
        ones_array = np.ones(num_obs)


        # Create a fake model class that will implement the T(P) function
        # through it's fit_mle method.
        class FakeModel(object):
            def __init__(self):
                # Create needed attributes to successfully mock an MNDC_Model
                #instance in this test
                self.data = pd.Series(np.arange(3))
                self.obs_id_col = np.arange(3, dtype=int)

            # Create a get_mappings_for_fit function that will allow for
            # successful mocking in this test
            def get_mappings_for_fit(self):
                return {"rows_to_obs": fake_rows_to_obs}

            # Create a fake T(P) function that will be used to verify the
            # correctness of calc_finite_diff_terms_for_abc
            def fit_mle(self, init_vals, weights=ones_array, **kwargs):
                max_pos = weights.argmax()
                if (max_pos % 2) == 0:
                    mle = np.array([weights.max(),
                                    weights.max(),
                                    weights.min(),
                                    weights.min()])
                else:
                    mle = np.array([weights.max(),
                                    weights.min(),
                                    weights.max(),
                                    weights.min()])
                return {'x': mle}

        # Create the fake epsilon that will be used in our test
        fake_epsilon = 0.01

        # Create the expected results
        plus_min = (1 - fake_epsilon) / num_obs
        plus_max = plus_min + fake_epsilon
        expected_plus =\
            (np.array([[plus_max, plus_max, plus_min, plus_min],
                       [plus_max, plus_min, plus_max, plus_min],
                       [plus_max, plus_max, plus_min, plus_min]]) *
             ones_array[:, None])

        minus_max = (1 + fake_epsilon) / num_obs
        minus_min = minus_max - fake_epsilon
        expected_minus =\
            (np.array([[minus_max, minus_min, minus_max, minus_min],
                       [minus_max, minus_max, minus_min, minus_min],
                       [minus_max, minus_max, minus_min, minus_min]]) *
             ones_array[:, None])
        expected_result = (expected_plus, expected_minus)

        # Create the remaining fake arguments needed for the test.
        fake_model_obj = FakeModel()
        fake_mle = np.ones(num_params)
        fake_init_vals = fake_mle

        # Alias the function being tested
        func = abc.calc_finite_diff_terms_for_abc

        # Compute the functon result
        func_result = func(fake_model_obj,
                           fake_mle,
                           fake_init_vals,
                           fake_epsilon)

        # Perform the desired tests
        self.assertIsInstance(func_result, Iterable)
        self.assertEqual(len(func_result), 2)
        for pos, func_array in enumerate(func_result):
            expected_array = expected_result[pos]
            self.assertIsInstance(func_array, np.ndarray)
            self.assertTrue(func_array.shape, expected_array.shape)
            npt.assert_allclose(func_array, expected_array)
        return None

    def test_calc_empirical_influence_abc(self):
        # Create the expected result
        expected_result = np.tile(np.arange(1, 5), 3).reshape((4, 3))
        # Create fake arguments to produce the expected result
        fake_epsilon = 0.001
        fake_term_plus =\
            fake_epsilon * np.tile(np.arange(1, 5), 3).reshape((4, 3))
        fake_term_minus = -1 * fake_term_plus
        # Alias the function being tested
        func = abc.calc_empirical_influence_abc
        # Calculate the function result
        func_result = func(fake_term_plus, fake_term_minus, fake_epsilon)
        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        return None

    def test_calc_2nd_order_influence_abc(self):
        # Create fake arguments for the test
        fake_term_minus = np.arange(-3, 0)[None, :] * np.ones(3)[:, None]
        fake_term_plus =\
            np.arange(1, -2, step=-1)[None, :] * np.ones(3)[:, None]
        fake_mle = np.zeros(3)
        fake_epsilon = 0.1

        # Create the expected result
        expected_result = -200 * np.ones((3, 3))

        # Alias the function being tested
        func = abc.calc_2nd_order_influence_abc

        # Calculate the function result
        func_result =\
            func(fake_mle, fake_term_plus, fake_term_minus, fake_epsilon)

        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        return None

    def test_calc_influence_arrays_for_abc(self):
        return None

    def test_calc_std_error_abc(self):
        # Create a fake empirical influence function
        fake_empirical_influence =\
            np.array([-1, -2, 1, 2])[:, None] * np.arange(1, 5)[None, :]
        # Calculate the expected result
        num_obs = float(fake_empirical_influence.shape[0])
        expected_result =\
            (np.var(fake_empirical_influence, axis=0) / num_obs)**0.5
        # Alias the function being tested
        func = abc.calc_std_error_abc
        # Calcuate the function result
        func_result = func(fake_empirical_influence)
        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        return None

    def test_calc_acceleration_abc(self):
        # Create a fake empirical influence matrix
        np.random.seed(8302017)
        fake_empirical_influence =\
            np.array([-1, -2, -3, 6])[:, None] * np.arange(1, 5)[None, :]
        # Calculate the expected result. Note that these formulas are derived
        # from analytically writing out the skewness formula and then solving
        # for the acceleration.
        num_obs = fake_empirical_influence.shape[0]
        denom = 6 * num_obs**0.5
        expected_result =\
            scipy.stats.skew(fake_empirical_influence, axis=0) / denom
        # Alias the function being tested
        func = abc.calc_acceleration_abc
        # Calculate the function result
        func_result = func(fake_empirical_influence)
        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        return None

    def test_calc_bias_abc(self):
        # Create a second order influence array
        fake_second_order_influence = np.arange(12).reshape((3, 4))
        # Note the expected results of the test.
        denominator = float(2 * (fake_second_order_influence.shape[0])**2)
        numerator = fake_second_order_influence.sum(axis=0)
        expected_result = numerator / denominator
        # Alias the function being tested.
        func = abc.calc_bias_abc
        # Calcuate the function result
        func_result = func(fake_second_order_influence)
        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        return None

    def test_calc_quadratic_coef_abc(self):
        # Create fake arguments for the test
        fake_data = np.arange(1, 5)
        num_obs = fake_data.size
        fake_epsilon = 0.01
        fake_rows_to_obs = scipy.sparse.eye(num_obs, format='csr', dtype=int)
        t_func = lambda p: np.array([p.dot(fake_data)])
        init_weights = np.ones(num_obs) / float(num_obs)
        mle_est = t_func(init_weights)
        fake_std_error = np.array([2])
        fake_empirical_influence =\
            8 * np.array([-1, 1, 1, -1], dtype=float)[:, None]
        fake_std_influence =\
            fake_empirical_influence / (num_obs**2 * fake_std_error)

        # Create a fake model class that will implement the T(P) function
        # through it's fit_mle method.
        class FakeModel(object):
            def __init__(self):
                # Create needed attributes to successfully mock an MNDC_Model
                #instance in this test
                self.data = pd.Series(fake_data)
                self.obs_id_col = np.arange(num_obs, dtype=int)

            # Create a get_mappings_for_fit function that will allow for
            # successful mocking in this test
            def get_mappings_for_fit(self):
                return {"rows_to_obs": fake_rows_to_obs}

            # Use the T(P) function from the spatial test data example.
            def fit_mle(self,
                        init_vals,
                        weights=init_weights,
                        **kwargs):
                return {'x': np.array([t_func(weights)])}
        fake_model_obj = FakeModel()

        # Calculate the expected result. Note that I specifically chose values
        # such that the expected result would be -500
        weight_1 = ((1 - fake_epsilon) * init_weights[:, None] +
                    fake_epsilon * fake_std_influence)
        weight_3 = ((1 - fake_epsilon) * init_weights[:, None] -
                    fake_epsilon * fake_std_influence)
        term_1 = np.array([weight_1.ravel().dot(fake_data)])
        term_3 = np.array([weight_3.ravel().dot(fake_data)])
        expected_result = ((term_1 - 2 * mle_est + term_3) /
                           (fake_epsilon**2))

        # Alias the function being tested
        func = abc.calc_quadratic_coef_abc

        # Calculate the function result
        func_result = func(fake_model_obj,
                           mle_est,
                           mle_est,
                           fake_empirical_influence,
                           fake_std_error,
                           fake_epsilon)

        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        npt.assert_allclose(func_result[0], -500)
        return None

    def test_efron_quadratic_coef_abc(self):
        # Create fake arguments for the test
        fake_data = np.array([1, 5, 10, 20], dtype=float)
        num_obs = fake_data.size
        fake_epsilon = 0.01
        fake_rows_to_obs = scipy.sparse.eye(num_obs, format='csr', dtype=int)
        t_func = lambda p: np.array([(p**-1).dot(fake_data**-1)])
        init_weights = np.ones(num_obs) / float(num_obs)
        mle_est = t_func(init_weights)
        fake_std_error = np.array([2])
        fake_empirical_influence =\
            160 * np.array([1, 1, 1, 1], dtype=float)[:, None]
        fake_std_influence =\
            fake_empirical_influence / (num_obs**2 * fake_std_error)

        # Create a fake model class that will implement the T(P) function
        # through it's fit_mle method.
        class FakeModel(object):
            def __init__(self):
                # Create needed attributes to successfully mock an MNDC_Model
                #instance in this test
                self.data = pd.Series(fake_data)
                self.obs_id_col = np.arange(num_obs, dtype=int)

            # Create a get_mappings_for_fit function that will allow for
            # successful mocking in this test
            def get_mappings_for_fit(self):
                return {"rows_to_obs": fake_rows_to_obs}

            # Use the T(P) function from the spatial test data example.
            def fit_mle(self,
                        init_vals,
                        weights=init_weights,
                        **kwargs):
                return {'x': np.array([t_func(weights)])}
        fake_model_obj = FakeModel()

        # Calculate the expected result. Note that I specifically chose values
        # such that the expected result would be 1,125
        expected_result = np.array([1125])

        # Alias the function being tested
        func = abc.efron_quadratic_coef_abc

        # Calculate the function result
        func_result = func(fake_model_obj,
                           mle_est,
                           mle_est,
                           fake_empirical_influence,
                           fake_std_error,
                           fake_epsilon)

        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        npt.assert_allclose(func_result[0], 1125)
        return None

    def test_calc_total_curvature_abc(self):
        # Create fake arguments
        fake_bias = np.array([3, 6, 9, 12], dtype=float)
        fake_std_error = np.array([2, 4, 6, 8], dtype=float)
        fake_quadratic_coef = 1.5 * np.ones(fake_bias.size)
        # Create the expected result
        expected_result = np.zeros(fake_bias.size)
        # Alias the function being tested.
        func = abc.calc_total_curvature_abc
        # Calculate the function result
        func_result = func(fake_bias, fake_std_error, fake_quadratic_coef)
        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        return None

    def test_calc_bias_correction_abc(self):
        # Create fake arguments for the test
        num_params = 3
        ones_array = np.ones(num_params)
        fake_acceleration = -1.645 * ones_array
        fake_total_curvature = np.zeros(num_params)

        # Create the expected results
        norm_dist = scipy.stats.norm
        # Note the expected result is the acceleration because the standard
        # normal cdf of zero is 0.5, 0.5 * 2 is 1.0, and the inverse cdf of
        # the cdf of acceleration is acceleration
        expected_result = fake_acceleration
        # Alias the function being tested
        func = abc.calc_bias_correction_abc

        # Calculate the function result
        func_result = func(fake_acceleration, fake_total_curvature)

        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)

        # Perform a second test where we reverse the arrays
        func_result_2 = func(fake_total_curvature, fake_acceleration)
        expected_result_2 = -1 * fake_acceleration
        self.assertIsInstance(func_result_2, np.ndarray)
        self.assertEqual(func_result_2.shape, expected_result_2.shape)
        npt.assert_allclose(func_result_2, expected_result_2)
        return None

    def test_calc_endpoint_from_percentile_abc(self):
        # Create the fake arguments for the test
        num_obs = 3
        num_params = 4
        fake_percentile = 97.5
        ones_array_params = np.ones(num_params)
        ones_array_obs = np.ones(num_obs)
        norm = scipy.stats.norm
        fake_bias_correction =\
            (1 - norm.ppf(fake_percentile * 0.01)) * ones_array_params
        fake_acceleration = 0.5 * ones_array_params
        fake_std_error = 2 * ones_array_params
        fake_empirical_influence =\
            ((1.0 / 6) * np.array([2.0, 5.0, 8.0, 11.0])[None, :] *
             np.array([0.25, 0.5, 1.0])[:, None])

        # Create a fake rows_to_obs mapping matrix for this test
        fake_rows_to_obs = scipy.sparse.eye(num_obs, format='csr', dtype=int)

        # Create a fake model class that will implement the T(P) function
        # through it's fit_mle method.
        class FakeModel(object):
            # Create a get_mappings_for_fit function that will allow for
            # successful mocking in this test
            def get_mappings_for_fit(self):
                return {"rows_to_obs": fake_rows_to_obs}

            # Create a fake T(P) function that will be used to verify the
            # correctness of calc_endpoint_from_percentile_abc
            def fit_mle(self, init_vals, weights=ones_array_obs, **kwargs):
                mle = weights.max() * ones_array_params
                return {'x': mle}
        fake_model_obj = FakeModel()

        # Create the expected result array. Note that the fake arguments and
        # fake model object are designed to yield a endpoint of np.arange(1, 5)
        expected_result = np.arange(1, 5)

        # Alias the function being tested
        func = abc.calc_endpoint_from_percentile_abc

        # Calculate the function result
        args = [fake_model_obj,
                ones_array_params,
                fake_percentile,
                fake_bias_correction,
                fake_acceleration,
                fake_std_error,
                fake_empirical_influence]
        func_result = func(*args)

        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        return None

    def test_efron_endpoint_from_percentile_abc(self):
        # Create the fake arguments for the test
        num_obs = 3
        num_params = 4
        fake_percentile = 97.5
        ones_array_params = np.ones(num_params)
        ones_array_obs = np.ones(num_obs)
        norm = scipy.stats.norm
        fake_bias_correction =\
            (1 - norm.ppf(fake_percentile * 0.01)) * ones_array_params
        fake_acceleration = 0.5 * ones_array_params
        fake_std_error = 2 * ones_array_params
        fake_empirical_influence =\
            ((9.0 / 6) * np.array([2.0, 5.0, 8.0, 11.0])[None, :] *
             np.array([0.25, 0.5, 1.0])[:, None])

        # Create a fake rows_to_obs mapping matrix for this test
        fake_rows_to_obs = scipy.sparse.eye(num_obs, format='csr', dtype=int)

        # Create a fake model class that will implement the T(P) function
        # through it's fit_mle method.
        class FakeModel(object):
            # Create a get_mappings_for_fit function that will allow for
            # successful mocking in this test
            def get_mappings_for_fit(self):
                return {"rows_to_obs": fake_rows_to_obs}

            # Create a fake T(P) function that will be used to verify the
            # correctness of calc_endpoint_from_percentile_abc
            def fit_mle(self, init_vals, weights=ones_array_obs, **kwargs):
                mle = weights.max() * ones_array_params
                return {'x': mle}
        fake_model_obj = FakeModel()

        # Create the expected result array. Note that the fake arguments and
        # fake model object are designed to yield a endpoint of np.arange(1, 5)
        expected_result = np.arange(1, 5)

        # Alias the function being tested
        func = abc.efron_endpoint_from_percentile_abc

        # Calculate the function result
        args = [fake_model_obj,
                ones_array_params,
                fake_percentile,
                fake_bias_correction,
                fake_acceleration,
                fake_std_error,
                fake_empirical_influence]
        func_result = func(*args)

        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)
        return None

    def test_calc_endpoints_for_abc_confidence_interval(self):
        return None

    def test_efron_endpoints_for_abc_confidence_interval(self):
        return None

    def test_calc_abc_interval(self):
        # Create local versions of attributes on this testcase instance
        fake_rows_to_obs = self.rows_to_obs
        t_func = self.calc_theta
        fake_data = self.test_data
        init_weights = np.ones(self.num_obs, dtype=float)/self.num_obs

        # Create a fake model class that will implement the T(P) function
        # through it's fit_mle method.
        class FakeModel(object):
            def __init__(self):
                # Create needed attributes to successfully mock an MNDC_Model
                #instance in this test
                self.data = pd.Series([pos for pos, x in enumerate(fake_data)])
                self.obs_id_col = np.arange(fake_data.size, dtype=int)

            # Create a get_mappings_for_fit function that will allow for
            # successful mocking in this test
            def get_mappings_for_fit(self):
                return {"rows_to_obs": fake_rows_to_obs}

            # Use the T(P) function from the spatial test data example.
            def fit_mle(self,
                        init_vals,
                        weights=init_weights,
                        **kwargs):
                return {'x': np.array([t_func(weights)])}
        fake_model_obj = FakeModel()

        # Create the remaining arguments needed for the test.
        mle_est = np.array([t_func(init_weights)])
        fake_init_vals = mle_est
        conf_percentage = 90

        # Create the expected result. Note this is the ABC non-parametric
        # interval from Efron and Tibshirani (1994) p.183.
        expected_result = np.array([[116.7], [260.9]])

        # Alias the function being tested
        func = abc.calc_abc_interval

        # Calculate the function result
        func_result =\
            func(fake_model_obj, mle_est, fake_init_vals, conf_percentage)

        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result, atol=0.1, rtol=0)
        return None
