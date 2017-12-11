"""
Tests for the bootstrap_calcs.py file.
"""
import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.stats import norm, gumbel_r
import scipy.stats

import pylogit.bootstrap_calcs as bc

class ComputationalTests(unittest.TestCase):
    def setUp(self):
        """
        Note that the spatial test data used in many of these tests comes from
        Efron, Bradley, and Robert J. Tibshirani. An Introduction to the
        Bootstrap. CRC press, 1994. Chapter 14.
        """
        # Determine the number of parameters and number of bootstrap replicates
        num_replicates = 100
        num_params = 5
        # Create a set of fake bootstrap replicates
        self.bootstrap_replicates =\
            (np.arange(1, 1 + num_replicates)[:, None] *
             np.arange(1, 1 + num_params)[None, :])
        # Create a fake maximum likelihood parameter estimate
        self.mle_params = self.bootstrap_replicates[50, :]
        # Create a set of fake jackknife replicates
        array_container = []
        for est in self.mle_params:
            array_container.append(gumbel_r.rvs(loc=est, size=10))
        self.jackknife_replicates =\
            np.concatenate([x[:, None] for x in array_container], axis=1)
        # Create a fake confidence percentage.
        self.conf_percentage = 94.88

        # Store the spatial test data from Efron and Tibshirani (1994)
        self.test_data =\
            np.array([48, 36, 20, 29, 42, 42, 20, 42, 22, 41, 45, 14, 6,
                      0, 33, 28, 34, 4, 32, 24, 47, 41, 24, 26, 30, 41])

        # Note how many test data observations there are.
        num_test_obs = self.test_data.size

        # Create the function to calculate the jackknife replicates.
        def calc_theta(array):
            result = ((array - array.mean())**2).sum() / float(array.size)
            return result
        self.calc_theta = calc_theta
        self.test_theta_hat = np.array([calc_theta(self.test_data)])

        # Create a pandas series of the data. Allows for easy case deletion.
        raw_series = pd.Series(self.test_data)
        # Create the array of jackknife replicates
        jackknife_replicates = np.empty((num_test_obs, 1), dtype=float)
        for obs in xrange(num_test_obs):
            current_data = raw_series[raw_series.index != obs].values
            jackknife_replicates[obs] = calc_theta(current_data)
        self.test_jackknife_replicates = jackknife_replicates

        return None

    def test_calc_percentile_interval(self):
        # Get the alpha percentage. Should be 5.12 so alpha / 2 should be 2.56
        alpha = bc.get_alpha_from_conf_percentage(self.conf_percentage)
        # These next 2 statements work because there are exactly 100 replicates
        # We should have the value in BR[lower_row, 0] = 3 so that there are 2
        # elements in bootstrap_replicates (BR) that are less than this. I.e.
        # we want lower_row = 2. Note 2.56 rounded down is 2.
        lower_row = np.floor(alpha / 2.0)
        # 100 - 2.56 is 97.44. Rounded up, this is 98.
        # We want the row such that the value in the first column of that row
        # is 98, i.e. we want the row at index 97.
        upper_row = np.floor(100 - (alpha / 2.0))
        # Create the expected results
        expected_results =\
            bc.combine_conf_endpoints(self.bootstrap_replicates[lower_row],
                                      self.bootstrap_replicates[upper_row])
        # Alias the function being tested
        func = bc.calc_percentile_interval
        # Get the function results
        func_results =func(self.bootstrap_replicates, self.conf_percentage)
        # Perform the desired tests
        self.assertIsInstance(func_results, np.ndarray)
        self.assertEqual(func_results.shape, expected_results.shape)
        npt.assert_allclose(func_results, expected_results)
        return None

    def test_calc_bias_correction_bca(self):
        # There are 100 bootstrap replicates, already in ascending order for
        # each column. If we take row 51 to be the mle, then 50% of the
        # replicates are less than the mle, and we should have bias = 0.
        expected_result = np.zeros(self.mle_params.size)

        # Alias the function to be tested.
        func = bc.calc_bias_correction_bca

        # Perform the desired test
        func_result = func(self.bootstrap_replicates, self.mle_params)
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)

        # Create a fake mle that should be higher than 95% of the results
        fake_mle = self.bootstrap_replicates[95]
        expected_result_2 = norm.ppf(0.95) * np.ones(self.mle_params.size)
        func_result_2 = func(self.bootstrap_replicates, fake_mle)

        self.assertIsInstance(func_result_2, np.ndarray)
        self.assertEqual(func_result_2.shape, expected_result_2.shape)
        npt.assert_allclose(func_result_2, expected_result_2)
        return None

    def test_calc_acceleration_bca(self):
        # Get the expected result. See page 186 of Efron and Tibshirani (1994)
        expected_result = np.array([0.061])

        # Alias the function being tested
        func = bc.calc_acceleration_bca

        # Perform the desired test
        func_result = func(self.test_jackknife_replicates)
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        # Note the absolute tolerance of 5e-4 is used because the results
        # should agree when rounded to 3 decimal places. This will be the case
        # if the two sets of results agree to within 5e-4 of each other.
        npt.assert_allclose(func_result, expected_result, atol=5e-4)
        return None

    def test_calc_lower_bca_percentile(self):
        # Use the parameter values from
        # Efron, Bradley, and Robert J. Tibshirani. An Introduction to the
        # Bootstrap. CRC press, 1994. Pages 185-186
        # Note that my alpha is Efron's alpha / 2, in percents not decimals
        alpha_percent = 10
        bias_correction = np.array([0.146])
        acceleration = np.array([0.061])

        # Note the expected results
        expected_result = np.array([0.110])

        # Alias the function being tested
        func = bc.calc_lower_bca_percentile

        # Perform the desired tests
        # Note we divide the function results by 100 since our results are in
        # terms of percents and Efron's results are in decimals.
        func_result = func(alpha_percent, bias_correction, acceleration) / 100
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        # Note the absolute tolerance of 5e-4 is used because the results
        # should agree when rounded to 3 decimal places. This will be the case
        # if the two sets of results agree to within 5e-4 of each other.
        npt.assert_allclose(func_result, expected_result, atol=5e-4)
        return None

    def test_calc_upper_bca_percentile(self):
        # Use the parameter values from
        # Efron, Bradley, and Robert J. Tibshirani. An Introduction to the
        # Bootstrap. CRC press, 1994. Pages 185-186
        # Note that my alpha is Efron's alpha / 2, in percents not decimals
        alpha_percent = 10
        bias_correction = np.array([0.146])
        acceleration = np.array([0.061])

        # Note the expected results
        expected_result = np.array([0.985])

        # Alias the function being tested
        func = bc.calc_upper_bca_percentile

        # Perform the desired tests
        # Note we divide the function results by 100 since our results are in
        # terms of percents and Efron's results are in decimals.
        func_result = func(alpha_percent, bias_correction, acceleration) / 100
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        # Note the absolute tolerance of 1e-3 is used because the results
        # should be within 0.001 of each other.
        npt.assert_allclose(func_result, expected_result, atol=1e-3)
        return None

    def test_calc_bca_interval(self):
        # Create the bootstrap replicates for the test data
        num_test_reps = 5000
        num_test_obs = self.test_data.size
        test_indices = np.arange(num_test_obs)
        boot_indx_shape = (num_test_reps, num_test_obs)
        np.random.seed(8292017)
        boot_indices =\
            np.random.choice(test_indices,
                             replace=True,
                             size=num_test_obs*num_test_reps)
        self.test_bootstrap_replicates =\
            np.fromiter((self.calc_theta(self.test_data[x]) for x in
                         boot_indices.reshape(boot_indx_shape)),
                        dtype=float)[:, None]

        # Note the expected result. See page 183 of Efron and Tibshirani (1994)
        expected_result = np.array([[115.8], [259.6]])

        # Bundle the necessary arguments
        args = [self.test_bootstrap_replicates,
                self.test_jackknife_replicates,
                self.test_theta_hat,
                90]

        # Alias the function being tested
        func = bc.calc_bca_interval

        # Get the function results
        func_result = func(*args)

        # Perform the desired tests
        # Note we divide the function results by 100 since our results are in
        # terms of percents and Efron's results are in decimals.
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.shape, expected_result.shape)
        # Note the relative tolerance of 0.01 is used because the function
        # results should be within 1% of the expected result. Note that some
        # differences are expected due to simulation error on both the part of
        # Efron and Tibshirani (1994) when they reported their results, and on
        # our part when calculating the results.
        npt.assert_allclose(func_result, expected_result, rtol=0.01)
        return None
