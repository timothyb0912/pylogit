"""
Tests for the bootstrap_calcs.py file.
"""
import unittest

import numpy as np
import numpy.testing as npt
from scipy.stats import norm

import pylogit.bootstrap_calcs as bc

class ComputationalTests(unittest.TestCase):
    def setUp(self):
        # Determine the number of parameters and number of bootstrap replicates
        num_replicates = 100
        num_params = 5
        # Create a set of fake bootstrap replicates
        self.bootstrap_replicates =\
            (np.arange(1, 1 + num_replicates)[:, None] *
             np.arange(1, 1 + num_params)[None, :])
        # Create a set of fake jackknife replicates
        self.jackknife_replicates = None
        # Create a fake confidence percentage.
        self.conf_percentage = 94.88
        # Create a fake maximum likelihood parameter estimate
        self.mle_params = self.bootstrap_replicates[50, :]

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
        self.assertEqual(func_results.ndim, 2)
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
        self.assertEqual(func_result.ndim, 1)
        self.assertEqual(func_result.shape, expected_result.shape)
        npt.assert_allclose(func_result, expected_result)

        # Create a fake mle that should be higher than 95% of the results
        fake_mle = self.bootstrap_replicates[95]
        expected_result_2 = norm.ppf(0.95) * np.ones(self.mle_params.size)
        func_result_2 = func(self.bootstrap_replicates, fake_mle)

        self.assertIsInstance(func_result_2, np.ndarray)
        self.assertEqual(func_result_2.ndim, 1)
        self.assertEqual(func_result_2.shape, expected_result_2.shape)
        npt.assert_allclose(func_result_2, expected_result_2)
        return None
