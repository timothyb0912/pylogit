"""
Tests for the bootstrap_utils.py file.
"""
import unittest

import numpy as np
import numpy.testing as npt

import pylogit.bootstrap_utils as bu

class UtilityTester(unittest.TestCase):
    def test_check_conf_percentage_validity(self):
        # Create a list of valid and invalid arguments
        good_args = [80, 95.0, 30L]
        bad_args = [-2, '95', None, (90,)]
        # Note the message that should be displayed in case of errors.
        expected_err_msg =\
            "conf_percentage MUST be a number between 0.0 and 100."
        # Alias the function being tested
        func = bu.check_conf_percentage_validity
        # Perform the desired tests
        for arg in good_args:
            self.assertIsNone(func(arg))
        for arg in bad_args:
            self.assertRaisesRegexp(ValueError,
                                    expected_err_msg,
                                    func,
                                    arg)
        return None

    def test_ensure_samples_is_2d_ndarray(self):
        # Create a list of valid and invalid arguments
        base_array = np.arange(10)
        good_args = [base_array.copy().reshape((2, 5)),
                     base_array.copy().reshape((5, 2))]
        bad_args = [base_array, base_array[None, None, :], 30]
        # Create a 'name' argument
        fake_name = 'test'
        # Note the message that should be displayed in case of errors.
        expected_err_msg =\
             "`{}` MUST be a 2D ndarray.".format(fake_name + '_samples')
        # Alias the function being tested
        func = bu.ensure_samples_is_2d_ndarray
        # Perform the desired tests
        for arg in good_args:
            self.assertIsNone(func(arg, name=fake_name))
        for arg in bad_args:
            self.assertRaisesRegexp(ValueError,
                                    expected_err_msg,
                                    func,
                                    arg,
                                    name=fake_name)
        return None

    def test_get_alpha_from_conf_percentage(self):
        # Create a list of valid confidence percentages
        good_args = [80, 95.0, 30L]
        # Create a list of expected results
        expected_results = [20, 5, 70]
        # Alias the function being tested
        func = bu.get_alpha_from_conf_percentage
        # Perform the desired tests
        for pos, arg in enumerate(good_args):
            self.assertEqual(func(arg), expected_results[pos])
        return None

    def test_combine_conf_endpoints(self):
        # Create fake arguments
        lower_array = np.arange(5)
        upper_array = np.arange(2, 7)
        # Create the expected result
        expected_result =\
            np.array([lower_array.tolist(), upper_array.tolist()])
        # Alias the function being tested
        func = bu.combine_conf_endpoints
        # Perform the desired test
        npt.assert_allclose(expected_result, func(lower_array, upper_array))
        return None
