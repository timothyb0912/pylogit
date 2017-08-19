"""
Tests for the bootstrap_sampler.py file.
"""
import unittest
import warnings
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pandas as pd

import pylogit.bootstrap_sampler as bs


class SamplerTests(unittest.TestCase):
    def test_relate_obs_ids_to_chosen_alts(self):
        # Create fake data for the observation, alternative, and choice ids.
        obs_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        alt_ids = np.array([1, 2, 1, 3, 2, 3, 2, 3, 1, 3, 1, 2])
        choices = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])

        # Create the dictionary that we expect the tested function to return
        expected_dict = {1: np.array([2, 6]),
                         2: np.array([1, 4]),
                         3: np.array([3, 5])}

        # Alias the function being tested.
        func = bs.relate_obs_ids_to_chosen_alts

        # Execute the given tests.
        func_result = func(obs_ids, alt_ids, choices)
        self.assertIsInstance(func_result, dict)
        for key in expected_dict:
            self.assertIn(key, func_result)
            self.assertIsInstance(func_result[key], np.ndarray)
            self.assertEqual(func_result[key].ndim, 1)
            npt.assert_allclose(func_result[key], expected_dict[key])
        return None

    def test_get_num_obs_choosing_each_alternative(self):
        # Alias the function that is to be tested
        func = bs.get_num_obs_choosing_each_alternative

        # Create the dictionary of observations per alternative
        obs_per_group = {1: np.array([2, 6, 7]),
                         2: np.array([1]),
                         3: np.array([3, 5])}

        # Get the 'expected results'
        expected_dict = OrderedDict()
        expected_dict[1] = obs_per_group[1].size
        expected_dict[2] = obs_per_group[2].size
        expected_dict[3] = obs_per_group[3].size
        expected_num_obs = (obs_per_group[1].size +
                            obs_per_group[2].size +
                            obs_per_group[3].size)

        # Get the results from the function
        func_dict, func_num_obs = func(obs_per_group)

        # Perform the desired tests
        self.assertIsInstance(func_dict, OrderedDict)
        self.assertIsInstance(func_num_obs, int)
        self.assertEqual(func_num_obs, expected_num_obs)
        for key in func_dict:
            func_num = func_dict[key]
            self.assertIsInstance(func_num, int)
            self.assertEqual(func_num, expected_dict[key])
        return None

    def test_create_cross_sectional_bootstrap_samples(self):
        # Create fake data for the observation, alternative, and choice ids.
        obs_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        alt_ids = np.array([1, 2, 1, 3, 2, 3, 2, 3, 1, 3, 1, 2])
        choices = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])

        # Determine the number of samples to be taken
        num_samples = 5

        # Determine the random seed for reproducibility
        seed = 55
        np.random.seed(seed)

        # Create the dictionary of observations per alternative
        obs_per_group = {1: np.array([2, 6]),
                         2: np.array([1, 4]),
                         3: np.array([3, 5])}
        num_obs_per_group = {1: 2, 2: 2, 3: 2}

        # Determine the array that should be created.
        expected_ids = np.empty((num_samples, 6))

        expected_shape_1 = (num_samples, num_obs_per_group[1])
        expected_ids[:, :2] =\
            np.random.choice(obs_per_group[1],
                             size=num_samples * num_obs_per_group[1],
                             replace=True).reshape(expected_shape_1)

        expected_shape_2 = (num_samples, num_obs_per_group[2])
        expected_ids[:, 2:4] =\
            np.random.choice(obs_per_group[2],
                             size=num_samples * len(obs_per_group[2]),
                             replace=True).reshape(expected_shape_2)

        expected_shape_3 = (num_samples, num_obs_per_group[3])
        expected_ids[:, 4:6] =\
            np.random.choice(obs_per_group[3],
                             size=num_samples * len(obs_per_group[3]),
                             replace=True).reshape(expected_shape_3)

        # Alias the function being tested.
        func = bs.create_cross_sectional_bootstrap_samples

        # Get the desired results
        func_result = func(obs_ids, alt_ids, choices, num_samples, seed=seed)

        # Perform the requisite tests
        self.assertIsInstance(func_result, np.ndarray)
        self.assertEqual(func_result.ndim, 2)
        self.assertEqual(func_result.shape, expected_ids.shape)
        npt.assert_allclose(func_result, expected_ids)

        return None

    def test_create_bootstrap_id_array(self):
        return None

    def test_create_deepcopied_groupby_dict(self):
        return None

    def test_create_bootstrap_dataframe(self):
        return None
