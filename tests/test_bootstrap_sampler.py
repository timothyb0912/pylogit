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
        self.assertEqual(func_result.shape, expected_ids.shape)
        npt.assert_allclose(func_result, expected_ids)

        # Make sure the argument check works
        self.assertRaises(ValueError,
                          func,
                          obs_ids,
                          alt_ids,
                          choices,
                          num_samples,
                          "2")

        return None

    def test_create_bootstrap_id_array(self):
        # Create an array of fake bootstrapped observation ids
        fake_obs_id_per_sample = np.arange(25).reshape((5, 5))

        # Create the expected result denoting the "bootstrap ids" for each of
        # the sampled observation ids.
        expected_results = np.array([[1, 2, 3, 4, 5],
                                     [1, 2, 3, 4, 5],
                                     [1, 2, 3, 4, 5],
                                     [1, 2, 3, 4, 5],
                                     [1, 2, 3, 4, 5]])
        # Alias the function being tested
        func = bs.create_bootstrap_id_array
        # Get the function results
        func_result = func(fake_obs_id_per_sample)

        # Perform the desired tests
        self.assertIsInstance(func_result, np.ndarray)
        npt.assert_allclose(func_result, expected_results)

        return None

    def test_create_deepcopied_groupby_dict(self):
        # Create the dataframe of fake data
        fake_df = pd.DataFrame({"obs_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                                "alt_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                "choice": [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                                "x": [1, 1.2, 1.4, 0.3, 0.9, 1.11, 0.53, 0.82,
                                      1.31, 1.24, 0.98, 0.76]})
        # Create the result that we expect from the function being tested.
        expected_res = {1: fake_df.iloc[0:2],
                        2: fake_df.iloc[2:4],
                        3: fake_df.iloc[4:6],
                        4: fake_df.iloc[6:8],
                        5: fake_df.iloc[8:10],
                        6: fake_df.iloc[10:]}
        # Alias the function being tested
        func = bs.create_deepcopied_groupby_dict

        # Get the result of the function
        func_result = func(fake_df, "obs_id")

        # Perform the requisite tests
        # Ensure the returned value is a dictionary
        self.assertIsInstance(func_result, dict)
        # Ensure the returned value and the expected value have the same keys.
        self.assertEqual(sorted(func_result.keys()),
                         sorted(expected_res.keys()))
        for key in func_result:
            # Get the expected and returned dataframes for each observation id
            sub_func_result = func_result[key]
            sub_expected_res = expected_res[key]

            # Ensure that the dataframes have equal values.
            npt.assert_allclose(sub_func_result.values,
                                sub_expected_res.values)

            # Ensure the dataframes don't share the same location in memory.
            self.assertNotEqual(id(sub_func_result), id(sub_expected_res))
        return None

    def test_check_column_existence(self):
        # Create the fake dataframe for the test.
        fake_df = pd.DataFrame({"obs_id": [1, 1, 2, 2, 3, 3],
                                "alt_id": [1, 2, 1, 2, 1, 2],
                                "choice": [1, 0, 0, 1, 1, 0]})
        # Create the sets of arguments and keyword arguments that should not
        # lead to raising errors.
        good_cols = ["obs_id", "boot_id"]
        good_kwargs = [{"presence": True}, {"presence": False}]

        # Alias the function that is being tested
        func = bs.check_column_existence

        # Perform the desired tests.
        for pos in xrange(len(good_cols)):
            col = good_cols[pos]
            current_good_kwargs = good_kwargs[pos]
            current_bad_kwargs =\
                {"presence": bool(1 - current_good_kwargs["presence"])}
            pattern = ("Ensure that `{}` is ".format(col) +
                       "not " * (1 - current_bad_kwargs["presence"]) +
                       "in `df.columns`.")

            self.assertIsNone(func(col, fake_df, **current_good_kwargs))
            self.assertRaisesRegexp(ValueError,
                                    pattern,
                                    func,
                                    col,
                                    fake_df,
                                    **current_bad_kwargs)

        return None

    def test_ensure_resampled_obs_ids_in_df(self):
        # Create fake data for the test.
        good_resampled_obs_ids = np.array([1, 1, 4, 3, 4])
        bad_resampled_obs_ids = np.array([1, 1, 4, 3, 8])
        fake_orig_obs_ids = np.arange(1, 6)

        # Expected error msg pattern
        expected_err_msg =\
            "All values in `resampled_obs_ids` MUST be in `orig_obs_id_array`."

        # Alias the function being tested.
        func = bs.ensure_resampled_obs_ids_in_df

        # Perform the desired tests
        self.assertIsNone(func(good_resampled_obs_ids, fake_orig_obs_ids))
        self.assertRaisesRegexp(ValueError,
                                expected_err_msg,
                                func,
                                bad_resampled_obs_ids,
                                fake_orig_obs_ids)
        return None

    def test_create_bootstrap_dataframe(self):
        # Create the dataframe of fake data
        fake_df = pd.DataFrame({"obs_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                                "alt_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                                "choice": [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                                "x": [1, 1.2, 1.4, 0.3, 0.9, 1.11, 0.53, 0.82,
                                      1.31, 1.24, 0.98, 0.76]})
        # Note the observation id column
        obs_id_col = "obs_id"

        # Get the bootstrapped samples of the observation ids
        sampling_args = [fake_df["obs_id"].values,
                         fake_df["alt_id"].values,
                         fake_df["choice"].values,
                         5]
        sampled_obs_ids =\
            bs.create_cross_sectional_bootstrap_samples(*sampling_args)
        rel_sampled_ids = sampled_obs_ids[0, :]

        # Get the groupby dictionary for this dataframe.
        groupby_dictionary =\
            bs.create_deepcopied_groupby_dict(fake_df, obs_id_col)

        # Alias the function necessary to create the bootstrap dataframe
        func = bs.create_bootstrap_dataframe
        # Create the bootstrap id column name
        boot_id_col = "new_id"

        # Create the expected result.
        expected_result =\
            [groupby_dictionary[obs_id].copy() for obs_id in rel_sampled_ids]
        for pos in xrange(len(expected_result)):
            expected_result[pos][boot_id_col] = pos + 1
        expected_result = pd.concat(expected_result, axis=0, ignore_index=True)

        # Get the function result
        func_result = func(fake_df,
                           obs_id_col,
                           rel_sampled_ids,
                           groupby_dictionary,
                           boot_id_col=boot_id_col)

        # Perform the desired tests.
        self.assertIsInstance(func_result, pd.DataFrame)
        self.assertIn(boot_id_col, func_result.columns.values)
        self.assertEqual(expected_result.shape, func_result.shape)
        npt.assert_allclose(expected_result.values, func_result.values)
        return None
