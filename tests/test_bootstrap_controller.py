"""
Tests for the bootstrap.py (formerly bootstrap_controller.py) file.
"""
import unittest
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix, eye

import pylogit.bootstrap as bc
import pylogit.asym_logit as asym
import pylogit.mixed_logit_calcs as mlc
import pylogit.mixed_logit as mixed_logit
import pylogit.nested_logit as nested_logit
from pylogit.conditional_logit import MNL


class BootstrapTests(unittest.TestCase):
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
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1]]))

        # Get the mappping between rows and observations
        self.fake_rows_to_obs = csr_matrix(np.array([[1, 0, 0, 0, 0, 0],
                                                     [1, 0, 0, 0, 0, 0],
                                                     [1, 0, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0],
                                                     [0, 1, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0],
                                                     [0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 1, 0, 0],
                                                     [0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 0, 0, 1],
                                                     [0, 0, 0, 0, 0, 1],
                                                     [0, 0, 0, 0, 0, 1]]))

        # Create the fake design matrix with columns denoting X
        # The intercepts are not included because they are kept outside the
        # index in the scobit model.
        self.fake_design = np.array([[1],
                                     [2],
                                     [3],
                                     [1.5],
                                     [3.5],
                                     [0.78],
                                     [0.23],
                                     [1.04],
                                     [2.52],
                                     [1.49],
                                     [0.85],
                                     [1.37],
                                     [1.17],
                                     [2.03],
                                     [1.62],
                                     [1.94]])

        # Create the index array for this set of choice situations
        self.fake_index = self.fake_design.dot(self.fake_betas)

        # Create the needed dataframe for the Asymmetric Logit constructor
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2, 3, 3, 3,
                                                4, 4, 5, 5, 5, 6, 6, 6],
                                     "alt_id": [1, 2, 3, 1, 3, 1, 2, 3,
                                                2, 3, 1, 2, 3, 1, 2, 3],
                                     "choice": [0, 1, 0, 0, 1, 1, 0, 0,
                                                1, 0, 1, 0, 0, 0, 0, 1],
                                     "x": self.fake_design[:, 0],
                                     "intercept":
                                        np.ones(self.fake_design.shape[0])})

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
        self.asym_model_obj = asym.MNAL(*self.constructor_args,
                                   **self.constructor_kwargs)

        self.asym_model_obj.coefs = pd.Series(self.fake_betas)
        self.asym_model_obj.intercepts =\
            pd.Series(self.fake_intercepts, index=self.fake_intercept_names)
        self.asym_model_obj.shapes =\
            pd.Series(self.fake_shapes, index=self.fake_shape_names)
        self.asym_model_obj.params =\
            pd.Series(np.concatenate([self.fake_shapes,
                                      self.fake_intercepts,
                                      self.fake_betas]),
                      index=self.fake_shape_names +
                            self.fake_intercept_names +
                            self.fake_names["x"])
        self.asym_model_obj.nests = None

        #####
        # Initialize a basic MNL model
        #####
        # Create the MNL specification and name dictionaries.
        self.mnl_spec, self.mnl_names = OrderedDict(), OrderedDict()
        self.mnl_spec["intercept"] = [1, 2]
        self.mnl_names["intercept"] = self.fake_intercept_names

        self.mnl_spec.update(self.fake_specification)
        self.mnl_names.update(self.fake_names)

        mnl_construct_args = self.constructor_args[:-1] + [self.mnl_spec]
        mnl_kwargs = {"names": self.mnl_names}
        self.mnl_model_obj = MNL(*mnl_construct_args, **mnl_kwargs)

        return None

    def test_get_param_names(self):
        # Alias the function being tested.
        func = bc.get_param_names

        # Get the function results
        func_results = func(self.asym_model_obj)

        # Get the expected results
        expected_results = self.asym_model_obj.params.index.tolist()

        # Test the function results
        self.assertIsInstance(func_results, list)
        self.assertEqual(func_results, expected_results)

        # Set the nest names and re-test the function.
        self.asym_model_obj.nest_names = ["No Nest"]
        expected_results_2 = self.asym_model_obj.nest_names + expected_results
        func_results_2 = func(self.asym_model_obj)
        self.assertIsInstance(func_results_2, list)
        self.assertEqual(func_results_2, expected_results_2)

        return None

    def test_get_param_list_for_prediction(self):
        # Determine the number of replicates
        num_replicates = 10
        # Create a fake model object with the needed attributes
        class FakeModel(object):
            def __init__(self):
                self.nest_names = ['one', 'oneA']
                self.shape_names = ['two', 'twoA', 'twoB']
                self.intercept_names = ['three']
                self.ind_var_names =\
                    ['four', 'fourA', 'fourB', 'fourC', 'fourD']
        fake_model_obj = FakeModel()

        # Create a fake set of bootstrap replicates
        fake_replicates =\
            (np.array([1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4])[None, :] *
             np.ones(num_replicates)[:, None])

        # Create the expected result
        expected_param_list = [4 * np.ones((5, num_replicates)),
                               3 * np.ones((1, num_replicates)),
                               2 * np.ones((3, num_replicates)),
                               np.ones((2, num_replicates))]

        # Alias the function being tested
        func = bc.get_param_list_for_prediction

        # Calculate the function result
        func_result = func(fake_model_obj, fake_replicates)

        # Perform the desired tests with a full set of parameters
        self.assertIsInstance(func_result, list)
        self.assertEqual(len(func_result), 4)
        for pos, func_array in enumerate(func_result):
            expected_array = expected_param_list[pos]
            self.assertIsInstance(func_array, np.ndarray)
            self.assertEqual(func_array.shape, expected_array.shape)
            npt.assert_allclose(func_array, expected_array)

        # Perform the desired tests with just index coefficients
        for attr in ['intercept_names', 'shape_names', 'nest_names']:
            setattr(fake_model_obj, attr, None)

        func_result_2 = func(fake_model_obj, fake_replicates[:, -5:])
        expected_result_2 =\
            [4 * np.ones((5, num_replicates)), None, None, None]

        self.assertIsInstance(func_result_2, list)
        for pos in xrange(1, 4):
            self.assertIsNone(func_result_2[pos])
        self.assertIsInstance(func_result_2[0], np.ndarray)
        self.assertEqual(func_result_2[0].shape, expected_result_2[0].shape)
        npt.assert_allclose(func_result_2[0], expected_result_2[0])

        return None

    def test_ensure_replicates_kwarg_validity(self):
        # Create the 'good' and 'bad' arguments for testing
        good_args = ['bootstrap', 'jackknife']
        bad_args = ['bad', 2, None]
        # Alias the function being tested
        func = bc.ensure_replicates_kwarg_validity
        # Note the expected error messsage
        expected_error_msg =\
            "`replicates` MUST be either 'bootstrap' or 'jackknife'."
        # Perform the desired tests
        for good_arg in good_args:
            self.assertIsNone(func(good_arg))
        for bad_arg in bad_args:
            self.assertRaisesRegexp(ValueError,
                                    expected_error_msg,
                                    func,
                                    bad_arg)
        return None

    def test_boot_initialization(self):
        # Create the bootstrap object
        boot_obj =\
            bc.Boot(self.asym_model_obj, self.asym_model_obj.params.values)

        # Test the bootstrap object.
        self.assertIsInstance(boot_obj, bc.Boot)
        self.assertEqual(id(boot_obj.model_obj), id(self.asym_model_obj))
        self.assertEqual(self.asym_model_obj.params.index.tolist(),
                         boot_obj.mle_params.index.tolist())
        expected_attrs =\
            ["bootstrap_replicates", "jackknife_replicates",
             "percentile_interval", "bca_interval",
             "abc_interval", "all_intervals",
             "jackknife_log_likehoods",
             "bootstrap_log_likelihoods"]
        for current_attr in expected_attrs:
            self.assertTrue(hasattr(boot_obj, current_attr))
            self.assertIsNone(getattr(boot_obj, current_attr))

        return None

    def test_generate_bootstrap_replicates(self):
        # Create the bootstrap object.
        boot_obj =\
            bc.Boot(self.asym_model_obj, self.asym_model_obj.params.values)

        # Determine the number of bootstrap samples that we wish to take
        num_samples = 3

        # Create the necessary keyword arguments.
        mnl_init_vals =\
            np.zeros(len(self.fake_intercept_names) +
                     sum([len(x) for x in self.fake_names.values()]))

        mnl_kwargs = {"ridge": 0.01,
                      "maxiter": 1200,
                      "method": "bfgs"}

        bootstrap_kwargs = {"mnl_obj": self.mnl_model_obj,
                            "mnl_init_vals": mnl_init_vals,
                            "mnl_fit_kwargs": mnl_kwargs,
                            "constrained_pos": [0],
                            "boot_seed": 1988}

        # Alias the needed function
        func = boot_obj.generate_bootstrap_replicates

        # Get the function results
        func_results =\
            func(num_samples,
                 mnl_obj=self.mnl_model_obj,
                 mnl_init_vals=mnl_init_vals,
                 mnl_fit_kwargs=mnl_kwargs,
                 constrained_pos=[0],
                 boot_seed=1988)

        # Perform the requisite tests
        self.assertIsNone(func_results)
        self.assertIsInstance(boot_obj.bootstrap_replicates, pd.DataFrame)
        self.assertEqual(boot_obj.bootstrap_replicates.ndim, 2)

        expected_shape = (num_samples, self.asym_model_obj.params.size)
        self.assertEqual(boot_obj.bootstrap_replicates.shape, expected_shape)
        self.assertEqual(boot_obj.bootstrap_replicates
                                 .iloc[:, 0].unique().size, 1)
        self.assertEqual(boot_obj.bootstrap_replicates
                                 .iloc[:, 0].unique()[0], 0)
        self.assertTrue(boot_obj.bootstrap_replicates
                                .iloc[:, 1].unique().size > 1)
        return None

    def test_generate_jackknife_replicates(self):
        # Create the bootstrap object.
        boot_obj =\
            bc.Boot(self.asym_model_obj, self.asym_model_obj.params.values)



        # Create the necessary keyword arguments.
        mnl_init_vals =\
            np.zeros(len(self.fake_intercept_names) +
                     sum([len(x) for x in self.fake_names.values()]))

        mnl_kwargs = {"ridge": 0.01,
                      "maxiter": 1200,
                      "method": "bfgs"}

        bootstrap_kwargs = {"mnl_obj": self.mnl_model_obj,
                            "mnl_init_vals": mnl_init_vals,
                            "mnl_fit_kwargs": mnl_kwargs,
                            "constrained_pos": [0],
                            "boot_seed": 1988}

        # Alias the needed function
        func = boot_obj.generate_jackknife_replicates

        # Get the function results
        func_results =\
            func(mnl_obj=self.mnl_model_obj,
                 mnl_init_vals=mnl_init_vals,
                 mnl_fit_kwargs=mnl_kwargs,
                 constrained_pos=[0])

        # Perform the requisite tests
        self.assertIsNone(func_results)
        self.assertIsInstance(boot_obj.jackknife_replicates, pd.DataFrame)
        self.assertEqual(boot_obj.jackknife_replicates.ndim, 2)

        expected_shape =\
            (self.fake_rows_to_obs.shape[1], self.asym_model_obj.params.size)
        self.assertEqual(boot_obj.jackknife_replicates.shape, expected_shape)
        self.assertEqual(boot_obj.jackknife_replicates
                                 .iloc[:, 0].unique().size, 1)
        self.assertEqual(boot_obj.jackknife_replicates
                                 .iloc[:, 0].unique()[0], 0)
        self.assertTrue(boot_obj.jackknife_replicates
                                .iloc[:, 1].unique().size > 1)
        return None


class IntervalTests(unittest.TestCase):
    """
    References
    ----------
    Efron, Bradley, and Robert J. Tibshirani. An Introduction to the
        Bootstrap. CRC press, 1994. Chapter 14.

    Notes
    -----
    The data and tests used in the `IntervalTests` test suite come from the
    Efron & Tibshirani reference cited above.
    """
    def setUp(self):
        # Store the spatial test data from Efron and Tibshirani (1994)
        self.test_data =\
            np.array([48, 36, 20, 29, 42, 42, 20, 42, 22, 41, 45, 14, 6,
                      0, 33, 28, 34, 4, 32, 24, 47, 41, 24, 26, 30, 41])

        # Note how many test data observations there are.
        self.num_test_obs = self.test_data.size

        # Store the MLE estimate
        self.test_theta_hat = self.calc_theta(self.test_data)
        # Create a pandas series of the data. Allows for easy case deletion.
        self.raw_series = pd.Series(self.test_data)
        # Create the array of jackknife replicates
        self.jackknife_replicates =\
            np.empty((self.num_test_obs, 1), dtype=float)
        for obs in xrange(self.num_test_obs):
            current_data = self.raw_series[self.raw_series.index != obs].values
            self.jackknife_replicates[obs] = self.calc_theta(current_data)[0]

        # Create the bootstrap replicates
        num_test_reps = 5000
        test_indices = np.arange(self.num_test_obs)
        boot_indx_shape = (num_test_reps, self.num_test_obs)
        np.random.seed(8292017)
        boot_indices =\
            np.random.choice(test_indices,
                             replace=True,
                             size=self.num_test_obs*num_test_reps)
        self.bootstrap_replicates =\
            np.fromiter((self.calc_theta(self.test_data[x])[0] for x in
                         boot_indices.reshape(boot_indx_shape)),
                        dtype=float)[:, None]

        self.rows_to_obs = eye(self.test_data.size, format='csr', dtype=int)

        # Create a fake model object and a fake model class that will implement the
        # T(P) function through it's fit_mle method.
        test_data = self.test_data
        fake_rows_to_obs = self.rows_to_obs
        calc_theta = self.calc_theta

        class FakeModel(object):
            def __init__(self):
                # Create needed attributes to successfully mock an MNDC_Model
                #instance in this test
                self.data = pd.Series([pos for pos, x in enumerate(test_data)])
                self.obs_id_col = np.arange(self.data.size, dtype=int)

                needed_names = ['ind_var_names', 'intercept_names',
                                'shape_names', 'nest_names']
                for name in needed_names:
                    setattr(self, name, None)
                self.ind_var_names = ['variance']

            # Create a get_mappings_for_fit function that will allow for
            # successful mocking in this test
            def get_mappings_for_fit(self):
                return {"rows_to_obs": fake_rows_to_obs}

            # Use the T(P) function from the spatial test data example.
            def fit_mle(self,
                        init_vals,
                        weights=None,
                        **kwargs):
                return {'x': calc_theta(test_data, weights=weights)}
        self.fake_model_obj = FakeModel()

        # Create the bootstrap object
        self.boot =\
            bc.Boot(self.fake_model_obj,
                    pd.Series(self.test_theta_hat, index=["variance"]))

        self.boot.bootstrap_replicates =\
            pd.DataFrame(self.bootstrap_replicates, columns=['variance'])
        self.boot.jackknife_replicates =\
            pd.DataFrame(self.jackknife_replicates, columns=['variance'])

        # Store the confidence percentage that will be used for the test
        self.conf_percentage = 90
        return None

    # Create the function to calculate the objective function.
    def calc_theta(self, array, weights=None):
        if weights is None:
            result = ((array - array.mean())**2).sum() / float(array.size)
        else:
            a_mean = weights.dot(array)
            differences = (array - a_mean)
            squared_diffs = differences**2
            result = weights.dot(squared_diffs)
        return np.array([result])

    def test_calc_percentile_interval(self):
        # Alias the function being tested
        func = self.boot.calc_percentile_interval

        # Perform the first test
        self.assertIsNone(self.boot.percentile_interval)

        # Calculate the function result
        func(self.conf_percentage)

        # Note the expected result is from Table 14.2 on page 183 of
        # Efron & Tibshirani (1994)
        expected_result = np.array([100.8, 233.9])
        expected_columns = ['5%', '95%']

        # Perform the remaining tests
        self.assertIsInstance(self.boot.percentile_interval, pd.DataFrame)
        self.assertEqual(expected_columns,
                         self.boot.percentile_interval.columns.tolist())
        self.assertIn("variance", self.boot.percentile_interval.index)
        self.assertEqual(self.boot.percentile_interval.shape, (1, 2))
        npt.assert_allclose(self.boot.percentile_interval.iloc[0, :],
                            expected_result, rtol=0.02)

        # Set the percentile interval back to none.
        self.boot.percentile_interval = None
        self.assertIsNone(self.boot.percentile_interval)
        return None

    def test_calc_bca_interval(self):
        # Alias the function being tested
        func = self.boot.calc_bca_interval

        # Perform the first test
        self.assertIsNone(self.boot.bca_interval)

        # Calculate the function result
        func(self.conf_percentage)

        # Note the expected result is from Table 14.2 on page 183 of
        # Efron & Tibshirani (1994)
        expected_result = np.array([115.8, 259.6])
        expected_columns = ['5%', '95%']

        # Perform the remaining tests
        self.assertIsInstance(self.boot.bca_interval, pd.DataFrame)
        self.assertEqual(expected_columns,
                         self.boot.bca_interval.columns.tolist())
        self.assertIn("variance", self.boot.bca_interval.index)
        self.assertEqual(self.boot.bca_interval.shape, (1, 2))
        npt.assert_allclose(self.boot.bca_interval.iloc[0, :],
                            expected_result, rtol=0.01)

        # Set the percentile interval back to none.
        self.boot.bca_interval = None
        self.assertIsNone(self.boot.bca_interval)
        return None

    def test_calc_abc_interval(self):
        # Alias the function being tested
        func = self.boot.calc_abc_interval

        # Perform the first test
        self.assertIsNone(self.boot.abc_interval)

        # Calculate the function result
        func(self.conf_percentage, self.test_theta_hat, epsilon=0.001)

        # Note the expected result, from Table 14.2 on page 183 of
        # Efron & Tibshirani (1994)
        expected_result = np.array([116.7, 260.9])
        expected_columns = ['5%', '95%']

        # Perform the remaining tests
        self.assertIsInstance(self.boot.abc_interval, pd.DataFrame)
        self.assertEqual(expected_columns,
                         self.boot.abc_interval.columns.tolist())
        self.assertIn("variance", self.boot.abc_interval.index)
        self.assertEqual(self.boot.abc_interval.shape, (1, 2))
        npt.assert_allclose(self.boot.abc_interval.iloc[0, :],
                            expected_result, rtol=0.01)

        # Set the percentile interval back to none.
        self.boot.abc_interval = None
        self.assertIsNone(self.boot.abc_interval)
        return None

    def test_calc_conf_intervals_except_all(self):
        kwargs = {"init_vals": self.test_theta_hat,
                  "epsilon": 0.001}

        # Alias the function being tested
        func = self.boot.calc_conf_intervals

        # Create the list of attributes to be tested
        tested_attrs = ['percentile_interval', 'bca_interval', 'abc_interval']
        interval_types = ['pi', 'bca', 'abc']

        # Note the expected result, from Table 14.2 on page 183 of
        # Efron & Tibshirani (1994)
        expected_result =\
            np.array([[100.8, 233.9], [115.8, 259.6], [116.7, 260.9]])
        expected_columns = ['5%', '95%']

        # Perform the desired tests
        for pos, i_type in enumerate(interval_types):
            desired_attr = getattr(self.boot, tested_attrs[pos])
            self.assertIsNone(desired_attr)
            # Calculate the function result
            kwargs['interval_type'] = i_type
            func(self.conf_percentage, **kwargs)
            # Perform the remaining tests
            desired_attr = getattr(self.boot, tested_attrs[pos])
            self.assertIsInstance(desired_attr, pd.DataFrame)
            self.assertEqual(expected_columns,
                             desired_attr.columns.tolist())
            self.assertIn("variance", desired_attr.index)
            self.assertEqual(desired_attr.shape, (1, 2))
            npt.assert_allclose(desired_attr.iloc[0, :],
                                expected_result[pos], rtol=0.02)

            # Perform clean-up activities after the test
            setattr(self.boot, tested_attrs[pos], None)
        return None

    def test_calc_conf_intervals_all(self):
        kwargs = {"interval_type": 'all',
                  "init_vals": self.test_theta_hat,
                  "epsilon": 0.001}

        # Alias the function being tested
        func = self.boot.calc_conf_intervals

        # Create the list of attributes to be tested
        tested_attrs = ['percentile_interval', 'bca_interval', 'abc_interval']

        # Note the expected result, from Table 14.2 on page 183 of
        # Efron & Tibshirani (1994)
        expected_result =\
            np.array([[100.8, 233.9], [115.8, 259.6], [116.7, 260.9]])

        # Note the expected MultiIndex columns
        expected_columns_all = [("percentile_interval", "5%"),
                                ("percentile_interval", "95%"),
                                ("BCa_interval", "5%"),
                                ("BCa_interval", "95%"),
                                ("ABC_interval", "5%"),
                                ("ABC_interval", "95%")]
        expected_columns_single = ["5%", "95%"]

        # Perform the expected tests before running the function
        for attr in tested_attrs:
            self.assertIsNone(getattr(self.boot, attr))

        # Calculate the function results
        func(self.conf_percentage, **kwargs)

        # Perform the remaining tests
        for pos, attr in enumerate(tested_attrs):
            desired_attr = getattr(self.boot, attr)
            self.assertEqual(expected_columns_single,
                             desired_attr.columns.tolist())
            self.assertIn("variance", desired_attr.index)
            self.assertEqual(desired_attr.shape, (1, 2))
            npt.assert_allclose(desired_attr.iloc[0, :],
                                expected_result[pos], rtol=0.02)

        # Test the 'all_intervals' attribute.
        self.assertIsInstance(self.boot.all_intervals, pd.DataFrame)
        self.assertEqual(expected_columns_all,
                         self.boot.all_intervals.columns.tolist())
        self.assertIn("variance", self.boot.all_intervals.index)
        self.assertEqual(self.boot.all_intervals.shape, (1, 6))
        npt.assert_allclose(self.boot.all_intervals.values,
                            expected_result.reshape((1, 6)), rtol=0.02)

        # Set the various intervals back to None.
        for attr in tested_attrs + ['all_intervals']:
            setattr(self.boot, attr, None)
            self.assertIsNone(getattr(self.boot, attr))
        return None

    def test_interval_type_error_in_calc_conf_intervals(self):
        # Alias the function being tested
        func = self.boot.calc_conf_intervals

        # Create kwargs for the function to be tested
        kwargs = {"interval_type": 'bad_type',
                  "init_vals": self.test_theta_hat,
                  "epsilon": 0.001}

        # Note the expected error message.
        expected_error_msg =\
            "interval_type MUST be in `\['pi', 'bca', 'abc', 'all'\]`"

        # Ensure that the appropriate errors are raised.
        self.assertRaisesRegexp(ValueError,
                                expected_error_msg,
                                func,
                                self.conf_percentage,
                                **kwargs)
        return None


class AnalysisTests(unittest.TestCase):
    def make_mnl_model(self):
        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two
        # alternatives. There is one generic variable. Two alternative
        # specific constants and all three shape parameters are used.

        # Create the betas to be used during the tests
        fake_betas = np.array([-0.6])

        # Create the fake design matrix with columns denoting X
        # The intercepts are not included because they are kept outside the
        # index in the scobit model.
        fake_design = np.array([[1],
                                [2],
                                [3],
                                [1.5],
                                [3.5]])

        # Create the index array for this set of choice situations
        fake_index = fake_design.dot(fake_betas)

        # Create the needed dataframe for the model constructor
        fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                "alt_id": [1, 2, 3, 1, 3],
                                "choice": [0, 1, 0, 0, 1],
                                "x": fake_design[:, 0]})

        # Record the various column names
        alt_id_col = "alt_id"
        obs_id_col = "obs_id"
        choice_col = "choice"

        # Create the index specification  and name dictionaryfor the model
        mnl_spec = OrderedDict()
        mnl_names = OrderedDict()
        mnl_spec["x"] = [[1, 2, 3]]
        mnl_names["x"] = ["x (generic coefficient)"]

        # Bundle args and kwargs used to construct the Asymmetric Logit model.
        mnl_args = [fake_df, alt_id_col, obs_id_col, choice_col, mnl_spec]

        # Create a variable for the kwargs being passed to the constructor
        mnl_kwargs = {"names": mnl_names}

        # Initialize a basic choice model.
        mnl_obj = MNL(*mnl_args, **mnl_kwargs)

        # Create the desired model attributes for the clog log model
        mnl_obj.coefs = pd.Series(fake_betas, index=mnl_names["x"])
        mnl_obj.intercepts = None
        mnl_obj.shapes = None
        mnl_obj.nests = None
        mnl_obj.params = mnl_obj.coefs.copy()
        return mnl_obj

    def make_asym_model(self):
        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two
        # alternatives. There is one generic variable. Two alternative
        # specific constants and all three shape parameters are used.

        # Create the betas to be used during the tests
        fake_betas = np.array([-0.6])

        # Create the fake outside intercepts to be used during the tests
        fake_intercepts = np.array([1, 0.5])

        # Create names for the intercept parameters
        fake_intercept_names = ["ASC 1", "ASC 2"]

        # Record the position of the intercept that is not being estimated
        fake_intercept_ref_pos = 2

        # Create the shape parameters to be used during the tests. Note that
        # these are the reparameterized shape parameters, thus they will be
        # exponentiated in the fit_mle process and various calculations.
        fake_shapes = np.array([-1, 1])

        # Create names for the intercept parameters
        fake_shape_names = ["Shape 1", "Shape 2"]

        # Record the position of the shape parameter that is being constrained
        fake_shape_ref_pos = 2

        # Calculate the 'natural' shape parameters
        natural_shapes = asym._convert_eta_to_c(fake_shapes,
                                                fake_shape_ref_pos)

        # Create an array of all model parameters
        fake_all_params = np.concatenate((fake_shapes,
                                          fake_intercepts,
                                          fake_betas))

        # The mapping between rows and alternatives is given below.
        fake_rows_to_alts = csr_matrix(np.array([[1, 0, 0],
                                                 [0, 1, 0],
                                                 [0, 0, 1],
                                                 [1, 0, 0],
                                                 [0, 0, 1]]))

        # Get the mappping between rows and observations
        fake_rows_to_obs = csr_matrix(np.array([[1, 0],
                                                [1, 0],
                                                [1, 0],
                                                [0, 1],
                                                [0, 1]]))

        # Create the fake design matrix with columns denoting X
        # The intercepts are not included because they are kept outside the
        # index in the scobit model.
        fake_design = np.array([[1],
                                [2],
                                [3],
                                [1.5],
                                [3.5]])

        # Create the index array for this set of choice situations
        fake_index = fake_design.dot(fake_betas)

        # Create the needed dataframe for the Asymmetric Logit constructor
        fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                "alt_id": [1, 2, 3, 1, 3],
                                "choice": [0, 1, 0, 0, 1],
                                "x": fake_design[:, 0],
                                "intercept": [1 for i in range(5)]})

        # Record the various column names
        alt_id_col = "alt_id"
        obs_id_col = "obs_id"
        choice_col = "choice"

        # Create the index specification  and name dictionaryfor the model
        fake_specification = OrderedDict()
        fake_names = OrderedDict()
        fake_specification["x"] = [[1, 2, 3]]
        fake_names["x"] = ["x (generic coefficient)"]

        # Bundle args and kwargs used to construct the Asymmetric Logit model.
        constructor_args = [fake_df,
                            alt_id_col,
                            obs_id_col,
                            choice_col,
                            fake_specification]

        # Create a variable for the kwargs being passed to the constructor
        constructor_kwargs = {"intercept_ref_pos": fake_intercept_ref_pos,
                              "shape_ref_pos": fake_shape_ref_pos,
                              "names": fake_names,
                              "intercept_names": fake_intercept_names,
                               "shape_names": fake_shape_names}

        # Initialize a basic Asymmetric Logit model whose coefficients will be
        # estimated.
        model_obj = asym.MNAL(*constructor_args, **constructor_kwargs)

        # Get the fitted probabilities for this model and dataset
        # Note this relies on the calc_probabilities function being functional.
        # args = [fake_betas,
        #         fake_design,
        #         fake_df[alt_id_col].values,
        #         fake_rows_to_obs,
        #         fake_rows_to_alts,
        #         model_obj.utility_transform]
        # kwargs = {"intercept_params": fake_intercepts,
        #           "shape_params": fake_shapes,
        #           "return_long_probs": True}
        # model_obj.prob_array =\
        #     choice_calcs.calc_probabilities(*args, **kwargs)

        model_obj.coefs = pd.Series(fake_betas, index=fake_names["x"])
        model_obj.intercepts =\
            pd.Series(fake_intercepts, index=fake_intercept_names)
        model_obj.shapes = pd.Series(fake_shapes, index=fake_shape_names)
        model_obj.nests = None
        model_obj.params =\
            pd.concat([model_obj.shapes,
                       model_obj.intercepts,
                       model_obj.coefs],
                      axis=0, ignore_index=False)
        return model_obj

    def make_mixed_model(self):
        # Fake random draws where Row 1 is for observation 1 and row 2 is
        # for observation 2. Column 1 is for draw 1 and column 2 is for draw 2
        fake_draws = mlc.get_normal_draws(2, 2, 1, seed=1)[0]
        # Create the betas to be used during the tests
        fake_betas = np.array([0.3, -0.6, 0.2])
        fake_std = 1
        fake_betas_ext = np.concatenate((fake_betas,
                                         np.array([fake_std])),
                                        axis=0)

        # Create the fake design matrix with columns denoting ASC_1, ASC_2, X
        fake_design = np.array([[1, 0, 1],
                               [0, 1, 2],
                               [0, 0, 3],
                               [1, 0, 1.5],
                               [0, 1, 2.5],
                               [0, 0, 3.5],
                               [1, 0, 0.5],
                               [0, 1, 1.0],
                               [0, 0, 1.5]])
        # Record what positions in the design matrix are being mixed over
        mixing_pos = [2]

        # Create the arrays that specify the choice situation, individual id
        # and alternative ids
        situation_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        individual_ids = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2])
        alternative_ids = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        # Create a fake array of choices
        choice_array = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])

        # Create the 'rows_to_mixers' sparse array for this dataset
        # Denote the rows that correspond to observation 1 and observation 2
        obs_1_rows = np.ones(fake_design.shape[0])
        # Make sure the rows for observation 2 are given a zero in obs_1_rows
        obs_1_rows[-3:] = 0
        obs_2_rows = 1 - obs_1_rows
        # Create the row_to_mixers scipy.sparse matrix
        fake_rows_to_mixers = csr_matrix(obs_1_rows[:, None] ==
                                         np.array([1, 0])[None, :])
        # Create the rows_to_obs scipy.sparse matrix
        fake_rows_to_obs = csr_matrix(situation_ids[:, None] ==
                                      np.arange(1, 4)[None, :])
        # Create the rows_to_alts scipy.sparse matrix
        fake_rows_to_alts = csr_matrix(alternative_ids[:, None] ==
                                       np.arange(1, 4)[None, :])

        # Create the design matrix that we should see for draw 1 and draw 2
        arrays_to_join = (fake_design.copy(),
                          fake_design.copy()[:, -1][:, None])
        fake_design_draw_1 = np.concatenate(arrays_to_join, axis=1)
        fake_design_draw_2 = fake_design_draw_1.copy()

        # Multiply the 'random' coefficient draws by the corresponding variable
        fake_design_draw_1[:, -1] *= (obs_1_rows *
                                      fake_draws[0, 0] +
                                      obs_2_rows *
                                      fake_draws[1, 0])
        fake_design_draw_2[:, -1] *= (obs_1_rows *
                                      fake_draws[0, 1] +
                                      obs_2_rows *
                                      fake_draws[1, 1])
        extended_design_draw_1 = fake_design_draw_1[:, None, :]
        extended_design_draw_2 = fake_design_draw_2[:, None, :]
        fake_design_3d = np.concatenate((extended_design_draw_1,
                                         extended_design_draw_2),
                                        axis=1)

        # Create the fake systematic utility values
        sys_utilities_draw_1 = fake_design_draw_1.dot(fake_betas_ext)
        sys_utilities_draw_2 = fake_design_draw_2.dot(fake_betas_ext)

        #####
        # Calculate the probabilities of each alternatve in each choice
        # situation
        #####
        long_exp_draw_1 = np.exp(sys_utilities_draw_1)
        long_exp_draw_2 = np.exp(sys_utilities_draw_2)
        ind_exp_sums_draw_1 = fake_rows_to_obs.T.dot(long_exp_draw_1)
        ind_exp_sums_draw_2 = fake_rows_to_obs.T.dot(long_exp_draw_2)
        long_exp_sum_draw_1 = fake_rows_to_obs.dot(ind_exp_sums_draw_1)
        long_exp_sum_draw_2 = fake_rows_to_obs.dot(ind_exp_sums_draw_2)
        long_probs_draw_1 = long_exp_draw_1 / long_exp_sum_draw_1
        long_probs_draw_2 = long_exp_draw_2 / long_exp_sum_draw_2
        prob_array = np.concatenate((long_probs_draw_1[:, None],
                                     long_probs_draw_2[:, None]),
                                    axis=1)

        ###########
        # Create a mixed logit object for later use.
        ##########
        # Create a fake old long format dataframe for mixed logit model object
        alt_id_column = "alt_id"
        situation_id_column = "situation_id"
        obs_id_column = "observation_id"
        choice_column = "choice"

        data = {"x": fake_design[:, 2],
                alt_id_column: alternative_ids,
                situation_id_column: situation_ids,
                obs_id_column: individual_ids,
                choice_column: choice_array}
        fake_old_df = pd.DataFrame(data)
        fake_old_df["intercept"] = 1

        # Create a fake specification
        fake_spec = OrderedDict()
        fake_names = OrderedDict()

        fake_spec["intercept"] = [1, 2]
        fake_names["intercept"] = ["ASC 1", "ASC 2"]

        fake_spec["x"] = [[1, 2, 3]]
        fake_names["x"] = ["beta_x"]

        # Specify the mixing variable
        fake_mixing_vars = ["beta_x"]

        # Create a fake version of a mixed logit model object
        args = [fake_old_df,
                alt_id_column,
                situation_id_column,
                choice_column,
                fake_spec]
        kwargs = {"names": fake_names,
                  "mixing_id_col": obs_id_column,
                  "mixing_vars": fake_mixing_vars}
        mixl_obj = mixed_logit.MixedLogit(*args, **kwargs)

        # Set all the necessary attributes for prediction:
        # design_3d, coefs, intercepts, shapes, nests, mixing_pos
        mixl_obj.design_3d = fake_design_3d
        mixl_obj.ind_var_names += ["Sigma X"]
        mixl_obj.coefs =\
            pd.Series(fake_betas_ext, index=mixl_obj.ind_var_names)
        mixl_obj.intercepts = None
        mixl_obj.shapes = None
        mixl_obj.nests = None
        mixl_obj.params = mixl_obj.coefs.copy()
        return mixl_obj

    def make_nested_model(self):
        # Create the betas to be used during the tests
        fake_betas = np.array([0.3, -0.6, 0.2])
        # Create the fake nest coefficients to be used during the tests
        # Note that these are the 'natural' nest coefficients, i.e. the
        # inverse of the scale parameters for each nest. They should be bigger
        # than or equal to 1.
        natural_nest_coefs = np.array([1 - 1e-16, 0.5])
        # Create an array of all model parameters
        fake_all_params = np.concatenate((natural_nest_coefs,
                                          fake_betas))
        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two.
        # The nest memberships of these alternatives are given below.
        fake_rows_to_nests = csr_matrix(np.array([[1, 0],
                                                  [1, 0],
                                                  [0, 1],
                                                  [1, 0],
                                                  [0, 1]]))

        # Create a sparse matrix that maps the rows of the design matrix to the
        # observatins
        fake_rows_to_obs = csr_matrix(np.array([[1, 0],
                                                [1, 0],
                                                [1, 0],
                                                [0, 1],
                                                [0, 1]]))

        # Create the fake design matrix with columns denoting ASC_1, ASC_2, X
        fake_design = np.array([[1, 0, 1],
                                [0, 1, 2],
                                [0, 0, 3],
                                [1, 0, 1.5],
                                [0, 0, 3.5]])

        # Create fake versions of the needed arguments for the MNL constructor
        fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                "alt_id": [1, 2, 3, 1, 3],
                                "choice": [0, 1, 0, 0, 1],
                                "x": range(5),
                                "intercept": [1 for i in range(5)]})

        # Record the various column names
        alt_id_col = "alt_id"
        obs_id_col = "obs_id"
        choice_col = "choice"

        # Store the choice array
        choice_array = fake_df[choice_col].values

        # Create a sparse matrix that maps the chosen rows of the design
        # matrix to the observatins
        fake_chosen_rows_to_obs = csr_matrix(np.array([[0, 0],
                                                       [1, 0],
                                                       [0, 0],
                                                       [0, 0],
                                                       [0, 1]]))

        # Create the index specification  and name dictionaryfor the model
        fake_specification = OrderedDict()
        fake_specification["intercept"] = [1, 2]
        fake_specification["x"] = [[1, 2, 3]]
        fake_names = OrderedDict()
        fake_names["intercept"] = ["ASC 1", "ASC 2"]
        fake_names["x"] = ["x (generic coefficient)"]

        # Create the nesting specification
        fake_nest_spec = OrderedDict()
        fake_nest_spec["Nest 1"] = [1, 2]
        fake_nest_spec["Nest 2"] = [3]

        # Create a nested logit object
        args = [fake_df,
                alt_id_col,
                obs_id_col,
                choice_col,
                fake_specification]
        kwargs = {"names": fake_names,
                  "nest_spec": fake_nest_spec}
        model_obj = nested_logit.NestedLogit(*args, **kwargs)

        model_obj.coefs = pd.Series(fake_betas, index=model_obj.ind_var_names)
        model_obj.intercepts = None
        model_obj.shapes = None

        def logit(x):
            return np.log(x / (1 - x))
        model_obj.nests =\
            pd.Series(logit(natural_nest_coefs), index=fake_nest_spec.keys())
        model_obj.params =\
            pd.concat([model_obj.nests, model_obj.coefs],
                      axis=0, ignore_index=False)

        # Store a ridge parameter
        # ridge = 0.5

        # Gather the arguments needed for the calc_nested_probs function
        # args = [natural_nest_coefs,
        #         fake_betas,
        #         model_obj.design,
        #         fake_rows_to_obs,
        #         fake_rows_to_nests]
        # kwargs = {"return_type": "long_probs"}
        # model_obj.prob_array = nlc.calc_nested_probs(*args, **kwargs)
        return model_obj

    def setUp(self):
        """
        Create the real model objects.
        """
        self.mnl_model = self.make_mnl_model()
        self.asym_model = self.make_asym_model()
        self.mixed_model = self.make_mixed_model()
        self.nested_model = self.make_nested_model()
        return None

    def test_calc_log_likes_for_replicates(self):
        # Create the keyword arguments needed for the test.
        kwargs = {'num_draws': 10, 'seed': 932017}

        # Note the objects that are to be tested
        model_objects = [self.mnl_model,
                         self.asym_model,
                         self.mixed_model,
                         self.nested_model]

        # Iterate over the Asym, MNL, Mixed, and Nested models.
        for model_obj in model_objects:
            # Create the bootstrap object based on the model object.
            boot = bc.Boot(model_obj, model_obj.params.values)

            # Create the bootstrap and jackknife replicate attributes.
            replicates =\
                pd.DataFrame(np.concatenate([model_obj.params.values[None, :],
                                             model_obj.params.values[None, :]],
                                            axis=0))
            boot.bootstrap_replicates = replicates
            boot.jackknife_replicates = replicates

            # Alias the function being tested.
            func = boot.calc_log_likes_for_replicates

            for replicate_type in ['bootstrap', 'jackknife']:
                # Calculate function results using each bootstrap object
                kwargs["replicates"] = replicate_type
                func_result = func(**kwargs)

                # Ensure function results have the expected properties
                self.assertIsInstance(func_result, np.ndarray)
                self.assertEqual(func_result.ndim, 1)
                self.assertEqual(func_result.shape, (replicates.shape[0],))
        return None

    def test_calc_gradient_norm_for_replicates(self):
        # Create the bootstrap object based on the MNL model.
        base_array = self.mnl_model.params.values
        base_array_2d = base_array[None, :]
        boot = bc.Boot(self.mnl_model, base_array)

        # Create the bootstrap and jackknife replicate attributes.
        replicates = pd.DataFrame(np.concatenate((base_array_2d,
                                                  base_array_2d + 1,
                                                  base_array_2d - 1),
                                                 axis=0))
        boot.bootstrap_replicates = replicates
        boot.jackknife_replicates = replicates

        # Alias the function being tested.
        func = boot.calc_gradient_norm_for_replicates

        # Perform the desired tests.
        for replicate_type in ['bootstrap', 'jackknife']:
            func_result = func(replicates=replicate_type)
            self.assertIsInstance(func_result, np.ndarray)
            self.assertEqual(func_result.shape, (replicates.shape[0],))
            self.assertTrue(np.unique(func_result).size == func_result.size)
        return None
