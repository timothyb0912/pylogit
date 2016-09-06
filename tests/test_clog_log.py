"""
Tests for the clog_log.py file. These tests do not include tests of
the functions that perform the mathematical calculations necessary to estimate
the Clog-log model.
"""
import warnings
import unittest
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import pylogit.clog_log as clog


class GenericTestCase(unittest.TestCase):
    """
    Defines the common setUp method used for the different type of tests.
    """
    def setUp(self):
        # Create the betas to be used during the tests
        self.fake_betas = np.array([-0.6])

        # Create the fake outside intercepts to be used during the tests
        self.fake_intercepts = np.array([1, 0.5])

        # Create names for the intercept parameters
        self.fake_intercept_names = ["ASC 1", "ASC 2"]

        # Record the position of the intercept that is not being estimated
        self.fake_intercept_ref_pos = 2

        # Create an array of all model parameters
        self.fake_all_params = np.concatenate((self.fake_intercepts,
                                               self.fake_betas))

        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two.
        # The mapping between rows and alternatives is given below.
        self.fake_rows_to_alts = csr_matrix(np.array([[1, 0, 0],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [1, 0, 0],
                                                      [0, 0, 1]]))

        # Create the fake design matrix with columns denoting X
        # The intercepts are not included because they are kept outside the
        # index in the clog-log model.
        self.fake_design = np.array([[1],
                                     [2],
                                     [3],
                                     [1.5],
                                     [3.5]])

        # Create the index array for this set of choice situations
        self.fake_index = self.fake_design.dot(self.fake_betas)

        # Create the needed dataframe for the Clog-log constructor
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

        # Bundle the args and kwargs used to construct the Clog-log model.
        self.constructor_args = [self.fake_df,
                                 self.alt_id_col,
                                 self.obs_id_col,
                                 self.choice_col,
                                 self.fake_specification]

        # Create a variable for the kwargs being passed to the constructor
        self.constructor_kwargs = {"intercept_ref_pos":
                                   self.fake_intercept_ref_pos,
                                   "names": self.fake_names,
                                   "intercept_names":
                                   self.fake_intercept_names}

        # Initialize a basic clog-log model.
        # Create the clog model object whose coefficients will be estimated.
        self.base_clog = clog.MNCL(*self.constructor_args,
                                   **self.constructor_kwargs)

        return None


class HelperFuncTests(GenericTestCase):
    """
    Defines tests for the 'helper' functions for estimating the Clog-log model.
    """
    def test_split_param_vec(self):
        """
        Ensures that split_param_vec returns (None, intercepts, index_coefs)
        when called from within clog_log.py.
        """
        # Store the results of split_param_vec()
        split_results = clog.split_param_vec(self.fake_all_params,
                                             self.fake_rows_to_alts,
                                             self.fake_design)
        # Check for expected results.
        self.assertIsNone(split_results[0])
        for item in split_results[1:]:
            self.assertIsInstance(item, np.ndarray)
            self.assertEqual(len(item.shape), 1)
        npt.assert_allclose(split_results[1], self.fake_intercepts)
        npt.assert_allclose(split_results[2], self.fake_betas)

        return None

    def test_clog_utility_transform(self):
        """
        Ensures that `_clog_utility_transform()` returns correct results
        """
        # Note that this test index will test the function at values that
        # lead to underflow, standard calculations, and overflow. Note
        # overflow basically happens when the index is >= 3.7
        test_index = np.array([0, 4, -200, -1, 10])

        # Calculate the results "by hand", excluding the outside intercepts
        correct_results = np.array([np.log(np.exp(1) - 1),
                                    np.exp(4),
                                    -1 * clog.max_comp_value,
                                    np.log(np.exp(np.exp(-1)) - 1),
                                    np.exp(10)])[:, None]

        # Account for the outside intercepts
        correct_results += np.array([self.fake_intercepts[0],
                                     self.fake_intercepts[1],
                                     0,
                                     self.fake_intercepts[0],
                                     0])[:, None]

        # Bundle the remaining args and kwargs
        args = [self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_alts,
                None,
                self.fake_intercepts]
        kwargs = {"intercept_ref_pos": self.fake_intercept_ref_pos}

        # Get the results of _clog_utiilty_transform()
        transform_results = clog._cloglog_utility_transform(test_index,
                                                            *args,
                                                            **kwargs)

        # Check to make sure the results are as expected
        self.assertIsInstance(transform_results, np.ndarray)
        self.assertEqual(transform_results.shape, (test_index.shape[0], 1))
        npt.assert_allclose(transform_results, correct_results)

        return None

    def test_cloglog_transform_deriv_v(self):
        """
        Tests basic behavior of the cloglog_transform_deriv_v.
        """
        # Note the index has a value that is <= -40 to test whether or not
        # the function correctly uses L'Hopital's rule to deal with underflow
        # and calculating the derivative. When the index is <= -40, the
        # derivative should be 1.
        test_index = np.array([-40, 1, 7])
        # Note we use a compressed sparse-row matrix so that we can easily
        # convert the output matrix to a numpy array using the '.A' attribute.
        test_output = diags(np.ones(test_index.shape[0]),
                            0, format='csr')

        # Bundle the arguments needed for the function
        # Not all elements except for test_index are completely fake and only
        # needed because the function requires a given number of arguments.
        # This is for api compatibility with other models.
        args = [test_index,
                np.ones(3),
                diags(np.ones(3), 0, format='csr'),
                None]

        # Get the derivative using the function defined in clog_log.py.
        derivative = clog._cloglog_transform_deriv_v(*args,
                                                     output_array=test_output)

        # Calculate, 'by hand' what the results should be
        correct_derivatives = np.diag(np.array([1,
                                                2.910328703250801,
                                                1096.6331584284585]))

        self.assertIsInstance(derivative, type(test_output))
        self.assertEqual(len(derivative.shape), 2)
        self.assertEqual(derivative.shape, (3, 3))
        npt.assert_allclose(correct_derivatives, derivative.A)

        return None

    def test_cloglog_transform_deriv_c(self):
        """
        Ensures that cloglog_transform_deriv_c returns None.
        """
        fake_args = ["foo", 1, None]
        fake_kwargs = {"fake_arg_1": "bar",
                       "fake_arg_2": 2,
                       "fake_arg_3": True}
        derivative_results = clog._cloglog_transform_deriv_c(self.fake_index,
                                                             *fake_args,
                                                             **fake_kwargs)
        self.assertIsNone(derivative_results)

        return None

    def test_cloglog_transform_deriv_alpha(self):
        """
        Ensures that cloglog_transform_deriv_alpha returns the `output_array`
        kwarg.
        """
        # Bundle the args for the function being tested
        args = [self.fake_index,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_alts,
                self.fake_intercepts]

        # Take the relevant columns of rows_to_alts, as would be done with
        # outside intercepts, or test None for when there are no outside
        # intercepts.
        for test_output in [None, self.fake_rows_to_alts[:, [0, 1]]]:
            kwargs = {"output_array": test_output}
            derivative_results = clog._cloglog_transform_deriv_alpha(*args,
                                                                     **kwargs)
            if test_output is None:
                self.assertIsNone(derivative_results)
            else:
                npt.assert_allclose(test_output.A, derivative_results.A)

        return None


# Note that inheritance is used to share the setUp method.
class ChoiceObjectTests(GenericTestCase):
    """
    Defines the tests for the Clog-log model object's `__init__` function and
    its other methods.
    """
    def test_shape_ignore_msg_in_constructor(self):
        """
        Ensures that a UserWarning is raised when the 'shape_ref_pos' or
        'shape_names' keyword arguments are passed to the Clog-log model
        constructor. This warns people against expecting the Clog-log to use
        shape parameters, and alerts them that they are using an Clog-log
        model when they might have been expecting to instantiate a different
        choice model.
        """
        # Create a variable for the standard arguments to this function.
        standard_args = [self.fake_df,
                         self.alt_id_col,
                         self.obs_id_col,
                         self.choice_col,
                         self.fake_specification]

        # Create a variable for the kwargs being passed to the constructor
        kwarg_map_1 = {"intercept_ref_pos": self.fake_intercept_ref_pos,
                       "names": self.fake_names,
                       "intercept_names": self.fake_intercept_names,
                       "shape_ref_pos": 2}
        kwarg_map_2 = {"intercept_ref_pos": self.fake_intercept_ref_pos,
                       "names": self.fake_names,
                       "intercept_names": self.fake_intercept_names,
                       "shape_names": OrderedDict([("x", ["foo"])])}

        # Test to ensure that the shape ignore message is printed when using
        # either of these two kwargs
        with warnings.catch_warnings(record=True) as context:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            for pos, bad_kwargs in enumerate([kwarg_map_1, kwarg_map_2]):
                # Create a Clog-log model object with the irrelevant kwargs.
                # This should trigger a UserWarning
                clog_obj = clog.MNCL(*standard_args, **bad_kwargs)
                # Check that the warning has been created.
                self.assertEqual(len(context), pos + 1)
                self.assertIsInstance(context[-1].category, type(UserWarning))
                self.assertIn(clog._shape_ignore_msg, str(context[-1].message))

        return None

    def test_init_shapes_error_in_fit_mle(self):
        """
        Ensures that a ValueError is raised when users try to use any other
        type of initial value input methods other than the `init_vals`
        argument of `fit_mle()`. This prevents people from expecting the use
        of shape parameters to work with the Clog-log model.
        """
        # Create a variable for the arguments to the fit_mle function.
        # Note `None` is the argument passed when using the init_intercepts
        # and init_coefs keyword arguments.
        fit_args = [None]

        # Create variables for the incorrect kwargs.
        # The print_res = False arguments are to make sure strings aren't
        # printed to the console unnecessarily.
        kwargs = {"init_shapes": np.array([1, 2]),
                  "init_intercepts": self.fake_intercepts,
                  "init_coefs": self.fake_betas,
                  "print_res": False}

        # Test to ensure that the ValueError is raised when using the
        # init_shapes kwarg.
        self.assertRaises(ValueError, self.base_clog.fit_mle,
                          *fit_args, **kwargs)

        return None

    def test_ridge_warning_in_fit_mle(self):
        """
        Ensure that a UserWarning is raised when one passes the ridge keyword
        argument to the `fit_mle` method of an Clog-log model object.
        """
        # Create a variable for the fit_mle function's kwargs.
        # The print_res = False arguments are to make sure strings aren't
        # printed to the console unnecessarily.
        kwargs = {"ridge": 0.5,
                  "print_res": False}

        # Test to make sure that the ridge warning message is printed when
        # using the ridge keyword argument
        with warnings.catch_warnings(record=True) as w:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            self.base_clog.fit_mle(self.fake_all_params, **kwargs)
            self.assertGreaterEqual(len(w), 1)
            self.assertIsInstance(w[0].category, type(UserWarning))
            self.assertIn(clog._ridge_warning_msg, str(w[0].message))

        return None

    def test_init_intercepts_length_error_in_fit_mle(self):
        """
        Ensures that ValueError is raised if init_intercepts has wrong length.
        """
        # Create a variable for the arguments to the fit_mle function.
        # Note `None` is the argument passed when using the init_intercepts
        # and init_coefs keyword arguments.
        fit_args = [None]

        # Create base set of kwargs for fit_mle function
        kwargs = {"init_coefs": self.fake_betas,
                  "print_res": False}

        for i in [1, -1]:
            # This will ensure we have too many or too few intercepts
            num_intercepts = self.fake_intercepts.shape[0] + i
            kwargs["init_intercepts"] = np.arange(num_intercepts)

            # Test to ensure that the ValueError when using an init_intercepts
            # kwarg with an incorrect number of parameters
            self.assertRaises(ValueError, self.base_clog.fit_mle,
                              *fit_args, **kwargs)

        return None

    def test_init_coefs_length_error_in_fit_mle(self):
        """
        Ensures that ValueError is raised if init_coefs has wrong length.
        """
        # Create a variable for the arguments to the fit_mle function.
        # Note `None` is the argument passed when using the init_intercepts
        # and init_coefs keyword arguments.
        fit_args = [None]

        # Create base set of kwargs for fit_mle function
        kwargs = {"init_intercepts": self.fake_intercepts,
                  "print_res": False}

        # Note there is only one beta, so we can't go lower than zero betas.
        for i in [1, -1]:
            # This will ensure we have too many or too few intercepts
            num_coefs = self.fake_betas.shape[0] + i
            kwargs["init_coefs"] = np.arange(num_coefs)

            # Test to ensure that the ValueError when using an init_intercepts
            # kwarg with an incorrect number of parameters
            self.assertRaises(ValueError, self.base_clog.fit_mle,
                              *fit_args, **kwargs)

        return None
