"""
Tests for the scobit.py file. These tests do not include tests of
the functions that perform the mathematical calculations necessary to estimate
the multinomial Scobit model.
"""
import warnings
import unittest
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import pylogit.scobit as scobit


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
        self.fake_shapes = np.array([-1, 0, 1])

        # Create names for the intercept parameters
        self.fake_shape_names = ["Shape 1", "Shape 2", "Shape 3"]

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

        # Create the needed dataframe for the Scobit constructor
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

        # Bundle the args and kwargs used to construct the Scobit model.
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
                                   self.fake_intercept_names,
                                   "shape_names": self.fake_shape_names}

        # Initialize a basic scobit model.
        # Create the scobit model object whose coefficients will be estimated.
        self.model_obj = scobit.MNSL(*self.constructor_args,
                                     **self.constructor_kwargs)

        return None


# Note that inheritance is used to share the setUp method.
class ChoiceObjectTests(GenericTestCase):
    """
    Defines the tests for the Scobit model object's `__init__` function and
    its class methods.
    """
    def test_shape_ignore_msg_in_constructor(self):
        """
        Ensures that a UserWarning is raised when the 'shape_ref_pos' keyword
        argument is passed to the Scobit model constructor. This warns people
        against expecting the shape parameters of the Scobit model to suffer
        from identification problems. It also alerts users that they are using
        a Scobit model when they might have been expecting to instantiate a
        different choice model.
        """
        # Create a variable for the kwargs being passed to the constructor
        kwargs = deepcopy(self.constructor_kwargs)
        kwargs["shape_ref_pos"] = 2

        # Test to ensure that the shape ignore message is printed when using
        # either of these two kwargs
        with warnings.catch_warnings(record=True) as context:
            # Use this filter to always trigger the  UserWarnings
            warnings.simplefilter('always', UserWarning)

            # Create a Scobit model object with the irrelevant kwargs.
            # This should trigger a UserWarning
            scobit_obj = scobit.MNSL(*self.constructor_args, **kwargs)
            # Check that the warning has been created.
            self.assertEqual(len(context), 1)
            self.assertIsInstance(context[-1].category, type(UserWarning))
            self.assertIn(scobit._shape_ref_msg, str(context[-1].message))

        return None

    def test_ridge_warning_in_fit_mle(self):
        """
        Ensure that a UserWarning is raised when one passes the ridge keyword
        argument to the `fit_mle` method of an Scobit model object.
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

            self.model_obj.fit_mle(self.fake_all_params, **kwargs)
            self.assertGreaterEqual(len(w), 1)
            self.assertIsInstance(w[0].category, type(UserWarning))
            self.assertIn(scobit._ridge_warning_msg, str(w[0].message))

        return None

    def test_init_shapes_length_error_in_fit_mle(self):
        """
        Ensures that ValueError is raised if init_shapes has wrong length.
        """
        # Create a variable for the arguments to the fit_mle function.
        # Note `None` is the argument passed when using the init_shapes,
        # init_intercepts and init_coefs keyword arguments.
        fit_args = [None]

        # Create base set of kwargs for fit_mle function
        kwargs = {"init_intercepts": self.fake_intercepts,
                  "init_coefs": self.fake_betas,
                  "print_res": False}

        for i in [1, -1]:
            # This will ensure we have too many or too few shape parameters.
            num_shapes = self.fake_rows_to_alts.shape[1] + i
            kwargs["init_shapes"] = np.arange(num_shapes)

            # Test to ensure that the ValueError is raised when using an
            # init_shapes kwarg with an incorrect number of parameters
            self.assertRaises(ValueError, self.model_obj.fit_mle,
                              *fit_args, **kwargs)

    def test_init_intercepts_length_error_in_fit_mle(self):
        """
        Ensures that ValueError is raised if init_intercepts has wrong length.
        """
        # Create a variable for the arguments to the fit_mle function.
        # Note `None` is the argument passed when using the init_shapes,
        # init_intercepts and init_coefs keyword arguments.
        fit_args = [None]

        # Create base set of kwargs for fit_mle function
        kwargs = {"init_shapes": self.fake_shapes,
                  "init_coefs": self.fake_betas,
                  "print_res": False}

        for i in [1, -1]:
            # This will ensure we have too many or too few intercepts
            num_intercepts = self.fake_intercepts.shape[0] + i
            kwargs["init_intercepts"] = np.arange(num_intercepts)

            # Test to ensure that the ValueError when using an init_intercepts
            # kwarg with an incorrect number of parameters
            self.assertRaises(ValueError, self.model_obj.fit_mle,
                              *fit_args, **kwargs)

        return None

    def test_init_coefs_length_error_in_fit_mle(self):
        """
        Ensures that ValueError is raised if init_coefs has wrong length.
        """
        # Create a variable for the arguments to the fit_mle function.
        # Note `None` is the argument passed when using the init_shapes,
        # init_intercepts and init_coefs keyword arguments.
        fit_args = [None]

        # Create base set of kwargs for fit_mle function
        kwargs = {"init_shapes": self.fake_shapes,
                  "init_intercepts": self.fake_intercepts,
                  "print_res": False}

        # Note there is only one beta, so we can't go lower than zero betas.
        for i in [1, -1]:
            # This will ensure we have too many or too few intercepts
            num_coefs = self.fake_betas.shape[0] + i
            kwargs["init_coefs"] = np.arange(num_coefs)

            # Test to ensure that the ValueError when using an init_intercepts
            # kwarg with an incorrect number of parameters
            self.assertRaises(ValueError, self.model_obj.fit_mle,
                              *fit_args, **kwargs)

        return None

    def test_insufficient_initial_values_in_fit_mle(self):
        """
        Ensure that value errors are raised if neither init_vals OR
        (init_shapes and init_coefs) are passed.
        """
        # Create a variable for the arguments to the fit_mle function.
        # Note `None` is the argument passed when using the init_shapes,
        # init_intercepts and init_coefs keyword arguments.
        fit_args = [None]

        # Create base set of incorrect kwargs for fit_mle function
        kwargs = {"init_shapes": None,
                  "init_intercepts": self.fake_intercepts,
                  "init_coefs": None,
                  "print_res": False}

        kwargs_2 = {"init_shapes": self.fake_shapes,
                    "init_intercepts": self.fake_intercepts,
                    "init_coefs": None,
                    "print_res": False}

        kwargs_3 = {"init_shapes": None,
                    "init_intercepts": self.fake_intercepts,
                    "init_coefs": self.fake_betas,
                    "print_res": False}

        for bad_kwargs in [kwargs, kwargs_2, kwargs_3]:
            # Test to ensure that the ValueError when not passing
            # kwarg with an incorrect number of parameters
            self.assertRaises(ValueError, self.model_obj.fit_mle,
                              *fit_args, **bad_kwargs)

        return None


# As before, inheritance is used to share the setUp method.
class HelperFuncTests(GenericTestCase):
    """
    Defines tests for the 'helper' functions for estimating the Scobit model.
    """
    def test_split_param_vec(self):
        """
        Ensures that split_param_vec returns (shapes, intercepts, index_coefs)
        when called from within scobit.py.
        """
        # Store the results of split_param_vec()
        split_results = scobit.split_param_vec(self.fake_all_params,
                                               self.fake_rows_to_alts,
                                               self.fake_design)
        # Check for expected results.
        for item in split_results[1:]:
            self.assertIsInstance(item, np.ndarray)
            self.assertEqual(len(item.shape), 1)
        npt.assert_allclose(split_results[0], self.fake_shapes)
        npt.assert_allclose(split_results[1], self.fake_intercepts)
        npt.assert_allclose(split_results[2], self.fake_betas)

        return None

    def test_scobit_utility_transform(self):
        """
        Ensures that `_scobit_utility_transform()` returns correct results
        """
        # Create a set of systematic utilities that will test the function for
        # correct calculations, for proper dealing with overflow, and for
        # proper dealing with underflow.

        # The first and third elements tests general calculation.
        # The second element of index_array should lead to the transformation
        # equaling the intercept for alternative 2.
        # The fourth element should test what happens with underflow and should
        # lead to max_comp_value + ASC 1.
        # The fifth element should test what happens with overflow and should
        # lead to 50 * np.exp(1)
        index_array = np.array([1,
                                0,
                                -1,
                                50,
                                -50])

        # Crerate the array of expected results
        intercept_1 = self.fake_intercepts[0]
        intercept_3 = 0
        shape_1 = np.exp(self.fake_shapes[0])
        shape_3 = np.exp(self.fake_shapes[2])

        result_1 = (intercept_1 -
                    np.log((1.0 + np.exp(-1 * index_array[0]))**shape_1 - 1.0))

        result_3 = (intercept_3 -
                    np.log((1.0 + np.exp(-1 * index_array[2]))**shape_3 - 1.0))

        expected_results = np.array([result_1,
                                     self.fake_intercepts[1],
                                     result_3,
                                     scobit.max_comp_value + intercept_1,
                                     intercept_3 - 50 * np.exp(1)])[:, None]

        # Use the utility transformation function
        args = [index_array,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_alts,
                self.fake_shapes,
                self.fake_intercepts]
        kwargs = {"intercept_ref_pos": self.fake_intercept_ref_pos}
        func_results = scobit._scobit_utility_transform(*args, **kwargs)

        # Check the correctness of the result
        self.assertIsInstance(func_results, np.ndarray)
        self.assertEqual(len(func_results.shape), 2)
        self.assertEqual(func_results.shape[1], expected_results.shape[1])
        self.assertEqual(func_results.shape[0], expected_results.shape[0])
        npt.assert_allclose(expected_results, func_results)

        return None

    def test_scobit_transform_deriv_v(self):
        """
        Tests basic behavior of the scobit_transform_deriv_v.
        """
        # Note the index has a value that is small and a value that is large to
        # test whether or not the function correctly uses L'Hopital's rule to
        # deal with underflow and overflow when calculating the derivative.
        # When the index is small, the derivative should be the associated
        # shape parameter value. When the index is large the derivative should
        # be 1.
        test_index = np.array([-2, 0, 2, -800, 300])
        # Note we use a compressed sparse-row matrix so that we can easily
        # convert the output matrix to a numpy array using the '.A' attribute.
        num_rows = test_index.shape[0]
        test_output = diags(np.ones(num_rows),
                            0, format='csr')

        # Bundle the arguments needed for _scobit_transform_deriv_v()
        args = [test_index,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_alts,
                self.fake_shapes]

        # Get the derivative using the function defined in clog_log.py.
        derivative = scobit._scobit_transform_deriv_v(*args,
                                                      output_array=test_output)

        # Initialize an array of correct results
        # Note the second element is by design where each of the terms in the
        # derivative should evaluate to 1 and we get 1 / 1 = 1.
        correct_derivatives = np.array([np.nan,
                                        1.0,
                                        np.nan,
                                        np.exp(self.fake_shapes[0]),
                                        1.0])

        # Calculate 'by hand' what the correct results should be
        for i in [0, 2]:
            shape = np.exp(self.fake_shapes[i])
            numerator = (shape * 
                         np.exp(-test_index[i]) * 
                         np.power(1 + np.exp(-test_index[i]), shape - 1))
            denominator = np.power(1 + np.exp(-test_index[i]), shape) - 1
            correct_derivatives[i] = numerator / denominator

        self.assertIsInstance(derivative, type(test_output))
        self.assertEqual(len(derivative.shape), 2)
        self.assertEqual(derivative.shape, (num_rows, num_rows))
        npt.assert_allclose(correct_derivatives,
                            np.diag(derivative.A))

        return None

    def test_scobit_transform_deriv_alpha(self):
        """
        Ensures that scobit_transform_deriv_alpha returns the `output_array`
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
            derivative_results = scobit._scobit_transform_deriv_alpha(*args,
                                                                      **kwargs)
            if test_output is None:
                self.assertIsNone(derivative_results)
            else:
                npt.assert_allclose(test_output.A, derivative_results.A)

        return None