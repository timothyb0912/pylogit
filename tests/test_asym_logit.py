"""
Tests for the asym_logit.py file. These tests do not include tests of
the functions that perform the mathematical calculations necessary to estimate
the multinomial Asymmetric Logit model.
"""
import warnings
import unittest
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import pylogit.asym_logit as asym

try:
    # in Python 3 range returns an iterator instead of list
    # to maintain backwards compatibility use "old" version of range
    from past.builtins import xrange
except ImportError:
    pass

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

        return None


# Note that inheritance is used to share the setUp method.
class ChoiceObjectTests(GenericTestCase):
    """
    Defines the tests for the Asymmetric Logit model's `__init__` function and
    its class methods.
    """

    def test_shape_ignore_msg_in_constructor(self):
        """
        Ensures that a UserWarning is raised when the 'shape_ref_pos' keyword
        argument is not an integer. This warns people against expecting all of
        the shape parameters of the Asymmetric Logit model to be identified.
        It also alerts users that they are using a Asymmetric Logit model when
        they might have been expecting to instantiate a different choice model.
        """
        # Create a variable for the kwargs being passed to the constructor
        kwargs = deepcopy(self.constructor_kwargs)

        for pos, item in enumerate(['foo', None]):
            kwargs["shape_ref_pos"] = item

            # Check that the warning has been created.
            # Test to ensure that the ValueError is raised when using a
            # shape_ref_pos kwarg with an incorrect number of parameters
            self.assertRaises(ValueError, asym.MNAL,
                              *self.constructor_args, **kwargs)

        return None

    def test_ridge_warning_in_fit_mle(self):
        """
        Ensure that a UserWarning is raised when one passes the ridge keyword
        argument to the `fit_mle` method of an Asymmetric Logit model object.
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
            self.assertIn(asym._ridge_warning_msg, str(w[0].message))

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
            num_shapes = self.fake_rows_to_alts.shape[1] - 1 + i
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
            num_intercepts = self.fake_rows_to_alts.shape[1] - 1 + i
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

    def test_keyword_argument_constructor_in_fit_mle(self):
        """
        Ensures that the init_vals object can be successfully created from the
        various init_shapes, init_intercepts, and init_coefs arguments.
        """
        # Create a variable for the arguments to the fit_mle function.
        # Note `None` is the argument passed when using the init_shapes,
        # init_intercepts and init_coefs keyword arguments.
        fit_args = [None]

        # Create base set of incorrect kwargs for fit_mle function (note the
        # ridge is the thing that is incorrect)
        kwargs_1 = {"init_shapes": self.fake_shapes,
                    "init_intercepts": self.fake_intercepts,
                    "init_coefs": self.fake_betas,
                    "ridge": "foo",
                    "print_res": False}

        kwargs_2 = {"init_shapes": self.fake_shapes,
                    "init_coefs": self.fake_betas,
                    "ridge": "foo",
                    "print_res": False}

        # Test to ensure that the raised ValueError is printed when using
        # either of these two kwargs. This ensures that we were able to
        # create the init_vals object since the ridge error check is after
        # the creation of this argurment.
        for kwargs in [kwargs_1, kwargs_2]:
            self.assertRaisesRegexp(TypeError,
                                    "ridge",
                                    self.model_obj.fit_mle,
                                    *fit_args,
                                    **kwargs)

        return None

    def test_init_vals_length_error_in_fit_mle(self):
        """
        Ensures that ValueError is raised if init_vals has wrong length.
        """
        # Note there is only one beta, so we can't go lower than zero betas.
        original_intercept_ref_position = self.fake_intercept_ref_pos
        for intercept_ref_position in [None, original_intercept_ref_position]:
            self.model_obj.intercept_ref_position = intercept_ref_position
            for i in [1, -1]:
                # This will ensure we have too many or too few intercepts
                num_coefs = self.fake_betas.shape[0] + i

                # Test to ensure that the ValueError when using an
                # init_intercepts kwarg with an incorrect number of parameters
                self.assertRaisesRegexp(ValueError,
                                        "dimension",
                                        self.model_obj.fit_mle,
                                        np.arange(num_coefs),
                                        print_res=False)

        return None


# As before, inheritance is used to share the setUp method.
class HelperFuncTests(GenericTestCase):
    """
    Defines tests for the 'helper' functions for estimating the Asymmetric
    Logit model.
    """

    def test_split_param_vec_with_intercepts(self):
        """
        Ensures that split_param_vec returns (shapes, intercepts, index_coefs)
        when called from within asym_logit.py.
        """
        # Store the results of split_param_vec()
        split_results = asym.split_param_vec(self.fake_all_params,
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

    def test_split_param_vec_without_intercepts(self):
        """
        Ensures that split_param_vec returns (shapes, intercepts, index_coefs)
        when called from within asym_logit.py.
        """
        # Store the results of split_param_vec()
        shapes_and_betas = np.concatenate([self.fake_shapes,
                                           self.fake_betas])
        split_results = asym.split_param_vec(shapes_and_betas,
                                             self.fake_rows_to_alts,
                                             self.fake_design)
        # Check for expected results.
        for idx in [0, 2]:
            self.assertIsInstance(split_results[idx], np.ndarray)
            self.assertEqual(len(split_results[idx].shape), 1)
        npt.assert_allclose(split_results[0], self.fake_shapes)
        self.assertIsNone(split_results[1])
        npt.assert_allclose(split_results[2], self.fake_betas)

        return None

    def test_convert_eta_to_c(self):
        """
        Check general transformation of estimated shape parameters to the
        original parameterization. Check overflow handling.
        """
        # Create a 2d array of shapes, where each row represents another
        # vector of shapes to be transformed and tested
        test_shapes = np.array([[-1, 1],
                                [800, 1],
                                [-1, -800],
                                [0, 0]])

        #####
        # Determine the expected results
        #####
        expected_results = np.ones((test_shapes.shape[0], 3))
        expected_results[-1] = 1.0 / 3
        # Explicitly calculate the results for the first row
        vals_1 = np.array([test_shapes[0, 0], test_shapes[0, 1], 0])
        exp_vals_1 = np.exp(vals_1)
        denom_1 = exp_vals_1.sum()
        expected_results[0] = exp_vals_1 / denom_1
        # Explicitly calculate the results for the middle rows, taking care of
        # overflow and underflow.
        vals_2 = np.array([test_shapes[1, 0], test_shapes[1, 1], 0])
        exp_vals_2 = np.exp(vals_2)
        exp_vals_2[0] = asym.max_comp_value
        denom_2 = exp_vals_2.sum()
        expected_results[1] = exp_vals_2 / denom_2

        vals_3 = np.array([test_shapes[2, 0], test_shapes[2, 1], 0])
        exp_vals_3 = np.exp(vals_3)
        exp_vals_3[1] = asym.min_comp_value
        denom_3 = exp_vals_3.sum()
        expected_results[2] = exp_vals_3 / denom_3

        # Use _convert_eta_to_c and compare the results for 1D inputs
        for i in xrange(test_shapes.shape[0]):
            func_results = asym._convert_eta_to_c(test_shapes[i],
                                                  self.fake_shape_ref_pos)

            # Make sure the results are correct
            self.assertIsInstance(func_results, np.ndarray)
            self.assertEqual(len(func_results.shape), 1)
            self.assertEqual(func_results.shape[0], expected_results.shape[1])
            npt.assert_allclose(func_results, expected_results[i])

        # Check the 2D inputs. Note that for 2D ndarray, we need the array of
        # shape parameters to be transposed such that we have an array of
        # shape (num_estimated_shapes, num_samples_of_parameters).
        # To adequately compare, we need to transpose expected_results too.
        func_results_2d = asym._convert_eta_to_c(test_shapes.T,
                                                 self.fake_shape_ref_pos)
        # Make sure the results are correct
        self.assertIsInstance(func_results_2d, np.ndarray)
        self.assertEqual(len(func_results_2d.shape), 2)
        self.assertEqual(func_results_2d.shape, expected_results.T.shape)
        npt.assert_allclose(func_results_2d, expected_results.T)

        return None

    def test_asym_utility_transform(self):
        """
        Ensures that `_asym_utility_transform()` returns correct results
        """
        # Create a set of systematic utilities that will test the function for
        # correct calculations, for proper dealing with overflow, and for
        # proper dealing with underflow.

        # The first and third elements tests general calculation.
        # The second element of index_array should lead to the transformation
        # equaling the 'natural' shape parameter for alternative 2.
        # The fourth element should test what happens with underflow and should
        # lead to max_comp_value.
        # The fifth element should test what happens with overflow and should
        # lead to -1.0 * max_comp_value
        index_array = np.array([1, 0, -1, 1e400, -1e400])

        # We can use a the following array of the shape parameters to test
        # the underflow capabilities with respect to the shape
        # parameters.
        test_shapes_2 = np.array([-800, 0])

        test_shapes_3 = np.array([800, 0])

        # Figure out the value of the 'natural' shape parameters
        natural_shapes_2 = asym._convert_eta_to_c(test_shapes_2,
                                                  self.fake_shape_ref_pos)
        natural_shapes_3 = asym._convert_eta_to_c(test_shapes_3,
                                                  self.fake_shape_ref_pos)

        # Crerate the array of expected results when using shape parameters
        # of 'normal' magnitudes.
        intercept_1 = self.fake_intercepts[0]
        intercept_2 = self.fake_intercepts[1]
        intercept_3 = 0

        result_1 = (intercept_1 +
                    np.log(self.natural_shapes[0]) * (1 - index_array[0]))

        result_2 = intercept_2 + np.log(self.natural_shapes[1])

        # Note the division by 2 is due to the 'J - 1' term. See the original
        # definition of the transformation.
        result_3 = (intercept_3 +
                    np.log(self.natural_shapes[2]) -
                    np.log((1 - self.natural_shapes[2]) / 2) * index_array[2])

        expected_results = np.array([result_1,
                                     result_2,
                                     result_3,
                                     asym.max_comp_value + intercept_1,
                                     - asym.max_comp_value])[:, None]

        # Crerate the array of expected results when using shape parameters
        # of 'abnormally' small magnitudes.
        # Note the division by 2 is due to the 'J - 1' term. See the original
        # definition of the transformation.
        result_2_2 = intercept_2 + np.log(natural_shapes_2[1])
        result_3_2 = (intercept_3 +
                      np.log(natural_shapes_2[2]) -
                      np.log((1 - natural_shapes_2[2]) / 2) * index_array[2])

        # Note the '0' comes from (1-1) * ln(shape)
        expected_results_2 = np.array([0 + intercept_1,
                                       result_2_2,
                                       result_3_2,
                                       asym.max_comp_value + intercept_1,
                                       -asym.max_comp_value])[:, None]

        # Create the array of expected results when using shape parameters
        # of 'abnormally' large magnitudes.
        result_2_3 = intercept_2 + np.log(natural_shapes_3[1])
        result_3_3 = (intercept_3 +
                      np.log(natural_shapes_3[2]) -
                      np.log((1 - natural_shapes_3[2]) / 2) * index_array[2])

        expected_results_3 = np.array([0 + intercept_1,
                                       result_2_3,
                                       result_3_3,
                                       0 + intercept_1,
                                       -asym.max_comp_value])[:, None]

        #####
        # Perform various rounds of checking
        #####
        # Use the utility transformation function, round_1
        alt_id_vals = self.fake_df[self.alt_id_col].values
        args = [index_array,
                alt_id_vals,
                self.fake_rows_to_alts,
                self.fake_shapes,
                self.fake_intercepts]
        kwargs = {"intercept_ref_pos": self.fake_intercept_ref_pos,
                  "shape_ref_position": self.fake_shape_ref_pos}
        func_results = asym._asym_utility_transform(*args, **kwargs)

        # Use the utility transformation function, round_2
        args[3] = test_shapes_2
        func_results_2 = asym._asym_utility_transform(*args, **kwargs)

        # Use the utility transformation function, round_3
        args[3] = test_shapes_3
        func_results_3 = asym._asym_utility_transform(*args, **kwargs)

        # Check the correctness of the results
        all_results = [(func_results, expected_results),
                       (func_results_2, expected_results_2),
                       (func_results_3, expected_results_3)]

        for pos, (test_results, correct_results) in enumerate(all_results):
            self.assertIsInstance(test_results, np.ndarray)
            self.assertEqual(len(test_results.shape), 2)
            self.assertEqual(test_results.shape[1], correct_results.shape[1])
            self.assertEqual(test_results.shape[0], correct_results.shape[0])
            npt.assert_allclose(test_results, correct_results)

        return None

    def test_asym_utility_transform_2d(self):
        """
        Ensures that `_asym_utility_transform()` returns correct results when
        called with 2 dimensional systematic utility arrays and
        """
        # Create a set of systematic utilities that will test the function for
        # correct calculations, for proper dealing with overflow, and for
        # proper dealing with underflow.

        # The first and third elements tests general calculation.
        # The second element of index_array should lead to the transformation
        # equaling the 'natural' shape parameter for alternative 2.
        # The fourth element should test what happens with underflow and should
        # lead to max_comp_value.
        # The fifth element should test what happens with overflow and should
        # lead to -1.0 * max_comp_value
        index_array = np.array([1, 0, -1, 1e400, -1e400])
        index_array_2d = np.concatenate([index_array[:, None],
                                         index_array[:, None]],
                                        axis=1)

        # Create 2d array of shapes
        shapes_2d = np.concatenate([self.fake_shapes[:, None],
                                    self.fake_shapes[:, None]],
                                   axis=1)

        # Create 2d array of intercepts
        intercepts_2d = np.concatenate([self.fake_intercepts[:, None],
                                        self.fake_intercepts[:, None]],
                                       axis=1)

        # We can use a the following array of the shape parameters to test
        # the underflow capabilities with respect to the shape
        # parameters.
        test_shapes_2 = np.array([-800, 0])
        test_shapes_2_2d = np.concatenate([test_shapes_2[:, None],
                                           test_shapes_2[:, None]],
                                          axis=1)

        test_shapes_3 = np.array([800, 0])
        test_shapes_3_2d = np.concatenate([test_shapes_3[:, None],
                                           test_shapes_3[:, None]],
                                          axis=1)

        # Figure out the value of the 'natural' shape parameters
        natural_shapes_2 = asym._convert_eta_to_c(test_shapes_2,
                                                  self.fake_shape_ref_pos)
        natural_shapes_3 = asym._convert_eta_to_c(test_shapes_3,
                                                  self.fake_shape_ref_pos)

        # Crerate the array of expected results when using shape parameters
        # of 'normal' magnitudes.
        intercept_1 = self.fake_intercepts[0]
        intercept_2 = self.fake_intercepts[1]
        intercept_3 = 0

        result_1 = (intercept_1 +
                    np.log(self.natural_shapes[0]) * (1 - index_array[0]))

        result_2 = intercept_2 + np.log(self.natural_shapes[1])

        # Note the division by 2 is due to the 'J - 1' term. See the original
        # definition of the transformation.
        result_3 = (intercept_3 +
                    np.log(self.natural_shapes[2]) -
                    np.log((1 - self.natural_shapes[2]) / 2) * index_array[2])

        expected_results = np.array([result_1,
                                     result_2,
                                     result_3,
                                     asym.max_comp_value + intercept_1,
                                     - asym.max_comp_value])[:, None]

        # Crerate the array of expected results when using shape parameters
        # of 'abnormally' small magnitudes.
        # Note the division by 2 is due to the 'J - 1' term. See the original
        # definition of the transformation.
        result_2_2 = intercept_2 + np.log(natural_shapes_2[1])
        result_3_2 = (intercept_3 +
                      np.log(natural_shapes_2[2]) -
                      np.log((1 - natural_shapes_2[2]) / 2) * index_array[2])

        # Note the '0' comes from (1-1) * ln(shape)
        expected_results_2 = np.array([0 + intercept_1,
                                       result_2_2,
                                       result_3_2,
                                       asym.max_comp_value + intercept_1,
                                       -asym.max_comp_value])[:, None]

        # Create the array of expected results when using shape parameters
        # of 'abnormally' large magnitudes.
        result_2_3 = intercept_2 + np.log(natural_shapes_3[1])
        result_3_3 = (intercept_3 +
                      np.log(natural_shapes_3[2]) -
                      np.log((1 - natural_shapes_3[2]) / 2) * index_array[2])

        expected_results_3 = np.array([0 + intercept_1,
                                       result_2_3,
                                       result_3_3,
                                       0 + intercept_1,
                                       -asym.max_comp_value])[:, None]

        #####
        # Perform various rounds of checking
        #####
        # Use the utility transformation function, round_1
        alt_id_vals = self.fake_df[self.alt_id_col].values
        args = [index_array_2d,
                alt_id_vals,
                self.fake_rows_to_alts,
                shapes_2d,
                intercepts_2d]
        kwargs = {"intercept_ref_pos": self.fake_intercept_ref_pos,
                  "shape_ref_position": self.fake_shape_ref_pos}
        func_results = asym._asym_utility_transform(*args, **kwargs)

        # Use the utility transformation function, round_2
        args[3] = test_shapes_2_2d
        func_results_2 = asym._asym_utility_transform(*args, **kwargs)

        # Use the utility transformation function, round_3
        args[3] = test_shapes_3_2d
        func_results_3 = asym._asym_utility_transform(*args, **kwargs)

        # Check the correctness of the results
        all_results = [(func_results, expected_results),
                       (func_results_2, expected_results_2),
                       (func_results_3, expected_results_3)]

        for pos, (test_results, correct_results) in enumerate(all_results):
            self.assertIsInstance(test_results, np.ndarray)
            self.assertEqual(len(test_results.shape), 2)
            self.assertEqual(test_results.shape[1], 2)
            self.assertEqual(test_results.shape[0], correct_results.shape[0])
            for col in [0, 1]:
                npt.assert_allclose(test_results[:, col][:, None],
                                    correct_results)

        return None

    def test_asym_transform_deriv_v(self):
        """
        Tests basic behavior of the asym_transform_deriv_v. Essentially the
        only things that can go wrong is underflow or overflow from the shape
        parameters going to zero or 1.0.
        """
        # Declare  set of index values to be tested
        test_index = np.array([1, 0, -1, -2, 2])
        # Figure out how many alternatives are there
        num_alts = self.fake_rows_to_alts.shape[1]
        # Test what happens with large shape parameters (that result in
        # 'natural' shape parameters near 1.0)
        large_shapes = np.array([800, 0])
        large_results = np.array([0,
                                  -np.log(asym.min_comp_value),
                                  np.log(num_alts - 1),
                                  asym.max_comp_value,
                                  -np.log(asym.min_comp_value)])
        # Test what happens with large shape parameters (that result in
        # 'natural' shape parameters near 1.0)
        small_shapes = np.array([-800, 0])
        natural_small_shapes = asym._convert_eta_to_c(small_shapes,
                                                      self.fake_shape_ref_pos)
        small_results = np.array([-np.log(natural_small_shapes[0]),
                                  -np.log(natural_small_shapes[1]),
                                  (np.log(num_alts - 1) -
                                   np.log(1 - natural_small_shapes[2])),
                                  (np.log(num_alts - 1) -
                                   np.log(1 - natural_small_shapes[0])),
                                  -np.log(natural_small_shapes[2])])

        # Bundle the arguments needed for _asym_transform_deriv_v()
        args = [test_index,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_alts,
                self.fake_shapes]

        # Create the needed output array
        num_rows = test_index.shape[0]
        output = diags(np.ones(num_rows), 0, format='csr')

        # Validate the results from asym_transform_deriv_v
        for shape_array, results in [(large_shapes, large_results),
                                     (small_shapes, small_results)]:
            # Get the reslts from asym_transform_deriv_v
            args[-1] = shape_array
            kwargs = {"ref_position": self.fake_shape_ref_pos,
                      "output_array": output}
            derivative = asym._asym_transform_deriv_v(*args, **kwargs)

            # Ensure the results are as expected
            self.assertIsInstance(derivative, type(output))
            self.assertEqual(len(derivative.shape), 2)
            self.assertEqual(derivative.shape, (num_rows, num_rows))
            npt.assert_allclose(np.diag(derivative.A), results)

        return None

    def test_asym_transform_deriv_alpha(self):
        """
        Ensures that asym_transform_deriv_alpha returns the `output_array`
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
            derivative_results = asym._asym_transform_deriv_alpha(*args,
                                                                  **kwargs)
            if test_output is None:
                self.assertIsNone(derivative_results)
            else:
                npt.assert_allclose(test_output.A, derivative_results.A)

        return None

    def test_calc_deriv_c_with_respect_to_eta(self):
        """
        Ensure the calculation of the derivative of the natural shape
        parameters with respect to the reparameterized shape parameters is
        correct.
        """
        # Specify some shape parameters to test the function
        natural_test_shapes = np.array([0.5, 0.25, 0.25])
        # Specify an alternative to serve as the reference parameter
        ref_position = 2
        # Create an output array that will contain the derivative
        output = np.zeros((natural_test_shapes.shape[0],
                           natural_test_shapes.shape[0] - 1))

        # Record the correct derivatives
        interim_results = np.array([[0.5 - 0.25, -0.5 * 0.25, -0.5 * 0.25],
                                    [-0.5 * 0.25, 0.25 - 0.0625, -0.25**2],
                                    [-0.5 * 0.25, -0.25**2, 0.25 - 0.0625]])
        correct_results = interim_results[:, [0, 1]]

        # Get the results from the function
        deriv_args = (natural_test_shapes, ref_position)
        kwargs = {'output_array': output}
        function_results = asym._calc_deriv_c_with_respect_to_eta(*deriv_args,
                                                                  **kwargs)

        # Ensure that the results are what I expect them to be
        self.assertIsInstance(function_results, np.ndarray)
        self.assertEqual(len(function_results.shape), 2)
        self.assertEqual(function_results.shape, output.shape)
        npt.assert_allclose(function_results, correct_results)

        return None

    def test_asym_transform_deriv_shape(self):
        """
        Ensures that the asym_transform_deriv_shape() function provides
        correct results, and handles all forms of overflow and underflow
        correctly.
        """
        # Create index values to test the function with.
        test_index = np.array([1, 0, -2, -1e400, 1e400])

        # Figure out how many alternatives are there
        num_alts = self.fake_rows_to_alts.shape[1]

        # Test what happens with shape parameters of 'normal' magnitude
        regular_shapes = self.fake_shapes
        natural_reg_shapes = self.natural_shapes

        result_2 = (natural_reg_shapes[2]**-1 +
                    test_index[2] / (1.0 - natural_reg_shapes[2]))

        regular_results = np.array([0,
                                    natural_reg_shapes[1]**-1,
                                    result_2,
                                    -asym.max_comp_value,
                                    -asym.max_comp_value])

        # Test what happens with large shape parameters (that result in
        # 'natural' shape parameters near 1.0)
        large_shapes = np.array([800, 0])
        natural_large_shapes = asym._convert_eta_to_c(large_shapes,
                                                      self.fake_shape_ref_pos)
        large_results = np.array([0,
                                  asym.max_comp_value,
                                  asym.max_comp_value,
                                  -asym.max_comp_value,
                                  -asym.max_comp_value])

        # Test what happens with large shape parameters (that result in
        # 'natural' shape parameters near 1.0)
        small_index = test_index.copy()
        small_index[0] = 0.5

        small_shapes = np.array([-800, 0])
        natural_small_shapes = asym._convert_eta_to_c(small_shapes,
                                                      self.fake_shape_ref_pos)
        result_3_3 = (natural_small_shapes[2]**-1 +
                      test_index[2] / (1 - natural_small_shapes[2]))
        small_results = np.array([asym.max_comp_value,
                                  natural_small_shapes[1]**-1,
                                  result_3_3,
                                  -asym.max_comp_value,
                                  -asym.max_comp_value])

        # Bundle the arguments needed for _asym_transform_deriv_shape()
        args = [test_index,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_alts,
                self.fake_shapes]

        # Create the needed output array for the overall derivative with
        # respect to the reparameterized shape parameters
        num_rows = test_index.shape[0]
        output = np.matrix(np.zeros((num_rows, num_alts - 1)), dtype=float)

        # Create the needed output array for the intermediate derivative of the
        # 'natural' shape parameters with respect to the reparameterized shape
        # parameters.
        deriv_output = np.zeros((num_alts, num_alts - 1), dtype=float)

        # Create the array needed for dh_dc
        dh_dc_array = self.fake_rows_to_alts.copy()

        # Create a function that will take the reparameterized shapes and the
        # shape reference position and return the derivative of the natural
        # shapes with respect to the reparameterized shapes.
        interim_deriv_func = partial(asym._calc_deriv_c_with_respect_to_eta,
                                     output_array=deriv_output)

        # Take note of the kwargs needed for asym_transform_deriv_shape()
        deriv_kwargs = {"ref_position": self.fake_shape_ref_pos,
                        "dh_dc_array": dh_dc_array,
                        "fill_dc_d_eta": interim_deriv_func,
                        "output_array": output}

        # Validate the results from asym_transform_deriv_shape
        test_arrays = [(regular_shapes, natural_reg_shapes, regular_results),
                       (large_shapes, natural_large_shapes, large_results),
                       (small_shapes, natural_small_shapes, small_results)]
        for shape_array, natural_shape_array, interim_results in test_arrays:
            if shape_array[0] == small_shapes[0]:
                args[0] = small_index

            # Get the reslts from asym_transform_deriv_shape()
            args[-1] = shape_array
            derivative = asym._asym_transform_deriv_shape(*args,
                                                          **deriv_kwargs)

            # Calculate the derivative of the 'natural' shape parameters with
            # espect to the reparameterized shape parameters.
            interim_args = (natural_shape_array, self.fake_shape_ref_pos)
            kwargs = {"output_array": deepcopy(deriv_output)}
            dc_d_eta = asym._calc_deriv_c_with_respect_to_eta(*interim_args,
                                                              **kwargs)

            # Place the interim results in the correct shape
            interim_results = interim_results[:, None]
            interim_results = self.fake_rows_to_alts.multiply(interim_results)

            # Calculate the correct results
            results = interim_results.A.dot(dc_d_eta)

            # Ensure the results are as expected
            self.assertIsInstance(derivative, type(output))
            self.assertEqual(len(derivative.shape), 2)
            self.assertEqual(derivative.shape, (num_rows, num_alts - 1))
            npt.assert_allclose(derivative.A, results)

        return None
