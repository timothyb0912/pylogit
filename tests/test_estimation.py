"""
Use this file to test methods and classes in test_estimation.py
"""
import warnings
import unittest
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix

import pylogit.asym_logit as asym
import pylogit.estimation as estimation

# Use the following to always show the warnings
np.seterr(all='warn')
warnings.simplefilter("always")


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


class EstimationObjTests(GenericTestCase):
    """
    Store the tests for the basic methods in the EstimationObj class.
    """

    def test_constructor(self):
        # Create a zero vector
        zero_vector = np.zeros(self.fake_all_params.shape[0])
        # Create a ridge parameter
        ridge_param = 0.5
        # Split parameter function
        split_param_func = asym.split_param_vec
        # Store the mapping dictionaries
        mapping_dict = self.model_obj.get_mappings_for_fit()
        # Store the positions of the parameters to be constrained
        constrained_pos = [0]
        # Create the kewargs for the estimation object
        kwargs = {"constrained_pos": constrained_pos}

        # Create the estimation object
        estimation_object = estimation.EstimationObj(self.model_obj,
                                                     mapping_dict,
                                                     ridge_param,
                                                     zero_vector,
                                                     split_param_func,
                                                     **kwargs)

        # Perform the tests to ensure that the desired attributes were
        # correctly created
        attr_names = ["alt_id_vector",
                      "choice_vector",
                      "design",
                      "intercept_ref_pos",
                      "shape_ref_pos",
                      "rows_to_obs",
                      "rows_to_alts",
                      "chosen_row_to_obs",
                      "rows_to_nests",
                      "rows_to_mixers",
                      "ridge",
                      "constrained_pos",
                      "zero_vector",
                      "split_params",
                      "utility_transform",
                      "block_matrix_idxs",
                      "calc_dh_dv",
                      "calc_dh_d_alpha",
                      "calc_dh_d_shape"]
        for attr in attr_names:
            self.assertTrue(hasattr(estimation_object, attr))

        # Make sure that the objects that should be arrays, are arrays
        for attr in ["alt_id_vector",
                     "choice_vector",
                     "design",
                     "zero_vector"]:
            self.assertIsInstance(getattr(estimation_object, attr), np.ndarray)
        # Ensure that the arrays have the correct values
        npt.assert_allclose(estimation_object.alt_id_vector,
                            self.model_obj.alt_IDs)
        npt.assert_allclose(estimation_object.choice_vector,
                            self.model_obj.choices)
        npt.assert_allclose(estimation_object.design, self.model_obj.design)
        npt.assert_allclose(estimation_object.zero_vector, zero_vector)

        # Ensure that the scalars are scalars with the correct values
        for attr in ["intercept_ref_pos", "shape_ref_pos", "ridge"]:
            self.assertIsInstance(getattr(estimation_object, attr),
                                  (int, float, long))
        self.assertEqual(estimation_object.intercept_ref_pos,
                         self.model_obj.intercept_ref_position)
        self.assertEqual(estimation_object.shape_ref_pos,
                         self.model_obj.shape_ref_position)
        self.assertEqual(estimation_object.ridge, ridge_param)

        # Ensure that the mapping matrices are correct
        for attr in ["rows_to_obs", "rows_to_alts", "chosen_row_to_obs",
                     "rows_to_nests", "rows_to_mixers"]:
            # Get the mapping matrix as stored on the model object.
            matrix_on_object = getattr(estimation_object, attr)
            if matrix_on_object is not None:
                npt.assert_allclose(matrix_on_object.A, mapping_dict[attr].A)
            else:
                self.assertIsNone(mapping_dict[attr])

        # Ensure that the function definitions point to the correct locations
        self.assertEqual(id(estimation_object.split_params),
                         id(split_param_func))
        self.assertEqual(id(estimation_object.utility_transform),
                         id(self.model_obj.utility_transform))

        # Make sure that the derivative functions return None, for now.
        for attr in ["calc_dh_dv",
                     "calc_dh_d_alpha",
                     "calc_dh_d_shape"]:
            func = getattr(estimation_object, attr)
            self.assertIsNone(func("foo"))

        return None

    def test_not_implemented_error_in_example_functions(self):
        # Create a zero vector
        zero_vector = np.zeros(self.fake_all_params.shape[0])
        # Create a ridge parameter
        ridge_param = 0.5
        # Split parameter function
        split_param_func = asym.split_param_vec
        # Store the mapping dictionaries
        mapping_dict = self.model_obj.get_mappings_for_fit()
        # Store the positions of the parameters to be constrained
        constrained_pos = [0]
        # Create the kwargs for the estimation object
        kwargs = {"constrained_pos": constrained_pos}

        # Create the estimation object
        estimation_object = estimation.EstimationObj(self.model_obj,
                                                     mapping_dict,
                                                     ridge_param,
                                                     zero_vector,
                                                     split_param_func,
                                                     **kwargs)

        # Record the names of the methods that are created as examples
        example_methods = ["convenience_calc_probs",
                           "convenience_calc_log_likelihood",
                           "convenience_calc_gradient",
                           "convenience_calc_hessian",
                           "convenience_calc_fisher_approx"]
        for method_name in example_methods:
            func = getattr(estimation_object, method_name)
            error_msg = "Method should be defined by descendant classes"
            self.assertRaisesRegexp(NotImplementedError,
                                    error_msg,
                                    func,
                                    None)

        return None
