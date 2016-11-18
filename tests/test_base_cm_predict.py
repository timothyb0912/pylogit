"""
This file tests the predict function in base_multinomial_cm_v2.py.
"""
import unittest
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix

import pylogit.asym_logit as asym
import pylogit.choice_calcs as choice_calcs
import pylogit.mixed_logit_calcs as mlc
import pylogit.mixed_logit as mixed_logit
import pylogit.nested_logit as nested_logit
import pylogit.nested_choice_calcs as nlc


class PredictFunctionTestsMixl(unittest.TestCase):
    """
    Make sure that, at least sometimes, we return expected results when using
    the predict function with a mixed logit model.
    """

    def setUp(self):
        """
        Set up a mixed logit model
        """
        # Fake random draws where Row 1 is for observation 1 and row 2 is
        # for observation 2. Column 1 is for draw 1 and column 2 is for draw 2
        self.fake_draws = mlc.get_normal_draws(2, 2, 1, seed=1)[0]
        # Create the betas to be used during the tests
        self.fake_betas = np.array([0.3, -0.6, 0.2])
        self.fake_std = 1
        self.fake_betas_ext = np.concatenate((self.fake_betas,
                                              np.array([self.fake_std])),
                                             axis=0)

        # Create the fake design matrix with columns denoting ASC_1, ASC_2, X
        self.fake_design = np.array([[1, 0, 1],
                                     [0, 1, 2],
                                     [0, 0, 3],
                                     [1, 0, 1.5],
                                     [0, 1, 2.5],
                                     [0, 0, 3.5],
                                     [1, 0, 0.5],
                                     [0, 1, 1.0],
                                     [0, 0, 1.5]])
        # Record what positions in the design matrix are being mixed over
        self.mixing_pos = [2]

        # Create the arrays that specify the choice situation, individual id
        # and alternative ids
        self.situation_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.individual_ids = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2])
        self.alternative_ids = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        # Create a fake array of choices
        self.choice_array = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0])

        # Create the 'rows_to_mixers' sparse array for this dataset
        # Denote the rows that correspond to observation 1 and observation 2
        self.obs_1_rows = np.ones(self.fake_design.shape[0])
        # Make sure the rows for observation 2 are given a zero in obs_1_rows
        self.obs_1_rows[-3:] = 0
        self.obs_2_rows = 1 - self.obs_1_rows
        # Create the row_to_mixers scipy.sparse matrix
        self.fake_rows_to_mixers = csr_matrix(self.obs_1_rows[:, None] ==
                                              np.array([1, 0])[None, :])
        # Create the rows_to_obs scipy.sparse matrix
        self.fake_rows_to_obs = csr_matrix(self.situation_ids[:, None] ==
                                           np.arange(1, 4)[None, :])
        # Create the rows_to_alts scipy.sparse matrix
        self.fake_rows_to_alts = csr_matrix(self.alternative_ids[:, None] ==
                                            np.arange(1, 4)[None, :])

        # Create the design matrix that we should see for draw 1 and draw 2
        arrays_to_join = (self.fake_design.copy(),
                          self.fake_design.copy()[:, -1][:, None])
        self.fake_design_draw_1 = np.concatenate(arrays_to_join, axis=1)
        self.fake_design_draw_2 = self.fake_design_draw_1.copy()

        # Multiply the 'random' coefficient draws by the corresponding variable
        self.fake_design_draw_1[:, -1] *= (self.obs_1_rows *
                                           self.fake_draws[0, 0] +
                                           self.obs_2_rows *
                                           self.fake_draws[1, 0])
        self.fake_design_draw_2[:, -1] *= (self.obs_1_rows *
                                           self.fake_draws[0, 1] +
                                           self.obs_2_rows *
                                           self.fake_draws[1, 1])
        extended_design_draw_1 = self.fake_design_draw_1[:, None, :]
        extended_design_draw_2 = self.fake_design_draw_2[:, None, :]
        self.fake_design_3d = np.concatenate((extended_design_draw_1,
                                              extended_design_draw_2),
                                             axis=1)

        # Create the fake systematic utility values
        self.sys_utilities_draw_1 = (self.fake_design_draw_1
                                         .dot(self.fake_betas_ext))
        self.sys_utilities_draw_2 = (self.fake_design_draw_2
                                         .dot(self.fake_betas_ext))

        #####
        # Calculate the probabilities of each alternatve in each choice
        # situation
        #####
        long_exp_draw_1 = np.exp(self.sys_utilities_draw_1)
        long_exp_draw_2 = np.exp(self.sys_utilities_draw_2)
        ind_exp_sums_draw_1 = self.fake_rows_to_obs.T.dot(long_exp_draw_1)
        ind_exp_sums_draw_2 = self.fake_rows_to_obs.T.dot(long_exp_draw_2)
        long_exp_sum_draw_1 = self.fake_rows_to_obs.dot(ind_exp_sums_draw_1)
        long_exp_sum_draw_2 = self.fake_rows_to_obs.dot(ind_exp_sums_draw_2)
        long_probs_draw_1 = long_exp_draw_1 / long_exp_sum_draw_1
        long_probs_draw_2 = long_exp_draw_2 / long_exp_sum_draw_2
        self.prob_array = np.concatenate((long_probs_draw_1[:, None],
                                          long_probs_draw_2[:, None]),
                                         axis=1)

        ###########
        # Create a mixed logit object for later use.
        ##########
        # Create a fake old long format dataframe for mixed logit model object
        self.alt_id_column = "alt_id"
        self.situation_id_column = "situation_id"
        self.obs_id_column = "observation_id"
        self.choice_column = "choice"

        data = {"x": self.fake_design[:, 2],
                self.alt_id_column: self.alternative_ids,
                self.situation_id_column: self.situation_ids,
                self.obs_id_column: self.individual_ids,
                self.choice_column: self.choice_array}
        self.fake_old_df = pd.DataFrame(data)
        self.fake_old_df["intercept"] = 1

        # Create a fake specification
        self.fake_spec = OrderedDict()
        self.fake_names = OrderedDict()

        self.fake_spec["intercept"] = [1, 2]
        self.fake_names["intercept"] = ["ASC 1", "ASC 2"]

        self.fake_spec["x"] = [[1, 2, 3]]
        self.fake_names["x"] = ["beta_x"]

        # Specify the mixing variable
        self.fake_mixing_vars = ["beta_x"]

        # Create a fake version of a mixed logit model object
        args = [self.fake_old_df,
                self.alt_id_column,
                self.situation_id_column,
                self.choice_column,
                self.fake_spec]
        kwargs = {"names": self.fake_names,
                  "mixing_id_col": self.obs_id_column,
                  "mixing_vars": self.fake_mixing_vars}
        self.mixl_obj = mixed_logit.MixedLogit(*args, **kwargs)

        # Set all the necessary attributes for prediction:
        # design_3d, coefs, intercepts, shapes, nests, mixing_pos
        self.mixl_obj.design_3d = self.fake_design_3d
        self.mixl_obj.ind_var_names += ["Sigma X"]
        self.mixl_obj.coefs = pd.Series(self.fake_betas_ext)
        self.mixl_obj.intercepts = None
        self.mixl_obj.shapes = None
        self.mixl_obj.nests = None

        return None

    def test_predict_for_mixed_logit(self):
        """
        Ensure that the predict function works for the mixed logit model.
        """
        # Alias the function that will be tested
        func = self.mixl_obj.predict
        # Get the expected results
        expected_results = self.prob_array.mean(axis=1)

        # Gather the needed arguments for the test
        args = [self.fake_old_df]
        kwargs = {"param_list": [self.fake_betas_ext, None, None, None],
                  "return_long_probs": True,
                  "choice_col": None,
                  "num_draws": 2,
                  "seed": 1}

        # Get the function results
        func_results = func(*args, **kwargs)

        # Perform the desired tests
        self.assertIsInstance(func_results, np.ndarray)
        self.assertEqual(func_results.shape, expected_results.shape)
        npt.assert_allclose(func_results, expected_results)

        # Test the function without passing param_list
        kwargs["param_list"] = None
        func_results_2 = func(*args, **kwargs)
        self.assertIsInstance(func_results_2, np.ndarray)
        self.assertEqual(func_results_2.shape, expected_results.shape)
        npt.assert_allclose(func_results_2, expected_results)

        return None


class PredictFunctionTestLogitType(unittest.TestCase):
    """
    Make sure that, at least sometimes, we return expected results when using
    the predict function with a logit-type model.
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

        # Get the mappping between rows and observations
        self.fake_rows_to_obs = csr_matrix(np.array([[1, 0],
                                                     [1, 0],
                                                     [1, 0],
                                                     [0, 1],
                                                     [0, 1]]))

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

        # Get the fitted probabilities for this model and dataset
        # Note this relies on the calc_probabilities function being functional.
        args = [self.fake_betas,
                self.fake_design,
                self.fake_df[self.alt_id_col].values,
                self.fake_rows_to_obs,
                self.fake_rows_to_alts,
                self.model_obj.utility_transform]
        kwargs = {"intercept_params": self.fake_intercepts,
                  "shape_params": self.fake_shapes,
                  "return_long_probs": True}
        self.prob_array = choice_calcs.calc_probabilities(*args, **kwargs)

        self.model_obj.coefs = pd.Series(self.fake_betas)
        self.model_obj.intercepts = pd.Series(self.fake_intercepts)
        self.model_obj.shapes = pd.Series(self.fake_shapes)
        self.model_obj.nests = None

        return None

    def test_predict_function(self):
        """
        Ensure that the predict function returns expected results for the
        asymmetric logit model.
        """
        # Alias the function that will be tested
        func = self.model_obj.predict
        # Get the expected results
        expected_results = self.prob_array

        # Gather the needed arguments for the test
        args = [self.fake_df]
        kwargs = {"param_list": [self.model_obj.coefs.values,
                                 self.model_obj.intercepts.values,
                                 self.model_obj.shapes.values,
                                 None],
                  "return_long_probs": True,
                  "choice_col": None}

        # Get the function results
        func_results = func(*args, **kwargs)

        # Perform the desired tests
        self.assertIsInstance(func_results, np.ndarray)
        self.assertEqual(func_results.shape, expected_results.shape)
        npt.assert_allclose(func_results, expected_results)

        # Test the function without passing param_list
        kwargs["param_list"] = None
        func_results_2 = func(*args, **kwargs)
        self.assertIsInstance(func_results_2, np.ndarray)
        self.assertEqual(func_results_2.shape, expected_results.shape)
        npt.assert_allclose(func_results_2, expected_results)

        return None


class PredictFunctionTestNestedLogit(unittest.TestCase):
    """
    Ensure that, at least sometimes, we return expected results when using the
    predict function with a nested logit model.
    """

    def setUp(self):
        # Create the betas to be used during the tests
        self.fake_betas = np.array([0.3, -0.6, 0.2])
        # Create the fake nest coefficients to be used during the tests
        # Note that these are the 'natural' nest coefficients, i.e. the
        # inverse of the scale parameters for each nest. They should be bigger
        # than or equal to 1.
        self.natural_nest_coefs = np.array([1 - 1e-16, 0.5])
        # Create an array of all model parameters
        self.fake_all_params = np.concatenate((self.natural_nest_coefs,
                                               self.fake_betas))
        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two.
        # The nest memberships of these alternatives are given below.
        self.fake_rows_to_nests = csr_matrix(np.array([[1, 0],
                                                       [1, 0],
                                                       [0, 1],
                                                       [1, 0],
                                                       [0, 1]]))

        # Create a sparse matrix that maps the rows of the design matrix to the
        # observatins
        self.fake_rows_to_obs = csr_matrix(np.array([[1, 0],
                                                     [1, 0],
                                                     [1, 0],
                                                     [0, 1],
                                                     [0, 1]]))

        # Create the fake design matrix with columns denoting ASC_1, ASC_2, X
        self.fake_design = np.array([[1, 0, 1],
                                     [0, 1, 2],
                                     [0, 0, 3],
                                     [1, 0, 1.5],
                                     [0, 0, 3.5]])

        # Create fake versions of the needed arguments for the MNL constructor
        self.fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                     "alt_id": [1, 2, 3, 1, 3],
                                     "choice": [0, 1, 0, 0, 1],
                                     "x": range(5),
                                     "intercept": [1 for i in range(5)]})

        # Record the various column names
        self.alt_id_col = "alt_id"
        self.obs_id_col = "obs_id"
        self.choice_col = "choice"

        # Store the choice array
        self.choice_array = self.fake_df[self.choice_col].values

        # Create a sparse matrix that maps the chosen rows of the design
        # matrix to the observatins
        self.fake_chosen_rows_to_obs = csr_matrix(np.array([[0, 0],
                                                            [1, 0],
                                                            [0, 0],
                                                            [0, 0],
                                                            [0, 1]]))

        # Create the index specification  and name dictionaryfor the model
        self.fake_specification = OrderedDict()
        self.fake_specification["intercept"] = [1, 2]
        self.fake_specification["x"] = [[1, 2, 3]]
        self.fake_names = OrderedDict()
        self.fake_names["intercept"] = ["ASC 1", "ASC 2"]
        self.fake_names["x"] = ["x (generic coefficient)"]

        # Create the nesting specification
        self.fake_nest_spec = OrderedDict()
        self.fake_nest_spec["Nest 1"] = [1, 2]
        self.fake_nest_spec["Nest 2"] = [3]

        # Create a nested logit object
        args = [self.fake_df,
                self.alt_id_col,
                self.obs_id_col,
                self.choice_col,
                self.fake_specification]
        kwargs = {"names": self.fake_names,
                  "nest_spec": self.fake_nest_spec}
        self.model_obj = nested_logit.NestedLogit(*args, **kwargs)

        self.model_obj.coefs = pd.Series(self.fake_betas)
        self.model_obj.intercepts = None
        self.model_obj.shapes = None

        def logit(x):
            return np.log(x / (1 - x))
        self.model_obj.nests = pd.Series(logit(self.natural_nest_coefs))

        # Store a ridge parameter
        self.ridge = 0.5

        # Gather the arguments needed for the calc_nested_probs function
        args = [self.natural_nest_coefs,
                self.fake_betas,
                self.model_obj.design,
                self.fake_rows_to_obs,
                self.fake_rows_to_nests]
        kwargs = {"return_type": "long_probs"}
        self.prob_array = nlc.calc_nested_probs(*args, **kwargs)

        return None

    def test_predict_function(self):
        """
        Ensure that the predict function returns expected results for the
        nested logit model.
        """
        # Alias the function that will be tested
        func = self.model_obj.predict
        # Get the expected results
        expected_results = self.prob_array

        # Gather the needed arguments for the test
        args = [self.fake_df]
        kwargs = {"param_list": [self.model_obj.coefs.values,
                                 None,
                                 None,
                                 self.model_obj.nests.values],
                  "return_long_probs": True,
                  "choice_col": None}

        # Get the function results
        func_results = func(*args, **kwargs)

        # Perform the desired tests
        self.assertIsInstance(func_results, np.ndarray)
        self.assertEqual(func_results.shape, expected_results.shape)
        npt.assert_allclose(func_results, expected_results)

        # Test the function without passing param_list
        kwargs["param_list"] = None
        func_results_2 = func(*args, **kwargs)
        self.assertIsInstance(func_results_2, np.ndarray)
        self.assertEqual(func_results_2.shape, expected_results.shape)
        npt.assert_allclose(func_results_2, expected_results)

        # Test the function when only asking for chosen_probs
        kwargs["return_long_probs"] = False
        kwargs["choice_col"] = self.choice_col
        func_results_3 = func(*args, **kwargs)
        self.assertIsInstance(func_results_3, np.ndarray)
        self.assertEqual(func_results_3.shape,
                         (self.choice_array.sum(),))
        npt.assert_allclose(func_results_3,
                            expected_results[np.where(self.choice_array)])

        return None
