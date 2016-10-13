"""
Tests for the nested_choice_calcs.py file.
"""
import unittest
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import numpy.testing as npt
import pylogit.nested_choice_calcs as nlc


# Use the following to always show the warnings
np.seterr(all='warn')
warnings.simplefilter("always")


class ComputationalSetUp(unittest.TestCase):
    """
    Defines the common setUp method used for the different type of tests.
    """
    def setUp(self):
        # Create the betas to be used during the tests
        self.fake_betas = np.array([0.3, -0.6, 0.2])
        # Create the fake nest coefficients to be used during the tests
        # Note that these are the 'natural' nest coefficients, i.e. the
        # inverse of the scale parameters for each nest.
        self.natural_nest_coefs = np.array([1, 0.5])
        # Create an array of all model parameters
        self.fake_all_params = np.concatenate((self.natural_nest_coefs,
                                               self.fake_betas))
        # The set up being used is one where there are two choice situations,
        # The first having three alternatives, and the second having only two.
        # The nest memberships of these alternatives are given below.
        self.fake_rows_to_nests = csr_matrix(np.array([[0, 1],
                                                       [0, 1],
                                                       [1, 0],
                                                       [0, 1],
                                                       [1, 0]]))

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

        return None

    def test_2d_error_in_calc_nested_probs(self):
        """
        Ensure that a NotImplementedError is raised whenever calc_nested_probs
        is called with a 2D array of nest coefficients or index coefficients.
        """
        # Create a 2D array of index coefficients
        index_2d = np.concatenate([self.fake_betas[:, None],
                                   self.fake_betas[:, None]], axis=1)

        # Create a 2D array of nest coefficients
        nest_coef_2d = np.concatenate([self.natural_nest_coefs[:, None],
                                       self.natural_nest_coefs[:, None]],
                                      axis=1)

        # Alias the function being tested
        func = nlc.calc_nested_probs

        # Get the arguments needed for the function. These are not the
        # legitimate arguments, but we just need a set of arrays to get to the
        # first argument check. 
        args = [np.arange(5) for x in range(5)]

        # Note the error message that should be raised.
        msg = "Support for 2D index_coefs or nest_coefs not yet implemented."

        for pos, array_2d in enumerate([nest_coef_2d, index_2d]):
            # Set the argument to the given 2D array.
            args[pos] = array_2d

            # Ensure that the appropriate error is raised.
            self.assertRaisesRegexp(NotImplementedError,
                                    msg,
                                    func,
                                    *args)

            # Set the argument back to None.
            args[pos] = None

        return None

    def test_return_type_error_in_calc_nested_probs(self):
        """
        Ensure that a ValueError is raised if return_type is not one of a
        handful of accepted values.
        """
        # Alias the function being tested
        func = nlc.calc_nested_probs

        # Get the arguments needed for the function. These are not the
        # legitimate arguments, but we just need a set of arrays to get to the
        # second argument check.
        args = [np.arange(5) for x in range(5)]

        # Note the error message that should be raised.
        msg = "return_type must be one of the following values: "

        # Create the kwargs to be tested
        bad_return_types = ["foo", 5, None]

        # Perform the tests
        kwargs = {"return_type": "long_probs"}
        for return_string in bad_return_types:
            kwargs["return_type"] = return_string
            self.assertRaisesRegexp(ValueError,
                                    msg,
                                    func,
                                    *args,
                                    **kwargs)

        return None

    def test_return_type_mismatch_error_in_calc_nested_probs(self):
        """
        Ensure that a ValueError is raised if return_type includes chosen_probs
        but chosen_row_to_obs is None.
        """
        # Alias the function being tested
        func = nlc.calc_nested_probs

        # Get the arguments needed for the function. These are not the
        # legitimate arguments, but we just need a set of arrays to get to the
        # third argument check.
        args = [np.arange(5) for x in range(5)]

        # Note the error message that should be raised.
        msg = "chosen_row_to_obs is None AND return_type in"

        # Create the kwargs to be tested
        bad_return_types = ['chosen_probs', 'long_and_chosen_probs']

        # Perform the tests
        kwargs = {"return_type": "long_probs",
                  "chosen_row_to_obs": None}
        for return_string in bad_return_types:
            kwargs["return_type"] = return_string
            self.assertRaisesRegexp(ValueError,
                                    msg,
                                    func,
                                    *args,
                                    **kwargs)

        return None
