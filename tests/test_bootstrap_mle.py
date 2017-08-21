"""
Tests for the bootstrap_mle.py file.
"""
import unittest
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pandas as pd

import pylogit.bootstrap_mle as bmle


class HelperTests(unittest.TestCase):
    def test_get_model_abbrev(self):
        good_types = bmle.model_type_to_display_name.values()
        bad_type = "fakeType"

        # Alias the function to be tested.
        func = bmle.get_model_abbrev

        # Note the error message that should be raised.
        err_msg = "Model object has an unknown or incorrect model type."

        # Create a fake object class.
        class FakeModel(object):
            def __init__(self, fake_model_type):
                self.model_type = fake_model_type

        # Perform the tests that should pass.
        for model_type in good_types:
            current_obj = FakeModel(model_type)
            self.assertIn(func(current_obj), bmle.model_type_to_display_name)

        # Perform a test that should fail. Ensure the correct error is raised.
        current_obj = FakeModel(bad_type)
        self.assertRaisesRegexp(ValueError,
                                err_msg,
                                func,
                                current_obj)

        return None

    def test_get_model_creation_kwargs(self):
        # Create a fake object class.
        class FakeModel(object):
            def __init__(self, fake_model_type, fake_attr=None):
                self.model_type = fake_model_type
                if isinstance(fake_attr, dict):
                    for key in fake_attr:
                        setattr(self, key, fake_attr[key])

        # Create a fake model object for the tests
        fake_type = bmle.model_type_to_display_name.values()[0]
        fake_attr_dict = {"name_spec": OrderedDict([("intercept", "ASC 1")]),
                          "intercept_names": None,
                          "intercept_ref_position": None,
                          "shape_names": None,
                          "shape_ref_position": None,
                          "nest_spec": None,
                          "mixing_vars": None,
                          "mixing_id_col": None}
        fake_obj = FakeModel(fake_type, fake_attr=fake_attr_dict)

        # Alias the function being tested
        func = bmle.get_model_creation_kwargs

        # Perform the desired tests
        func_results = func(fake_obj)
        self.assertIsInstance(func_results, dict)

        expected_keys = ["model_type", "names", "intercept_names",
                         "intercept_ref_pos", "shape_names", "shape_ref_pos",
                         "nest_spec", "mixing_vars", "mixing_id_col"]
        self.assertTrue(all([x in func_results for x in expected_keys]))

        self.assertEqual(func_results["model_type"],
                         bmle.model_type_to_display_name.keys()[0])

        self.assertEqual(func_results["names"], fake_attr_dict["name_spec"])

        for key in expected_keys[2:]:
            self.assertIsNone(func_results[key])

        return None

    def test_extract_default_init_vals(self):
        # Set a desired number of parameters
        num_params = 9

        # Create a fake object class.
        class FakeModel(object):
            def __init__(self, fake_model_type, fake_attr=None):
                self.model_type = fake_model_type
                if isinstance(fake_attr, dict):
                    for key in fake_attr:
                        setattr(self, key, fake_attr[key])

        # Create an array of MNL parameter estimates
        mnl_point_array = np.arange(1, 6)
        mnl_param_names = ["ASC 1", "ASC 2", "ASC 3", "x1", "x2"]
        mnl_point_series =\
            pd.Series(mnl_point_array, index=mnl_param_names)

        # Create the fake model object
        fake_attr_dict = {"intercept_names": mnl_param_names[:3],
                          "ind_var_names": mnl_param_names[3:],
                          "mixing_vars": None}
        fake_obj = FakeModel(bmle.model_type_to_display_name.keys()[0],
                             fake_attr=fake_attr_dict)

        # Create the array that we expect to be returned
        expected_array = np.concatenate([np.zeros(4), mnl_point_array], axis=0)

        # Alias the function being tested
        func = bmle.extract_default_init_vals

        # Perform the desired tests
        func_result = func(fake_obj, mnl_point_series, num_params)
        npt.assert_allclose(expected_array, func_result)

        new_fake_attrs = deepcopy(fake_attr_dict)
        new_fake_attrs["intercept_names"] = None
        new_fake_obj = FakeModel(bmle.model_type_to_display_name.keys()[0],
                                 fake_attr=new_fake_attrs)
        new_results = func(new_fake_obj, mnl_point_series, num_params)
        npt.assert_allclose(expected_array, new_results)

        new_fake_attrs_2 = deepcopy(fake_attr_dict)
        new_fake_attrs_2["intercept_names"] = None
        new_fake_attrs_2["mixing_vars"] = ["ASC 1"]
        new_fake_obj = FakeModel(bmle.model_type_to_display_name.keys()[0],
                                 fake_attr=new_fake_attrs_2)
        new_results_2 = func(new_fake_obj, mnl_point_series, num_params)

        new_expected_array =\
            np.concatenate([np.zeros(3), mnl_point_array, np.zeros(1)], axis=0)
        npt.assert_allclose(new_expected_array, new_results_2)

        return None
