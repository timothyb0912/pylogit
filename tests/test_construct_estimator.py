"""
Tests for the construct_estimator.py file.
"""
import unittest
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix, eye

import pylogit.asym_logit as asym
import pylogit.conditional_logit as mnl
import pylogit.clog_log as clog
import pylogit.scobit as scobit
import pylogit.uneven_logit as uneven
import pylogit.mixed_logit_calcs as mlc
import pylogit.mixed_logit as mixed_logit
import pylogit.nested_logit as nested_logit
import pylogit.construct_estimator as constructor

class ConstructorTests(unittest.TestCase):
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

    def make_uneven_and_scobit_models(self):
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
        fake_shapes = np.array([-1, 1, 2])

        # Create names for the intercept parameters
        fake_shape_names = ["Shape 1", "Shape 2", "Shape 3"]

        # Create an array of all model parameters
        fake_all_params = np.concatenate((fake_shapes,
                                          fake_intercepts,
                                          fake_betas))

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

        # Create the needed dataframe for the model constructor
        fake_df = pd.DataFrame({"obs_id": [1, 1, 1, 2, 2],
                                "alt_id": [1, 2, 3, 1, 3],
                                "choice": [0, 1, 0, 0, 1],
                                "x": fake_design[:, 0],
                                "intercept": [1 for i in range(5)]})

        # Record the various column names
        alt_id_col = "alt_id"
        obs_id_col = "obs_id"
        choice_col = "choice"

        # Create the index specification  and name dictionary for the model
        fake_specification = OrderedDict()
        fake_names = OrderedDict()
        fake_specification["x"] = [[1, 2, 3]]
        fake_names["x"] = ["x (generic coefficient)"]

        # Bundle args and kwargs used to construct the choice models.
        constructor_args = [fake_df,
                            alt_id_col,
                            obs_id_col,
                            choice_col,
                            fake_specification]

        # Create a variable for the kwargs being passed to the constructor
        constructor_kwargs = {"intercept_ref_pos": fake_intercept_ref_pos,
                              "names": fake_names,
                              "intercept_names": fake_intercept_names,
                              "shape_names": fake_shape_names}

        # Initialize the various choice models
        uneven_obj = uneven.MNUL(*constructor_args, **constructor_kwargs)
        scobit_obj = scobit.MNSL(*constructor_args, **constructor_kwargs)

        for model_obj in [uneven_obj, scobit_obj]:
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
        return uneven_obj, scobit_obj

    def make_clog_and_mnl_models(self):
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

        # Create an array of all model parameters
        fake_all_params = np.concatenate((fake_intercepts, fake_betas))

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

        # Create the needed dataframe for the model constructor
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

        mnl_spec = OrderedDict()
        mnl_names = OrderedDict()
        mnl_spec["intercept"] =[1, 2]
        mnl_names["intercept"] = fake_intercept_names
        mnl_spec["x"] = fake_specification["x"]
        mnl_names["x"] = fake_names["x"]

        # Bundle args and kwargs used to construct the Asymmetric Logit model.
        clog_args = [fake_df,
                     alt_id_col,
                     obs_id_col,
                     choice_col,
                     fake_specification]
        mnl_args = deepcopy(clog_args)
        mnl_args[-1] = mnl_spec

        # Create a variable for the kwargs being passed to the constructor
        clog_kwargs = {"names": fake_names,
                       "intercept_ref_pos": fake_intercept_ref_pos,
                       "intercept_names": fake_intercept_names}
        mnl_kwargs = {"names": mnl_names}

        # Initialize a basic Asymmetric Logit model whose coefficients will be
        # estimated.
        clog_obj = clog.MNCL(*clog_args, **clog_kwargs)
        mnl_obj = mnl.MNL(*mnl_args, **mnl_kwargs)

        # Create the desired model attributes for the clog log model
        clog_obj.coefs = pd.Series(fake_betas, index=fake_names["x"])
        clog_obj.intercepts =\
            pd.Series(fake_intercepts, index=fake_intercept_names)
        clog_obj.shapes = None
        clog_obj.nests = None
        clog_obj.params =\
            pd.concat([clog_obj.intercepts, clog_obj.coefs],
                      axis=0, ignore_index=False)

        mnl_obj.params = clog_obj.params.copy()
        mnl_obj.coefs = mnl_obj.params.copy()
        mnl_obj.intercepts = None
        mnl_obj.shapes = None
        mnl_obj.nests = None

        return clog_obj, mnl_obj

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
        return model_obj

    def setUp(self):
        """
        Create the real model objects.
        """
        self.asym_model = self.make_asym_model()
        self.uneven_model, self.scobit_model =\
            self.make_uneven_and_scobit_models()
        self.clog_model, self.mnl_model = self.make_clog_and_mnl_models()
        self.mixed_model = self.make_mixed_model()
        self.nested_model = self.make_nested_model()
        return None

    def test_create_estimation_obj(self):
        # Alias the function being tested
        func = constructor.create_estimation_obj

        # Take note of the models that are being used in this test
        models = [self.mnl_model,
                  self.clog_model,
                  self.asym_model,
                  self.scobit_model,
                  self.uneven_model,
                  self.nested_model,
                  self.mixed_model]

        # Perform the desired tests
        for model_obj in models:
            # Get the internal model name
            internal_model_name =\
                constructor.display_name_to_model_type[model_obj.model_type]
            # Get the estimation object class
            estimation_class = (constructor.model_type_to_resources
                                            [internal_model_name]
                                            ['estimator'])
            # Get the function results
            args = [model_obj, model_obj.params.values]
            kwargs = {"mappings": model_obj.get_mappings_for_fit(),
                      "ridge": 0.25,
                      "constrained_pos": [0],
                      "weights": np.ones(model_obj.data.shape[0])}

            # Make sure the function result is of the correct class.
            func_result = func(*args, **kwargs)
            self.assertIsInstance(func_result, estimation_class)
            for key in ['ridge', 'constrained_pos', 'weights']:
                expected_value = kwargs[key]
                self.assertTrue(hasattr(func_result, key))
                func_value = getattr(func_result, key)
                if isinstance(expected_value, np.ndarray):
                    npt.assert_allclose(expected_value, func_value)
                else:
                    self.assertEqual(expected_value, func_value)

        return None
