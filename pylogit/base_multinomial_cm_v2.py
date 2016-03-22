# -*- coding: utf-8 -*-
"""
Created on Tues Feb 22 09:30:44 2016

@author: Timothy Brathwaite
@notes:  Credit is due to Akshay Vij and John Canny for the idea of using 
         "mapping" matrices to avoid the need for "for loops" when computing 
         quantities of interest such as probabilities, log-likelihoods, 
         gradients, and hessians. This code is based on an earlier multinomial 
         logit implementation by Akshay Vij which made use of such mappings.
         
         This version differs from version 1 by partitioning the parameters to
         be estimated, theta, as shape parameters, intercept parameters, and
         index coefficients.
"""

import pickle
from copy import deepcopy

import scipy.linalg
import scipy.stats
import numpy as np
import pandas as pd

from choice_tools import create_design_matrix, create_long_form_mappings
from choice_calcs import calc_probabilities, calc_asymptotic_covariance
            
def _get_dataframe_from_data(data):
    """
    data:       string or pandas dataframe. If string, data should be an
                absolute or relative path to a CSV file containing the long
                format data for this choice model. Note long format is has one 
                row per available alternative for each observation. If pandas
                dataframe, the dataframe should be the long format data for the
                choice model.
    ====================
    Returns:    pandas dataframe of the long format data for the choice model.
    """
    if isinstance(data, str):
        if ".csv" == data[-4:]:
            dataframe = pd.read_csv(data)
        else:
            msg_1 = "data = {} is of unknown file type."
            msg_2 = " Please pass path to csv."
            raise ValueError(msg_1.format(data) + msg_2)
    elif isinstance(data, pd.DataFrame):
        dataframe = data
    else:
        msg_1 = "type(data) = {} is an invalid type."
        msg_2 = " Please pass pandas dataframe or path to csv."
        raise ValueError(msg_1.format(type(data)) + msg_2)
        
    return dataframe

# Create a basic class that sets the structure for the discrete outcome models
# to be specified later. MNDC stands for MultiNomial Discrete Choice.
class MNDC_Model(object):
    def __init__(self, data, alt_id_col, 
                 obs_id_col, choice_col,
                 specification,
                 intercept_ref_pos=None,
                 shape_ref_pos=None,
                 names=None,
                 intercept_names=None,
                 shape_names=None,
                 model_type=""):
        """
        data:               string or pandas dataframe. If string, data should
                            be an absolute or relative path to a CSV file 
                            containing the long format data for this choice 
                            model. Note long format is has one row per 
                            available alternative for each observation. If 
                            pandas dataframe, the dataframe should be the long 
                            format data for the choice model.
                        
        alt_id_col:         string. Should denote the column in data which 
                            contains the alternative identifiers for each row.
                        
        obs_id_col:         string. Should denote the column in data which 
                            contains the observation identifiers for each row.
                        
        choice_col:         string. Should denote the column in data which 
                            contains the ones and zeros that denote whether or 
                            not the given row corresponds to the chosen 
                            alternative for the given individual.
                        
        specification:      OrderedDict. Keys are a proper subset of the 
                            columns in long_form_df. Values are either a list 
                            or a single string, "all_diff" or "all_same". If a 
                            list, the elements should be:
                            1) single objects that are within the alternative  
                               ID column of long_form_df
                            2) lists of objects that are within the alternative
                               ID column of long_form_df.
                            For each single object in the list, a unique column
                            will be created (i.e. there will be a unique 
                            coefficient for that variable in the corresponding
                            utility equation of the corresponding alternative).
                            For lists within the specification_dict values, a 
                            single column will be created for all the 
                            alternatives within iterable (i.e. there will be 
                            one common coefficient for the variables in the 
                            iterable).
                        
        intercept_ref_pos:  OPTIONAL. int. Valid only when the intercepts being
                            estimated are not part of the index. Specifies the 
                            alternative in the ordered array of unique 
                            alternative ids whose intercept or 
                            alternative-specific constant is not estimated, to 
                            ensure model identifiability. Default == None. 
                        
        shape_ref_pos:      OPTIONAL. int. Specifies the alternative in the 
                            ordered array of unique alternative ids whose shape
                            parameter is not estimated, to ensure model 
                            identifiability. Default == None. 
                        
        names:              OPTIONAL. OrderedDict. Should have the same keys as
                            specification_dict. For each key:
                            1) if the corresponding value in specification_dict
                               is "all_same", then there should be a single 
                               string as the value in names.
                            2) if the corresponding value in specification_dict 
                               is "all_diff", then there should be a list of 
                               strings as the value in names. There should be 
                               one string in the value in names for each 
                               possible alternative.
                            3) if the corresponding value in specification_dict
                               is a list, then there should be a list of 
                               strings as the value in names. There should be 
                               one string the value in names per item in the 
                               value in specification_dict.
                            Default == None.
                            
        intercept_names:    OPTIONAL. list or None. If a list is passed, then 
                            the list should have the same number of elements as
                            there are possible alternatives in data, minus 1. 
                            Each element of the list should be a string--the 
                            name of the corresponding alternative's intercept
                            term, in sorted order of the possible alternative 
                            IDs. If None is passed, the resulting names that
                            are shown in the estimation results will be 
                            ["Outside_ASC_{}".format(x) for x in shape_names]. 
                            Default = None.
                        
        shape_names:        OPTIONAL. list or None. If a list is passed, then 
                            the list should have the same number of elements as
                            there are possible alternative IDs in data. Each 
                            element of the list should be a string denoting the
                            name of the corresponding alternative, in sorted 
                            order of the possible alternative IDs. The 
                            resulting names which are shown in the estimation 
                            results will be 
                            ["shape_{}".format(x) for x in shape_names]. 
                            Default = None.
        
        model_type:         OPTIONAL. string. Denotes the model type of the 
                            choice model being instantiated. Default == "".
        ====================
        Returns:        None.   
        """
        dataframe = _get_dataframe_from_data(data)
        
        ##########
        # Make sure all necessary columns are in the dataframe        
        ##########
        for column in [alt_id_col, obs_id_col, choice_col]:
            try:
                assert column in dataframe.columns
            except AssertionError as e:
                print("{} not in data.columns".format(column))
                raise e
        
        ##########
        # Make sure the various 'name' arguments are of the correct lengths
        ##########
        # Get a sorted array of all possible alternative ids in the dataset
        all_ids = np.sort(dataframe[alt_id_col].unique())
        
        # Check for correct length of shape_names and intercept_names
        for alt_param_names, alt_ref_pos, alt_param_string in [(shape_names,
                                                                shape_ref_pos,
                                                                "shape_names"),
                                                          (intercept_names,
                                                           intercept_ref_pos,
                                                           "intercept_names")]:
            if alt_param_names is not None:
                if alt_ref_pos is None:
                    if alt_param_string == "intercept_names":
                        msg = "At least one intercept should be constrained"
                        raise ValueError(msg)
                    alt_params_not_estimated = 0
                elif isinstance(alt_ref_pos, int): 
                    alt_params_not_estimated = 1
                else:
                    msg = "Ref position is of the wrong type. "
                    msg_2 = "Should be an integer"
                    raise AssertionError(msg + msg_2)
                try:
                    cond_1 = len(alt_param_names) == (len(all_ids) - 
                                                  alt_params_not_estimated)
                    assert cond_1
                except AssertionError as e:
                    print("{} is of the wrong length".format(alt_param_string))
                    print("len({}) == {}".format(alt_param_string,
                                                 len(alt_param_names)))
                    print("The correct length is: {}".format(len(all_ids) - 
                                                     alt_params_not_estimated))
                    raise e
        
        ##########
        # Add an intercept column to the data if necessary based on the model
        # specification.
        ##########
        condition_1 = "intercept" in specification
        condition_2 = "intercept" not in dataframe.columns
        
        if condition_1 and condition_2:
            dataframe["intercept"] = 1.0
            
        ##########
        # Make sure all the columns in the specification dict are all
        # in the dataframe
        ##########
        problem_cols = []
        dataframe_cols = dataframe.columns
        for key in specification:
            if key not in dataframe_cols:
                problem_cols.append(key)
        if problem_cols != []:
            msg = "The following keys in the specification are not in the 'data':"
            print(msg)
            print(problem_cols)
            raise ValueError
            
        ##########
        # Make sure that the columns we are using in the specification are all
        # numeric and exclude positive or negative infinity variables.
        ##########
        problem_cols = []
        for col in specification:
            # The condition below checks for positive or negative inifinity
            # values.            
            if np.isinf(dataframe[col]).any():
                problem_cols.append(col)
            # The condition below checks for values that are not real numbers
            # This will catch values that are strings.
            elif not np.isreal(dataframe[col]).all():
                problem_cols.append(col)
                
        if problem_cols != []:
            msg = "The following columns contain either +/- inifinity values "
            msg_2 = "or values that are not real numbers (e.g. strings):"
            print(msg + msg_2)
            print(problem_cols)
            raise ValueError
            
        ##########
        # Create the design matrix for this model
        ##########
        design_res =  create_design_matrix(dataframe, 
                                           specification,
                                           alt_id_col, 
                                           names=names)
        ##########
        # Store needed data
        ##########
        self.data = dataframe
        self.name_spec = names
        self.design = design_res[0]
        self.ind_var_names = design_res[1]
        self.alt_id_col = alt_id_col
        self.obs_id_col = obs_id_col
        self.choice_col = choice_col
        self.specification = specification
        self.alt_IDs = dataframe[alt_id_col].values
        self.choices = dataframe[choice_col].values
        self.model_type = model_type   
        self.shape_names = shape_names
        self.intercept_names = intercept_names
        self.shape_ref_position = shape_ref_pos
        self.intercept_ref_position = intercept_ref_pos
        
        return None
        
    def get_mappings_for_fit(self, dense=False):
        """
        dense:      OPTIONAL. bool. Determines whether or not sparse matrices 
                    will be returned or dense numpy arrays.
        ====================
        """
        return create_long_form_mappings(self.data,
                                         self.obs_id_col,
                                         self.alt_id_col,
                                         choice_col=self.choice_col,
                                         dense=dense)
                                          
    def store_fit_results(self, estimation_res):
        """
        estimation_res: dictionary. The estimation result dictionary that is
                        output from scipy.optimize.minimize. In addition to the
                        standard keys which are included, it should also 
                        contain the following keys: 
                        ["final_gradient", "final_hessian", "fisher_info"].
                        The "final_gradient", "final_hessian", and 
                        "fisher_info" values should be the gradient, hessian,
                        and Fisher-Information Matrix of the log likelihood,
                        evaluated at the final parameter vector.
        ====================
        Returns:        None. Will calculate and store a variety of attributes
                        to the model instance.s
        """
        # Alias for the dictionary of estimation results for easy access
        results_dict = estimation_res[0]
        
        # Store the log-likelilhood, fitted probabilities, residuals, and
        # individual chi-square statistics
        self.log_likelihood = estimation_res[1]
        self.fitted_probs = estimation_res[2]
        self.long_fitted_probs = estimation_res[3]
        self.long_residuals = estimation_res[4]
        self.ind_chi_squareds = estimation_res[5]
        self.chi_square = self.ind_chi_squareds.sum()
        
        # Store the 'estimation success' of the optimization
        self.estimation_success = results_dict["success"]
        self.estimation_message = results_dict["message"]
        
        # Account for peculiar things from the em algorithm
        if "log_likelihood_progress" in results_dict:
            self.log_likelihood_progression =\
                            np.array(results_dict["log_likelihood_progress"])
        else:
            self.log_likelihood_progression = None
        if "estimation_reason" in results_dict:
            self.em_estimation_reason = results_dict["estimation_reason"]
        else:
            self.em_estimation_reason = None
        
        # Store the summary measures of the model fit
        self.rho_squared = results_dict["rho_squared"]
        self.rho_bar_squared = results_dict["rho_bar_squared"]
        
        # Store the initial and null log-likelihoods
        self.null_log_likelihood = results_dict["log_likelihood_null"]
        
        # Initialize the lists of all parameter names and all parameter values       
        all_names = deepcopy(self.ind_var_names)
        all_params = [deepcopy(results_dict["utility_coefs"])]

        ##########
        # Figure out whether this model had shape parameters or 
        # intercept parameters and store each of these appropriately
        ##########
        if results_dict["intercept_params"] is not None:
            # Identify the number of intercept parameters
            num_intercepts = results_dict["intercept_params"].shape[0]
            
            # Get the names of the intercept parameters
            if self.intercept_names is None:
                intercept_names = ["Outside_ASC_{}".format(x) for x in 
                                   range(1, num_intercepts + 1)]
            else:
                intercept_names = self.intercept_names
                
            # Store the names of the intercept parameters                        
            all_names = intercept_names + all_names
            # Store the values of the intercept parameters
            all_params.insert(0, results_dict["intercept_params"])
            
            # Store the shape parameters
            self.intercepts = pd.Series(results_dict["intercept_params"],
                                        index=intercept_names,
                                        name="intercept_parameters")
        else:
            self.intercepts = None

        if results_dict["shape_params"] is not None:
            # Identify the number of shape parameters
            num_shapes = results_dict["shape_params"].shape[0]
            
            # Get the names of the shape parameters
            if self.shape_names is None:
                shape_names = ["shape_{}".format(x) for x in 
                               range(1, num_shapes + 1)]
            else:
                shape_names = self.shape_names
                
            # Store the names of the shape parameters                        
            all_names = shape_names + all_names
            # Store the values of the shape parameters
            all_params.insert(0, results_dict["shape_params"])
            
            # Store the shape parameters
            self.shapes = pd.Series(results_dict["shape_params"],
                                    index=shape_names,
                                    name="shape_parameters")
        else:
            self.shapes = None
        
        ##########
        # Store the model results and values needed for model inference
        ##########
        # Store the utility coefficients        
        self.coefs = pd.Series(results_dict["utility_coefs"],
                               index=self.ind_var_names,
                               name="coefficients")
        
        # Store the gradient 
        self.gradient = pd.Series(results_dict["final_gradient"],
                               index=all_names,
                               name="gradient")
                               
        # Store the hessian 
        self.hessian = pd.DataFrame(results_dict["final_hessian"],
                                    columns=all_names,
                                    index=all_names)
                                    
        # Store the variance-covariance matrix
        self.cov = pd.DataFrame(-1 * scipy.linalg.inv(self.hessian),
                                columns=all_names,
                                index=all_names)
                                
        # Store all of the estimated parameters
        self.params = pd.Series(np.concatenate(all_params, axis=0),
                                index=all_names,
                                name="parameters")
                               
        # Store the standard errors
        self.standard_errors = pd.Series(np.sqrt(np.diag(self.cov)),
                                         index=all_names,
                                         name="std_err")
                                         
        # Store the t-stats of the estimated parameters
        self.tvalues = self.params / self.standard_errors
        self.tvalues.name = "t_stats"
        
        # Store the p-values
        self.pvalues = pd.Series(2 *
                                 scipy.stats.norm.sf(np.abs(self.tvalues)),
                                 index=all_names,
                                 name="p_values")
                                 
        # Store the fischer information matrix of estimated coefficients
        self.fisher_information = pd.DataFrame(results_dict["fisher_info"],
                                               columns=all_names,
                                               index=all_names)
                                               
        # Store the 'robust' variance-covariance matrix
        self.robust_cov = calc_asymptotic_covariance(self.hessian,
                                                     self.fisher_information)
                                                      
        # Store the 'robust' standard errors
        self.robust_std_errs = pd.Series(np.sqrt(np.diag(self.robust_cov)),
                                         index=all_names,
                                         name="robust_std_err")
                                         
        # Store the 'robust' t-stats of the estimated coefficients
        self.robust_t_stats = self.params / self.robust_std_errs
        self.robust_t_stats.name = "robust_t_stats"
        
        # Store the 'robust' p-values
        self.robust_p_vals = pd.Series(2 * 
                          scipy.stats.norm.sf(np.abs(self.robust_t_stats)),
                                index=all_names,
                                name="robust_p_values")
        
        ##########                            
        # Store a summary dataframe of the estimation results
        # (base it on statsmodels summary dataframe/table perhaps?)
        ##########
        self.summary = pd.concat((self.params,
                                  self.standard_errors,
                                  self.tvalues,
                                  self.pvalues,
                                  self.robust_std_errs,
                                  self.robust_t_stats,
                                  self.robust_p_vals), axis=1)
                                  
        ##########
        # Record values for the fit_summary and statsmodels table
        ##########
        # Record the number of observations        
        self.nobs = self.fitted_probs.shape[0]
        # This is the number of estimated parameters        
        self.df_model = self.params.shape[0]
        # The number of observations minus the number of estimated parameters
        self.df_resid = self.nobs - self.df_model
        # This is just the log-likelihood. The opaque name is used for
        # conformance with statsmodels
        self.llf = self.log_likelihood
        # This is just a repeat of the standard errors
        self.bse = self.standard_errors
                                  
        ##########
        # Store a "Fit Summary"        
        ##########
        self.fit_summary = pd.Series([self.df_model,
                                      self.nobs,
                                      self.null_log_likelihood,
                                      self.log_likelihood,
                                      self.rho_squared,
                                      self.rho_bar_squared,
                                      self.estimation_message],
                                     index=["Number of Parameters",
                                            "Number of Observations",
                                            "Null Log-Likelihood",
                                            "Fitted Log-Likelihood",
                                            "Rho-Squared",
                                            "Rho-Bar-Squared",
                                            "Estimation Message"])
                            
        
    def fit_mle(self, init_vals, 
                print_res=True, method="BFGS",
                loss_tol=1e-06, gradient_tol=1e-06,
                maxiter=1000, ridge=None,
                 *args):
        """
        init_vals:      1D numpy array of the initial values to start the 
                        optimizatin process with. There should be one value
                        for each utility coefficient and shape parameter being
                        estimated.
                        
        print_res:      OPTIONAL. Bool. Determines whether the timing and
                        initial and final log likelihood results will be 
                        printed as they they are determined.
                        
        method:         OPTIONAL. String. Should be a valid string which can
                        be passed to scipy.optimize.minimize. Determines the
                        optimization algorithm which is used for this problem.
                        
        loss_tol:       OPTIONAL. Float. Determines the tolerance on the 
                        difference in objective function values from one 
                        iteration to the next which is needed to determine
                        convergence. Default = 1e-06.
                        
        gradient_tol:   OPTIONAL. Float. Determines the tolerance on the 
                        difference in gradient values from one 
                        iteration to the next which is needed to determine
                        convergence. Default = 1e-06.
                        
        ridge:          OPTIONAl. int, float, long, or None. Determines whether 
                        or not ridge regression is performed. If an int, float 
                        or long is passed, then that scalar determines the 
                        ridge penalty for the optimization. Default = None.
        ====================
        Returns:        None. Results of the estimation process are saved to
                        the model instance.
        """
                    
        print("This model class' fit_mle method has not been constructed.")
        raise NotImplementedError
        
        return None
    
    
    def print_summaries(self):
        """
        Returns None. Will print the measures of fit and the 
        estimation results for the  model.
        """
        if hasattr(self, "fit_summary") and hasattr(self, "summary"):
            print("\n")
            print(self.fit_summary)
            print("=" * 30)
            print(self.summary)
            
        else:
            msg = "This {} object has not yet been estimated so there "
            msg_2 = "are no estimation summaries to print."
            print(msg.format(self.model_type) + msg_2)
            
        return None
    
            
    def conf_int(self, alpha=0.05, cols=None, return_df=False):
        """
        alpha:      OPTIONAL. float. Between 0.0 and 1.0. 
                    Default = 0.05.
        
        cols:       OPTIONAL. array-like iterable. Should contains 
                    strings that denote the columns that one wants 
                    the confidence intervals for. Default = None 
                    because that will return the confidence 
                    interval for all variables.
                    
        return_df:  OPTIONAL. bool. Determines whether the returned
                    value will be a dataframe or a numpy array.
                    Default = False.
        ====================
        Returns:    pandas dataframe or numpy array, depending on
                    return_df kwarg. The first column contains
                    the lower bound to the confidence interval
                    whereas the second column contains the upper
                    values of the confidence intervals.
        """
        
        # Get the critical z-value for alpha / 2
        z_critical = scipy.stats.norm.ppf(1.0 - alpha / 2.0,
                                          loc=0, scale=1)
                                          
        # Calculate the lower and upper values for the confidence interval.
        lower = self.params - z_critical * self.standard_errors
        upper = self.params + z_critical * self.standard_errors
        
        # Combine the various series.
        combined = pd.concat((lower, upper), axis=1)
        
        # Subset the combined dataframe if need be.
        if cols is not None:
            combined = combined.loc[cols, :]
            
        # Return the desired object, whether dataframe or array
        if return_df:
            return combined
        else:
            return combined.values
        
    def get_statsmodels_summary(self, 
                                title=None, 
                                alpha=.05):
        """
        title:      OPTIONAL. string or None. Will be
                    the title of the returned summary.
                    If None, the default title is used.
                    
        alpha:      OPTIONAL. float. Between 0.0 and 
                    1.0. Determines the width of the
                    displayed, (1 - alpha)% confidence 
                    interval.
        ====================
        Returns:    statsmodels.summary object.
        """
        try:
            # Get the statsmodels Summary class            
            from statsmodels.iolib.summary import Summary
            # Get an instantiation of the Summary class.
            smry = Summary()

            # Get the yname and yname_list.
            # Note I'm not really sure what the yname_list is.
            new_yname, new_yname_list = self.choice_col, None

            # Get the model name
            model_name = self.model_type

            ##########
            # Note the following commands are basically directly from
            # statsmodels.discrete.discrete_model
            ##########
            top_left = [('Dep. Variable:', None),
                         ('Model:', [model_name]),
                         ('Method:', ['MLE']),
                         ('Date:', None),
                         ('Time:', None),
                #('No. iterations:', ["%d" % self.mle_retvals['iterations']]),
                         ('converged:', [str(self.estimation_success)])
                          ]

            top_right = [('No. Observations:', ["{:,}".format(self.nobs)]),
                         ('Df Residuals:', ["{:,}".format(self.df_resid)]),
                         ('Df Model:', ["{:,}".format(self.df_model)]),
                         ('Pseudo R-squ.:',
                          ["{:.3f}".format(self.rho_squared)]),
                         ('Pseudo R-bar-squ.:',
                          ["{:.3f}".format(self.rho_bar_squared)]),
                         ('Log-Likelihood:', ["{:,.3f}".format(self.llf)]),
                         ('LL-Null:', 
                          ["{:,.3f}".format(self.null_log_likelihood)]),
                         ]

            if title is None:
                title = model_name + ' ' + "Regression Results"

            xnames = self.params.index.tolist()

            # for top of table
            smry.add_table_2cols(self,
                                 gleft=top_left,
                                 gright=top_right, #[],
                                 yname=new_yname, 
                                 xname=xnames,
                                 title=title)
            # for parameters, etc
            smry.add_table_params(self,
                                  yname=[new_yname_list],
                                  xname=xnames,
                                  alpha=alpha,
                                  use_t=False)
            return smry
        except:
            print("statsmodels not installed. Resorting to standard summary")
            return self.print_summaries()
        
    
    def predict(self, data):
        """
        data:       string or pandas dataframe. If string, data should be 
                    an absolute or relative path to a CSV file containing 
                    the long format data for this choice model. Note long 
                    format is has one row per available alternative for 
                    each observation. If pandas dataframe, the dataframe 
                    should be the long format data for the choice model.
                    The data should include all of the same columns as the
                    original data used to construct the choice model, with
                    the sole exception of the "intercept" column. If needed
                    the "intercept" column will be dynamically created.
        ====================
        Returns:    1D numpy array with one element per observation per 
                    available alternative for the given observation. Elements
                    will be the probability of the given observation being
                    associated with the row's corresponding alternative.
        """
        dataframe = _get_dataframe_from_data(data)
        
        # Determine the conditions under which we will add an intercept column
        # to our long format dataframe.
        condition_1 = "intercept" in self.specification  
        condition_2 = "intercept" not in dataframe.columns
        
        if condition_1 and condition_2:
            dataframe["intercept"] = 1.0
        
        # Make sure the necessary columns are in the long format dataframe
        for column in [self.alt_id_col,
                       self.obs_id_col]:
            try:
                assert column in dataframe.columns
            except AssertionError as e:
                print("{} not in data.columns".format(column))
                raise e
        
        # Get the new column of alternative IDs and get the new design matrix
        new_alt_IDs = dataframe[self.alt_id_col].values
        
        new_design_res = create_design_matrix(dataframe, 
                                              self.specification,
                                              self.alt_id_col, 
                                              names=self.name_spec)
        
        new_design = new_design_res[0]
        
        # Get the new mappings between the alternatives and observations        
        new_mapping_res = create_long_form_mappings(dataframe,
                                                    self.obs_id_col,
                                                    self.alt_id_col)
        new_row_to_obs, new_row_to_shapes = new_mapping_res
        
        # Get the probability of each observation choosing each available 
        # alternative
        return calc_probabilities(self.coefs.values, 
                                  new_design,
                                  new_alt_IDs, 
                                  new_row_to_obs,
                                  new_row_to_shapes,
                                  self.utility_transform,
                                  intercept_params=self.intercepts,
                                  shape_params = self.shapes,
                                  chosen_row_to_obs=None,
                                  return_long_probs=True)
    
    
    def to_pickle(self, filepath):
        """
        filepath:   str. Should end in .pkl. If it does not, ".pkl" will be
                    appended to the passed string.
        ====================
        Returns:    None. Saves the model object to the passed in filepath.
        """
        assert isinstance(filepath, str)        
        if filepath[-4:] != ".pkl":
            filepath = filepath + ".pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print("Model saved to {}".format(filepath))
        
        return None
      