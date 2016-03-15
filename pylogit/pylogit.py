# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 22:07:30 2016

@module:    pylogit
@name:      Python Based Conditional Logit-type Models
@author:    Timothy Brathwaite
@summary:   Contains functions necessary for estimating multinomial, 
            conditional choice models. This includes standard conditional logit
            models as well as asymmetric choice models.
@notes:     "Under the hood", this module indirectly or directly relies upon
            the following files:
            [base_multinomial_cm_v2.py,
             choice_calcs.py,
             choice_tools.py,
             conditional_logit.py,
             multinomial_asym_logit_3.py,
             multinomial_uneven_logit_2.py,
             multinomial_scobit_2.py,
             multinomial_clog_log_2.py]
"""
import conditional_logit as mnl
#import multinomial_asym_logit_3 as asym_logit
#import multinomial_uneven_logit_2 as uneven_logit
#import multinomial_scobit_2 as scobit
#import multinomial_clog_log_2 as clog_log

from choice_tools import convert_wide_to_long
from choice_tools import convert_long_to_wide

# Create a dictionary relating the model type parameter to the class that
# the general choice model should inherit from
model_type_to_class = {"MNL": mnl.MNL,
    #                  "Asym": asym_logit.MNAL,
    #                  "Cloglog": clog_log.MNCL,
    #                  "Scobit": scobit.MNSL,
    #                  "Uneven": uneven_logit.MNUL
                       }

# Create a dictionary relating the model type parameter to the name of the
# class that the general choice model should inherit from                       
model_type_to_class_name = {"MNL": "MNL",
    #                        "Asym": "MNAL",
    #                        "Cloglog": "MNCL",
    #                        "Scobit": "MNSL",
    #                        "Uneven": "MNUL"
                            }
                            

def create_choice_model(data, alt_id_col, 
                        obs_id_col, choice_col,
                        specification, model_type,
                        intercept_ref_pos=None,
                        shape_ref_pos=None,
                        names=None,
                        intercept_names=None,
                        shape_names=None):
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
                            
        model_type:         string. Denotes the model type of the choice_model
                            being instantiated. Should be one of the following
                            values:
                            ["MNL", "Asym", "Cloglog", "Scobit", "Uneven"].
                            At the moment, only MNL is released for public use.
                        
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
        ====================
        Returns:        instantiation of the Choice Model Class corresponding 
                        to the model type passed as the function argument. The 
                        returned class will have been instantiated with the
                        arguments passed to this function.
        """
        # Carry out the appropriate instantiation process for the chosen
        # choice model
        return model_type_to_class[model_type](data,
                                               alt_id_col,
                                               obs_id_col,
                                               choice_col,
                                               specification,
                                         intercept_ref_pos=intercept_ref_pos,
                                         shape_ref_pos=shape_ref_pos,
                                               names=names,
                                               intercept_names=intercept_names,
                                               shape_names=shape_names)
        