"""
This file declares the strings that will be displayed for each model type based
on the abbriviated model type string that is passed to the choice model
constructor.
"""
from collections import OrderedDict
model_type_to_display_name = OrderedDict()
model_type_to_display_name["MNL"] = "Multinomial Logit Model"
model_type_to_display_name["Asym"] = "Multinomial Asymmetric Logit Model"
model_type_to_display_name["Cloglog"] = "Multinomial Clog-log Model"
model_type_to_display_name["Scobit"] = "Multinomial Scobit Model"
model_type_to_display_name["Uneven"] = "Multinomial Uneven Logit Model"
model_type_to_display_name["Nested Logit"] = "Nested Logit Model"
model_type_to_display_name["Mixed Logit"] = "Mixed Logit Model"
