What PyLogit is
============

PyLogit is a Python package for performing maximum likelihood estimation of conditional logit models and similar logit-like models.

Main Features
===========

* Conditional Logit Models
* Supports datasets where the choice set differs across observations
* Supports model specifications where the coefficients for a given variable may be
   - completely alternative-specific (i.e. one coefficient per alternative, subject to identification of the coefficients),
   - subset-specific (i.e. one coefficient per subset of alternatives, where each alternative belongs to only one subset, and there are more than 1 but less than J subsets, where J is the maximum number of available alternatives in the dataset),
    - completely generic (i.e. one coefficient across all alternatives). 

Where to get it
===============

Available from PyPi
    http://pypi.python.org/pypi/pylogit/

License
=======

Modified BSD (3-clause)
 
 