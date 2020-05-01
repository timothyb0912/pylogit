.. raw:: html
    <img src="./images/PyLogit_Final.png" height="50px">

|Build Status| |Coverage|

What PyLogit is
===============
PyLogit is a Python package for performing maximum likelihood estimation of conditional logit models and similar discrete choice models.

Main Features
=============

* Conditional Logit (Type) Models

   - Multinomial Logit Models
   - Multinomial Asymmetric Models

      + Multinomial Clog-log Model
      + Multinomial Scobit Model
      + Multinomial Uneven Logit Model
      + Multinomial Asymmetric Logit Model
   - Nested Logit Models
   - Mixed Logit Models (with Normal mixing distributions)

* Supports datasets where the choice set differs across observations
* Supports model specifications where the coefficient for a given variable may be

   - completely alternative-specific (i.e. one coefficient per alternative, subject to identification of the coefficients),
   - subset-specific (i.e. one coefficient per subset of alternatives, where each alternative belongs to only one subset, and there are more than 1 but less than J subsets, where J is the maximum number of available alternatives in the dataset),
   - completely generic (i.e. one coefficient across all alternatives).

Where to get it
===============
Available from PyPi::
    pip install pylogit

    https://pypi.python.org/pypi/pylogit/0.1.2

Available through Anaconda::
    conda install -c timothyb0912 pylogit

For More Information
====================
For more information about the asymmetric models that can be estimated with PyLogit, see the following paper
    Brathwaite, Timothy, and Joan Walker. "Asymmetric, Closed-Form, Finite-Parameter Models of Multinomial Choice." arXiv preprint arXiv:1606.05900 (2016). http://arxiv.org/abs/1606.05900.

Attribution
===========
If PyLogit (or its constituent models) is useful in your research or work, please cite this package by citing the paper above.

License
=======
Modified BSD (3-clause)

Changelog
=========

0.2.2 (December 11, 2017)
-------------------------
- Changed tqdm dependency to allow for anaconda compatibility.

0.2.1 (December 11, 2017)
-------------------------
- Added statsmodels and tqdm as package dependencies to fix errors with 0.2.0.

0.2.0 (December 10, 2017)
-------------------------
- Added support for Python 3.4 - 3.6

- Added AIC and BIC to summary tables of all models.

- Added support for bootstrapping and calculation of bootstrap confidence intervals:
  - percentile intervals
  - bias-corrected and accelerated (BCa) bootstrap confidence intervals
  - approximate bootstrap confidence (ABC) intervals.

- Changed sparse matrix creation to enable estimation of larger datasets.

- Refactored internal code organization and classes for estimation.

0.1.2 (December 4th, 2016)
--------------------------
- Added support to all logit-type models for parameter constraints during model estimation. All models now support the use of the constrained_pos keyword argument.

- Added new argument checks to provide user-friendly error messages.

- Created more than 175 tests, bringing statement coverage to 99%.

- Added new example notebooks demonstrating prediction, mixed logit, and converting long-format datasets to wide-format.

- Edited docstrings for clarity throughout the library.

- Extensively refactored codebase.

- Updated the underflow and overflow protections to make use of L’Hopital’s rule where appropriate.

- Fixed bugs with the nested logit model. In particular, the predict function, the BHHH approximation to the Fisher Information Matrix, and the ridge regression penalty in the log-likelihood, gradient, and hessian functions have been fixed.

0.1.1 (August 30th, 2016)
-------------------------
- Added python notebook examples demonstrating how to estimate the asymmetric choice models and the nested logit model.

- Corrected the docstrings in various places.

- Added new datasets to the github repo.

0.1.0 (August 29th, 2016)
-------------------------
- Added asymmetric choice models.

- Added nested logit and mixed logit models.

- Added tests for mixed logit models.

- Fixed typos in library documentation.

- Made print statements compatible with python3.

- Changed documentation to numpy doctoring standard.

- Internal refactoring.

- Added an example notebook demonstrating how to estimate the mixed logit model.

0.0.0 (March 15th, 2016)
-------------------------
- Initial package release with support for the conditional logit (MNL) model.

.. |Build status| image:: https://travis-ci.org/timothyb0912/pylogit.svg?branch=master
    :target: https://travis-ci.org/timothyb0912/pylogit
.. |Coverage| image:: https://coveralls.io/repos/github/timothyb0912/pylogit/badge.svg?branch=master
