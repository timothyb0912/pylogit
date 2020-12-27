=========
Changelog
=========

The format is based on [Keep-a-Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

.. towncrier release notes start

Pylogit 0.2.2 (2017-12-11)
==========================

Bug fixes
---------

- Changed tqdm dependency to allow for anaconda compatibility.


Pylogit 0.2.1 (2017-12-11)
==========================

Bug fixes
---------

- Added statsmodels and tqdm as package dependencies to fix errors with 0.2.0.


Pylogit 0.2.0 (2017-12-10)
==========================

Added new features
------------------

- Added support for Python 3.4 - 3.6
- Added AIC and BIC to summary tables of all models.
- Added support for bootstrapping and calculation of bootstrap confidence intervals:

  - percentile intervals,
  - bias-corrected and accelerated (BCa) bootstrap confidence intervals, and
  - approximate bootstrap confidence (ABC) intervals.

- Changed sparse matrix creation to enable estimation of larger datasets.


Trivial/Internal Changes
------------------------

- Refactored internal code organization and classes for estimation.


Pylogit 0.1.2 (2016-12-04)
==========================

Added new features
------------------

- Added support to all logit-type models for parameter constraints during model estimation.
  All models now support the use of the constrained_pos keyword argument.
- Added new argument checks to provide user-friendly error messages.
- Created more than 175 tests, bringing statement coverage to 99%.
- Updated the underflow and overflow protections to make use of L’Hopital’s rule where appropriate.


Bug fixes
---------

- Fixed bugs with the nested logit model.
  In particular, the predict function, the BHHH approximation to the Fisher Information Matrix, and the ridge regression penalty in the log-likelihood, gradient, and hessian functions have been fixed.


Improved Documentation
----------------------

- Added new example notebooks demonstrating prediction, mixed logit, and converting long-format datasets to wide-format.
- Edited docstrings for clarity throughout the library.


Trivial/Internal Changes
------------------------

- Extensively refactored codebase.


Pylogit 0.1.1 (2016-08-30)
==========================

Improved Documentation
----------------------
- Added python notebook examples demonstrating how to estimate the asymmetric choice models and the nested logit model.
- Corrected the docstrings in various places.
- Added new datasets to the github repo.


Pylogit 0.1.0 (2016-08-29)
==========================

Added new features
------------------

- Added asymmetric choice models.
- Added nested logit and mixed logit models.
- Added tests for mixed logit models.
- Added an example notebook demonstrating how to estimate the mixed logit model.


 Improved Documentation
 ----------------------

 - Changed documentation to numpy doctoring standard.


Trivial/Internal Changes
------------------------

 - Made print statements compatible with python3.
 - Fixed typos in library documentation.
 - Internal refactoring.


Pylogit 0.0.0 (2016-03-15)
==========================

Added new features
------------------

- Initial package release with support for the conditional logit (MNL) model.
