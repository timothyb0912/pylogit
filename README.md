![PyLogit Logo](./images/PyLogit_Final-small-04.png)

![Tests](https://github.com/timothyb0912/pylogit/workflows/Testing/badge.svg)

# PyLogit
PyLogit is a Python package for performing maximum likelihood estimation of conditional logit models and similar discrete choice models.

## Main Features
- It supports
   - Conditional Logit (Type) Models
     - Multinomial Logit Models
     - Multinomial Asymmetric Models
        - Multinomial Clog-log Model
        - Multinomial Scobit Model
        - Multinomial Uneven Logit Model
        - Multinomial Asymmetric Logit Model
   - Nested Logit Models
   - Mixed Logit Models (with Normal mixing distributions)
- It supports datasets where the choice set differs across observations
- It supports model specifications where the coefficient for a given variable may be
   - completely alternative-specific   
   (i.e. one coefficient per alternative, subject to identification of the coefficients),
   - subset-specific  
   (i.e. one coefficient per subset of alternatives, where each alternative belongs to only one subset, and there are more than 1 but less than J subsets, where J is the maximum number of available alternatives in the dataset),
   - completely generic  
   (i.e. one coefficient across all alternatives).

## Installation
Available from [PyPi](https://pypi.python.org/pypi/pylogit):
```
pip install pylogit
```

Available through [Anaconda](https://anaconda.org/conda-forge/pylogit):
```
conda install -c conda-forge pylogit
```

or

```
conda install -c timothyb0912 pylogit
```

## Usage
For Jupyter notebooks filled with examples, see [examples](./examples/).


## For More Information
For more information about the asymmetric models that can be estimated with PyLogit, see the following paper

> Brathwaite, T., & Walker, J. L. (2018). Asymmetric, closed-form, finite-parameter models of multinomial choice. Journal of Choice Modelling, 29, 78–112. https://doi.org/10.1016/j.jocm.2018.01.002

A free and better formatted version is available at [ArXiv](http://arxiv.org/abs/1606.05900).

## Attribution
If PyLogit (or its constituent models) is useful in your research or work, please cite this package by citing the paper above.

## License
Modified BSD (3-clause). See [here](./LICENSE.txt).

## Changelog
See [here](./CHANGELOG.rst).
