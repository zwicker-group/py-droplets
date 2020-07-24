# py-droplets

[![Build Status](https://travis-ci.org/zwicker-group/py-droplets.svg?branch=master)](https://travis-ci.org/zwicker-group/py-droplets)
[![Documentation Status](https://readthedocs.org/projects/py-droplets/badge/?version=latest)](https://py-droplets.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/py-droplets.svg)](https://badge.fury.io/py/py-droplets)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/zwicker-group/py-droplets/branch/master/graph/badge.svg)](https://codecov.io/gh/zwicker-group/py-droplets)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/zwicker-group/py-droplets.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/zwicker-group/py-droplets/context:python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`py-droplets` provides python code for representing physical droplets using
key parameters like position, size, or shape.
These droplets can also be represented as collections (emulsions) over time.
Moreover, the package provides methods for locating droplets in microscope
images or phase field data from simulations.


Installation
------------

`py-droplets` is available on `pypi`, so you should be able to install it
through `pip`:

```bash
pip install py-droplets
```

In order to have all features of the package available, you might also want to 
install the following optional packages:

```bash
pip install h5py tqdm
```


Usage
-----

More information
----------------
* Tutorial notebook in the [tutorials folder](https://github.com/zwicker-group/py-droplets/tree/master/examples/tutorial)
* Examples in the [examples folder](https://github.com/zwicker-group/py-droplets/tree/master/examples)
* [Full documentation on readthedocs](https://py-droplets.readthedocs.io/)
  or as [a single PDF file](https://py-droplets.readthedocs.io/_/downloads/en/latest/pdf/).
