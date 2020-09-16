Installation
############

This `py-droplets` package is developed for python 3.6+ and should run on all
common platforms.
The code is tested under Linux, Windows, and macOS.

Since the package is available on `pypi <https://pypi.org/project/py-droplets/>`_,
the installation is in principle as simple as running

.. code-block:: bash

    pip install py-droplets
    

In order to have all features of the package available, you might also want to 
install the following optional packages:

.. code-block:: bash

	pip install h5py pyfftw tqdm


Installing from source
^^^^^^^^^^^^^^^^^^^^^^
Installing from source can be necessary if the pypi installation does not work
or if the latest source code should be installed from github.


Prerequisites
-------------

The code builds on other python packages, which need to be installed for
`py-droplets` to function properly.
The required packages are listed in the table below:

===========  ========= =========
Package      Version   Usage 
===========  ========= =========
matplotlib   >= 3.1.0  Visualizing results
numpy        >=1.16    Array library used for storing data
numba        >=0.43    Just-in-time compilation to accelerate numerics
scipy        >=1.2     Miscellaneous scientific functions
sympy        >=1.4     Dealing with user-defined mathematical expressions
py-pde       >=0.4     Simulating partial differential equations
===========  ========= =========

These package can be installed via your operating system's package manager, e.g.
using :command:`macports`, :command:`homebrew`, :command:`conda`, or
:command:`pip`.
The package versions given above are minimal requirements, although
this is not tested systematically. Generally, it should help to install the
latest version of the package.
The `py-pde` package is available on `pip`, but if this is inconvenient the
package can also be installed from github sources, as `described in its 
documentation 
<https://py-pde.readthedocs.io/en/latest/installation.html#installing-from-source>`_.  

A small subset of the package will only be available if extra optional packages are
installed. Currently, this only concerns the `h5py` package for reading hdf files.


Downloading the package
-----------------------

The package can be simply checked out from
`github.com/zwicker-group/py-droplets <https://github.com/zwicker-group/py-droplets>`_.
To import the package from any python session, it might be convenient to include
the root folder of the package into the :envvar:`PYTHONPATH` environment variable.

This documentation can be built by calling the :command:`make html` in the
:file:`docs` folder.
The final documentation will be available in :file:`docs/build/html`.
Note that a LaTeX documentation can be build using :command:`make latexpdf`.

