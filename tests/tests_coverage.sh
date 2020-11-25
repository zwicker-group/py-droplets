#!/bin/bash

# add likely path to py-pde package
export PYTHONPATH=../py-pde:$PYTHONPATH

echo 'Determine coverage of all unittests...'

./run_tests.py --unit --coverage --no_numba --parallel