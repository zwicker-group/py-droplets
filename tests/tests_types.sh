#!/usr/bin/env bash
#
# This script checks the code format of this package without changing files
#

export MYPYPATH=../../py-pde:$MYPYPATH

./run_tests.py --style
