#!/bin/bash

# add likely path to py-pde package
export PYTHONPATH=../py-pde:$PYTHONPATH

if [ ! -z $1 ] 
then 
    # test pattern was specified 
    echo 'Run unittests with pattern '$1':'
    ./run_tests.py --unit --parallel --pattern "$1"
else
    # test pattern was not specified
    echo 'Run all unittests:'
    ./run_tests.py --unit --parallel
fi
