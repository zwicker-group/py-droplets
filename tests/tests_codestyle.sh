#!/usr/bin/env bash
#
# This script checks the code format of this package without changing files
#

echo "Checking codestyle in import statements..."
isort --diff ..

# format all code
for dir in droplets examples ; do
    echo "Checking codestyle in folder ${dir}..."
    black -t py36 --check ../${dir}
done