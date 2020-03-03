# build the distribution
python3 setup.py sdist bdist_wheel

# upload the package to pypi
# python3 -m twine upload dist/*

# clean data 
rm -rf build dist py_droplets.egg-info
