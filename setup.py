from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'py-droplets',
  package_data={"droplets": ["py.typed"]},
  packages = find_packages(),
  zip_safe=False,  # this is required for mypy to find the py.typed file
  version = '0.3',
  license='MIT',
  description = 'Python package for describing and analyzing droplets in experiments and simulations',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'David Zwicker',
  author_email = 'david.zwicker@ds.mpg.de',
  url = 'https://github.com/zwicker-group/py-droplets',
  download_url = 'https://github.com/zwicker-group/py-droplets/archive/v0.3.tar.gz',
  keywords = ['emulsions', 'image-analysis'],
  python_requires='>=3.6',
  install_requires=['matplotlib',
                    'numpy',
                    'numba',
                    'scipy',
                    'sympy',
                    'py-pde'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)