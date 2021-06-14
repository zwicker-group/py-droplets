from pathlib import Path

from setuptools import find_packages, setup

BASE_PATH = Path(__file__).resolve().parent


# read the version from the particular file
with open(BASE_PATH / "droplets" / "version.py", "r") as f:
    exec(f.read())

DOWNLOAD_URL = (
    f"https://github.com/zwicker-group/py-droplets/archive/v{__version__}.tar.gz"
)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="py-droplets",
    package_data={"droplets": ["py.typed"]},
    packages=find_packages(),
    zip_safe=False,  # this is required for mypy to find the py.typed file
    version=__version__,
    license="MIT",
    description=(
        "Python package for describing and analyzing droplets in experiments and "
        "simulations"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="David Zwicker",
    author_email="david.zwicker@ds.mpg.de",
    url="https://github.com/zwicker-group/py-droplets",
    download_url=DOWNLOAD_URL,
    keywords=["emulsions", "image-analysis"],
    python_requires=">=3.7",
    install_requires=["matplotlib", "numpy", "numba", "scipy", "sympy", "py-pde"],
    extras_require={
        "hdf": ["h5py>=2"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
