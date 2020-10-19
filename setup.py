from os import path

from setuptools import setup, find_packages

__version__ = "0.3.1"

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="route-powertrain",
    version=__version__,
    description=
    "RouteE is a tool for predicting energy usage over a set of road links.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.nrel.gov/MBAP/routee",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering"
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "pandas==0.24",
        "numpy==1.16",
        "scikit-learn==0.21",
        "scipy==1.2",
    ],
    extras_require={
        "optional": [
            "matplotlib",
        ],
    },
    author="National Renewable Energy Laboratory",
    author_email="Holden, Jacob <Jacob.Holden@nrel.gov>",
    license="Copyright ©2020 Alliance for Sustainable Energy, LLC All Rights Reserved",
)