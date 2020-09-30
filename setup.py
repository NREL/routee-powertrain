from os import path

from setuptools import setup, find_packages

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="route-powertrain",
    version="0.3.1",
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
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering"
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "xgboost",
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