import codecs
import os

from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(rel_path):
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="routee-powertrain",
    version=get_version(os.path.join("powertrain", "__init__.py")),
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
        "PyYAML",
    ],
    extras_require={
        "optional": [
            "matplotlib",
            "xgboost"
        ],
    },
    include_package_data=True,
    package_data={
        "powertrain.resources.default_models": ["*"],
        "powertrain.resources.sample_routes": ["*"],
    },
    author="National Renewable Energy Laboratory",
    author_email="Holden, Jacob <Jacob.Holden@nrel.gov>",
    license="Copyright ©2020 Alliance for Sustainable Energy, LLC All Rights Reserved",
)
