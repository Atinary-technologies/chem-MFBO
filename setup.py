# !/usr/bin/env python

from setuptools import find_packages, setup

__version__ = "0.0.0-beta"

__author__ = "Atinary Technologies Inc. & Atinary Technologies Sarl"

setup(
    name="chem_mfbo",
    author=__author__,
    version=__version__,
    author_email="vsabanzagil@atinary.com",
    description="Multi-fidelity BO for molecule and materials problems",
    # implement entry point here
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy==1.26.4",
        "pandas==1.5.3",
        "pyyaml",
        "tqdm",
        "botorch==0.10.0",
        "torch==2.2.1",
        "jaxtyping==0.2.28",
        "pydantic==2.6.4",
        "hydra-core",
        "seaborn",
        "scipy==1.12.0",
        "scikit-learn==1.2.0",
        "plotly",
        "rdkit",
        "seaborn"
        # any other ones here
    ],
)
