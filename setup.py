# setup.py
from setuptools import setup, find_packages

setup(
    name="acoustic_fem",
    version="1.0.0",
    description="A Python-based Acoustic FEM Solver for Metamaterials",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib",
        "tqdm"
    ],
    python_requires=">=3.8",
)