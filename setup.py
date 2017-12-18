# -*- coding: utf-8 -*-
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

How to generate new PyPI package
Update version information
Run tests
Generate package files
Uplad package files

"""
from setuptools import setup, find_packages


setup(
    packages=find_packages(exclude=('tests', 'docs')),

    install_requires=[
        'numpy',
    ],
)
