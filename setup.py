#!/usr/bin/env python
"""The setup script.

How to make a new release:

Update version information
> someeditor hoggorm/version.py

Run tests
> pytest

Generate package files
> python setup.py sdist
> python setup.py bdist_wheel

Upload package to PyPI
> twine upload dist/*

Tag the new release
> git tag -a vX.Y.Z -m "Tag release X.Y.Z"
> git push origin --tags
"""


from setuptools import setup


setup()
