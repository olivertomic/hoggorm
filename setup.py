# -*- coding: utf-8 -*-
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import re
from os import path
# To use a consistent encoding
from codecs import open
from setuptools import setup, find_packages


def get_version():
    VERSIONFILE = path.join('hoggorm', 'version.py')
    initfile_lines = open(VERSIONFILE, 'rt', encoding='utf-8').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))


with open('README.rst', encoding='utf-8') as f:
    readme = f.read()


setup(
    name='hoggorm',
    version=get_version(),

    description='Package for explorative multivariate statistics',
    long_description=readme,

    url='https://github.com/olivertomic/hoggorm',

    # Author details
    author='Oliver Tomic',
    author_email='olivertomic@zoho.com',

    # Maintainer details
    #maintainer='Thomas Graff',
    #maintainer_email='graff.thomas@gmail.com',

    license='BSD License',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Natural Language :: English',
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    # What does your project relate to?
    keywords='statistic education science',

    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'Numpy>=1.10',
    ],
)
