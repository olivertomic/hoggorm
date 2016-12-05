# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='hoggorm',
    version='0.1.0',

    description='Package for explorative multivariate statistics',
    long_description=readme,

    url='https://github.com/olivertomic/hoggorm',

    # Author details
    author='Oliver Tomic',
    author_email='olivertomic@zoho.com',

    # Maintainer details
    maintainer='Thomas Graff',
    maintainer_email='graff.thomas@gmail.com',

    license=license,

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
