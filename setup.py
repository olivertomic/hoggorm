#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

__version__ = '0.13.3'

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Oliver Tomic",
    author_email='oliver.tomic@nmbu.no',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Development Status :: 4 - Beta'
        'Natural Language :: English'
        'Programming Language :: Python :: 2.7'
        'Programming Language :: Python :: 3'
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Developers'
        'Intended Audience :: Education'
        'Intended Audience :: Science/Research'
        'Operating System :: OS Independent'
        'License :: OSI Approved :: BSD License'
        'Topic :: Scientific/Engineering'
        'Topic :: Scientific/Engineering :: Bio-Informatics'
        'Topic :: Scientific/Engineering :: Chemistry'
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    description="Package for explorative multivariate statistics.",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords=['hoggorm',
              'statistic',
            'education',
	'science'],
    name='hoggorm',
    packages=find_packages(include=['hoggorm', 'hoggorm.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=['pytest', 'pytest-cov','numpy'],
    url='https://github.com/olivertomic/hoggorm',
    version=__version__,
    zip_safe=False,
)
