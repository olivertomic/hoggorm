# Contributing to hoggorm

If you are interested in contributing to hoggorm, your contributions will fall into two categories:

1. You want to implement a new feature:
....
2. You want to fix a bug:
    - Feel free to send a Pull Request any time you encounter a bug. Please provide a clear and concise description of what the bug was. If you are unsure about if this is a bug at all or how to fix, post about it in an issue.

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/olivertomic/hoggorm

## Developing hoggorm

To develop hoggorm on your machine, here are some tips:

1. Create an enviroment for hoggorm

```
conda create -n fork5 python=3.6
```

2. Clone a copy of hoggorm from source:

```
cd tmp5
git clone https://github.com/andife/hoggorm.git -b hogCI
```

4. Install hoggorm in `dev` and `test` mode:

```
cd hoggorm
pip install -e .[dev,test]
```

5. Ensure that you have a working hoggorm installation by running the entire test suite with

```
python setup.py test
```


## Unit Testing

hoggorm's testing is located under `test/`.
Run the entire test suite with

```
python setup.py test
```


## Continuous Integration

hoggorm uses [GitHub Action](https://github.com/andife/hoggorm/actions?query=workflow%3Aci-build) in combination with [CodeCov](https://codecov.io/github/andife/hoggorm?branch=hogCI) for continuous integration.

Everytime you send a Pull Request, your commit will be built and checked against the hoggorm guidelines:

1. Ensure that your code is formatted correctly by testing against the styleguides of [`flake8`](https://github.com/PyCQA/flake8) and [`pycodestyle`](https://github.com/PyCQA/pycodestyle):

```
pycodestyle hoggorm test examples
flake8 hoggorm test examples
```

If you do not want to format your code manually, we recommend to use [`yapf`](https://github.com/google/yapf).


```
python setup.py test
```
