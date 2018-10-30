---
title: "hoggorm: a python library for explorative multivariate statistics"
tags:
  - multivariate statistics
  - explorative multivariate analysis
  - chemometrics
  - partial least squares regression
  - principal component regression
  - principal component analysis
authors:
 - name: Oliver Tomic
   orcid: 0000-0003-1595-9962
   affiliation: 1
 - name: Thomas Graff
   affiliation: 2
 - name: Kristian Hovde Liland
   orcid: 0000-0001-6468-9423
   affiliation: 1
 - name: Tormod Næs
   affiliation: 3
affiliations:
- name: Norwegian University of Life Sciences, Ås, Norway
  index: 1
- name: TGXnet, Norway
  index: 2
- name: Nofima, Norway
  index: 3
date: 16 August 2018
bibliography: paper.bib
---

# Summary
hoggorm is a python library for explorative analysis of multivariate data that implements statistical methods typically used in the field of chemometrics [@naes88]. Although hoggorm shares some statistical methods with the Python library scikit-learn for machine learning, hoggorm follows the chemometrics paradigm for data analysis where great attention is paid to understanding and interpretation of the variance in the data. For this purpose, hoggorm provides access to typical interpretation tools, such as scores, loadings, correlation loadings, explained variances for calibrated and validated models (both for individual variables as well as all variables together). Note that models trained with hoggorm may also be applied for prediction purposes, both for continuous and categorical variables, where appropriate. Currently (version 0.12.0), statistical methods implemented in hoggorm are: (I) principal component analysis (PCA) for analysis of single data arrays or matrices [@mardia79]; (II) principal component regression (PCR) [@martens88] and (III) partial least squares regression (PLSR) [@wold82] for analysis of two data arrays. PLSR is provided in two versions; (a) PLS1 for multivariate independent data and a single response variable; (b) PLS2 for situations where the independent data and response data are both multivariate. Furthermore, hoggorm implements the matrix correlation coefficient methods RV [@robert76], RV2 (also known as modified RV) [@smilde09] as well as the similarity index for comparing coupled matrices index (SMI) [@indahl18].

# Acknowledgements
Both users and developers have made valuable contributions to improve the usability the hoggorm library. This includes reporting of bugs, testing various features and other forms of feedback. A complete list of contributors is provided at https://github.com/olivertomic/hoggorm/graphs/contributors

# References
