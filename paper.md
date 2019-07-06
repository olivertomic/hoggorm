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
- name: Nofima, Ås, Norway
  index: 3
date: 16 August 2018
bibliography: paper.bib
---

# Summary
hoggorm is a python library for explorative analysis of multivariate data that implements statistical methods typically used in the field of chemometrics [@naes88]. Although hoggorm shares some statistical methods with the Python library scikit-learn for machine learning, it follows the chemometrics paradigm for data analysis where great attention is paid to understanding and interpretation of the variance in the data. 

Currently (version 0.13.0), statistical methods implemented in hoggorm are: (I) principal component analysis (PCA) for analysis of single data arrays or matrices [@mardia79]; (II) principal component regression (PCR) [@naes88] and (III) partial least squares regression (PLSR) [@wold83] for analysis of two data arrays. PLSR is provided in two versions; (a) PLS1 for multivariate independent data and a single response variable; (b) PLS2 for situations where the independent data and response data are both multivariate. PCA is an unsupervised method which compresses data into low dimensional representations that capture the dominant variation in the data. PCR uses the compressed features as a basis for regression, while PLSR uses supervised compression to capture the dominant co-varation between the data matrix and the target/response. Both PLS1, PLS2 and PCR posess a couple of useful properties: they easily handle situations where: (a) the multivariate independent data are short and wide, that is, data with few objects (instances) and many variables (features); (b) the multivariate independent data contain many highly correlated variables, thus providing stable models despite high correlations. 

The hoggorm package provides access to an extended repertiore of interpretation tools that are integrated in PCA, PCR, PLS1 and PLS2. These including scores, loadings, correlation loadings, explained variances for calibrated and validated models (both for individual variables as well as all variables together). Scores are the objects' coordinates in the compressed data representation and can for instance be used to search for patterns or groups among the objects. Loadings are the variables' representations in the compressed space showing their contribution to the components. Finally, correlation loadings show how each variable correlates to the score vectors/components and how much of the variation in each variable is explained across components. Note that models trained with hoggorm may also be applied for prediction purposes, both for continuous and categorical variables, where appropriate.

Furthermore, hoggorm implements the matrix correlation coefficient methods RV [@robert76] and RV2 (also known as modified RV) [@smilde09], as well as the similarity index for comparing coupled matrices index (SMI) [@indahl18]. These methods can be used to quickly determine how much common information there is between two data matrices. Results from models trained with hoggorm may be visualised using the complementary plotting package hoggormplot

# Acknowledgements
Both users and developers have made valuable contributions to improve the usability the hoggorm library. This includes reporting of bugs, testing various features and other forms of feedback. A complete list of contributors is provided at https://github.com/olivertomic/hoggorm/graphs/contributors

# References
