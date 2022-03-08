# Zero-Inflated Poisson Models on Surf Zone Assemblage

## Analysis of Species Abundance Data with Environmental Variables
#### By *Gammon Koval*

Repository with a Jupyter notebook and custom python package to run analysis of species abundance data gathered in the surf zone and determine relationship with several environmental factors collected *in-situ* using zero-inflated Poisson regression.

### Data Source

Data was collected under a project funded by the Ocean Protection Council and California SeaGrant as part of the 2022 statewide marine protected area assessment. Data was collected with appropriate permits. The data is still being actively analyzed, so it is not publicly available yet. If you're interesting in seeing or using the data, please reach out to me at [gkoval@csumb.edu](mailto:gkoval@csumb.edu) to discuss your use case.

### Contents

1. [surf_zone_final.ipynb](surf_zone_final.ipynb) - Jupyter notebook with code for analysis
2. [surfzone.py](surfzone.py) - Python script with custom function for running negative binomial regression and zero inflated poisson model

### Requirements

In addition to Python, Anaconda, and standard packages (pandas, numpy, matplotlib, seaborn), you will need the following packages:

* statsmodels.api
* statsmodels.formula.api
* dmatrices from the patsy package

### How to use

Open [surf_zone_final](surf_zone_final.ipynb) which includes a brief introduction, methods, question answered, results, conclusion, next steps, and references are included in the notebook to orient you to the project.
