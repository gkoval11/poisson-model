# Data Analysis Marine Science - Final Project Draft

## Analysis of Species Abundance Data with Environmental Variables
#### By *Gammon Koval*

Repository with Jupyter notebook, custom python package, and data to run analysis of species abundance data gathered in the surf zone and determine relationship with several environmental factors collected *in-situ* using negative binomial regression.

### Data Source

Data was collected under a project funded by the Ocean Protection Council and California SeaGrant as part of the 2022 statewide marine protected area assessment. Data was collected with appropriate permits.

### Contents

1. [surf_zone_final.ipynb](surf_zone_final.ipynb) - Jupyter notebook with code for analysis
2. [surfzone.py](surfzone.py) - Python script with custom function for running negative binomial regression and zero inflated poisson model
3. [Data/Formatted directory](Data/Formatted) - folder containing formatted data
4. [Results directory](Results) - folder to store data output for future analysis
5. [figure directory](figure) - folder to store figures created in the surf_zon_final notebook
6. [surf_zone-data_formatting](surf_zone-data_formatting.ipynb) - Jupyter notebook with code to tidy and format the data used in the analysis

### Requirements

In addition to Python, Anaconda, and standard packages (pandas, numpy, matplotlib, seaborn), you will need the following packages:

* statsmodels.api
* statsmodels.formula.api
* dmatrices from the patsy package

### How to use

Open [surf_zone-data_formatting](surf_zone-data_formatting.ipynb) and run the notebook to create the formatted data used in the [surf_zone_final](surf_zone_final.ipynb). A brief introduction, methods, question answered, results, conclusion, next steps, and references are included in the notebook to orient you to the project.
