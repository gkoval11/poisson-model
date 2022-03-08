#import necessary packages for function
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import numpy as np



#functio nfor variance 
def variance(data):
    '''Function used to calculate the variance of a dataset
    
    Input: 
    data = the data to calculate the variance of'''
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
     # Variance
    variance = sum(deviations) / n
    return variance

#function for negative binomial regression
def nb2(df, species):
    '''Function to model negative binomial regression for species with set environmental parameters
    
    Input:
    df = dataframe with data
    species = maxN of fish species to run model on'''
    
    #filter data to fish species and remove missing values
    df = df[df['Sci_Name'] == species]
    df = df.dropna()
    
    #create expression for model
    expr = """MaxN ~ Tide_height_ft + Wind_Average_mph + Salinity_PPT + Depth_ft + Visibility_ft + 
    Temp_C + Breaker_Height_m + Breaker_Period_s"""
    
    #Using glm from statsmodel.formula, run Poisson regression on data
    poisson_training_results = smf.glm(formula=expr, data=df, family=sm.families.Poisson()).fit()
    
    #create new column for lambda values
    df['MaxN_LAMBDA'] = poisson_training_results.mu
    
    #derive values of dependent variable for ordinary least squared regression
    df['AUX_OLS_DEP'] = df.apply(lambda x: ((x['MaxN'] - x['MaxN_LAMBDA'])**2 - x['MaxN_LAMBDA']) /
                                 x['MaxN_LAMBDA'], axis=1)

    #create an expression for model specification for the OLSR
    ols_expr = """AUX_OLS_DEP ~ MaxN_LAMBDA - 1"""

    #Configure and fit the ordinary least squared regression model
    aux_olsr_results = smf.ols(ols_expr, df).fit()
        
    #Using GLM from statsmodel.formula and results from OLSR , run the negative binomial regression on data
    nb2_training_results = smf.glm(formula=expr, data=df,
                                   family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()
    
    #return negative binomial regression model summary
    return nb2_training_results.summary()

#create function for zero-inflated poisson model
def zero_poisson(df, species):
    '''Function to run a zero-inflated poisson model
    
    Inputs:
    df = data table with the data for the model
    species = variable with species name to run the model on'''
    
    #filter data to fish species and remove missing values
    df = df[df['Sci_Name'] == species]
    df = df.dropna()
    
    #create training and testing data
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]
    
    #create expression for model
    expr = """MaxN ~ Tide_height_ft + Wind_Average_mph + Depth_ft + Visibility_ft + 
    Temp_C + Breaker_Height_m + Breaker_Period_s"""
    
    #create training and testing data for dependent and independent variables
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')
    
    #run the zero inflated poisson model on the training data using x data as inflation
    zip_training_results = sm.ZeroInflatedPoisson(endog = y_train, exog = X_train, 
                                                  exog_infl = X_train, inflation = 'logit').fit()
    
    #if the model doesn't work with the x data for the inflation 
    if str(zip_training_results.pvalues[0]) == 'nan':
        #create a matrix of ones and run again
        zip_training_results = sm.ZeroInflatedPoisson(endog=y_train, exog=X_train, 
                                                      exog_infl = np.ones((len(X_train),1)), 
                                                      inflation ='logit').fit() 
        #use the model results to precit the values
        zip_predictions = zip_training_results.predict(X_test, exog_infl = np.ones((len(X_test),1)))
    else:
        #run the preditions on the actual data
        zip_predictions = zip_training_results.predict(X_test, exog_infl = X_test)

    #create a variable for the predicted and actual counts
    predicted_counts=np.round(zip_predictions)
    actual_counts = y_test['MaxN']

    #calculate the root mean, squared error
    zip_rmse = np.sqrt(np.sum(np.power(np.subtract(predicted_counts,actual_counts),2)))
    
    return zip_training_results.summary(), zip_rmse
    
