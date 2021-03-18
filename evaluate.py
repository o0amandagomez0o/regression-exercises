# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from statsmodels.formula.api import ols




def plot_residuals(target, yhat):
    '''
    plot_residuals will take in a target series and prediction series
    and plot the residuals as a scatterplot.
    '''
    
    residual = target - yhat
    
    plt.scatter(target, residual)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel("target")
    plt.ylabel("residual")
    plt.title('Residual Plot')
    plt.show
    
    
    
    
def regression_errors(target, yhat):
    '''
    regression_errors takes in a target and prediction series
    and prints out the regression error metrics.
    '''
    residual = target - yhat
    
    mse = mean_squared_error(target, yhat)
    sse = (residual **2).sum()
    rmse = sqrt(mse)
    tss = ((target - yhat.mean()) ** 2).sum()
    ess = ((yhat - target.mean()) ** 2).sum()
    print(f"""
    MSE: {round(mse,2)}
    SSE: {round(sse,2)}
    RMSE: {round(rmse,2)}
    TSS: {round(tss,2)}
    ESS: {round(ess,2)}
    """)
    
    
    
    
    
def baseline_mean_errors(target):
    '''
    baseline_mean_errors takes in a target 
    and prints out the regression error metrics for the baseline.
    '''
    baseline = target.mean()
    
    residual = target - (baseline)
    
    sse_baseline = (residual **2).sum()
    mse_baseline = sse_baseline * len(target)
    rmse_baseline = sqrt(mse_baseline)
    
    print(f"""
    MSE_baseline: {round(MSE_baseline,2)}
    SSE_baseline: {round(SSE_baseline,2)}
    RMSE_baseline: {round(RMSE_baseline,2)}
    """)
    
    
    
    
def better_than_baseline(target, yhat):
    '''
    better_than_baseline takes in a target and prediction 
    and returns boolean answering if the model is better than the baseline.
    '''
    
    rmse_baseline = sqrt((((target - (target.mean())) **2).sum()) * len(target))
    rmse_model = sqrt((((target - yhat) **2).sum()) * len(target))
    return rmse_model < rmse_baseline




def model_significance(ols_model):
    return {
        'r^2 -- variance explained': ols_model.rsquared,
        'p-value -- P(data|model == baseline)': ols_model.f_pvalue,
    }




def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) **2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()





def reg_error_metrics(target, yhat):
    '''
    reg_error_metrics takes in target and prediction series 
    and returns a dataframe that contains the SSE/MSE/RMSE metrics 
    for. both model and baseline
    and answers if the model is better than the baseline.
    '''
    
    df = pd.DataFrame(np.array(['SSE', 'MSE','RMSE']), columns=['metric'])
    
    df['model_error'] = np.array([sse(target, yhat),  mse(target, yhat), rmse(target, yhat)])
    
    df['baseline_error'] = np.array([sse(target, target.mean()), mse(target, target.mean()), rmse(target, target.mean())])
    
    df['better_than_baseline'] = df.baseline_error > df.model_error
    
    df = df.set_index("metric")
    
    return df
    
    
    
    
    

def select_kbest(X, y, n):
    '''
    select_kbest takes in the 
    predictors (X), 
    the target (y), and 
    the number of features to select (k) and 
    returns the names of the top k selected features based on the SelectKBest class
    '''
    
    # parameters: f_regression stats test
    f_selector = SelectKBest(f_regression, k= n)
    
    # find the top 2 X-feats correlated with y
    f_selector.fit(X_train_scaled, y_train)
    
    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()
    
    # get list of top K features. 
    f_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    
    return f_feature





def rfe(X, y, n):
    '''
    rfe takes in the 
    predictors (X), 
    the target (y), and 
    the number of features to select (k) and 
    returns the names of the top k selected features based on the SelectKBest class
    '''
    
    # initialize the ML algorithm
    lm = LinearRegression()
    
    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n)
    
    # fit the data using RFE
    rfe.fit(X_train_scaled,y_train)  
    
    # get the mask of the columns selected
    feature_mask = rfe.support_
    
    # get list of the column names. 
    rfe_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    
    return rfe_feature