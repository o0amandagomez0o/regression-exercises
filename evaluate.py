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
    
    
    
    
MSE = mean_squared_error(tips.tip, tips.yhat)
SSE = MSE * len(tips)
RMSE = sqrt(MSE)
TSS = (mean_squared_error(tips.tip, tips.baseline)) * len ()
ESS = TSS - SSE


MSE_baseline = mean_squared_error(tips.tip, tips.baseline)
SSE_baseline = MSE_baseline * len(tips)
RMSE_baseline = sqrt(MSE_baseline)





R2 = 1 - (SSE/TSS)