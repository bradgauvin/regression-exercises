#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from scipy import stats 
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
import math

def plot_residuals(x, y, yhat):
    """ Plots residuals vs y """
    residuals = yhat - y
    
    baseline = np.full(len(y),y.mean())
    
    baseline_residuals = baseline - y
    
    plt.subplots(2,2, figsize=(16,5))
    plt.subplot(221)
    plt.scatter(x=x, y = residuals)
    plt.axhline(0)
    plt.xlabel('x')
    plt.ylabel('residual (yhat - y)')
    plt.title("OLS Residuals")
    plt.subplot(222)
    plt.scatter(x=x, y = baseline_residuals)
    plt.axhline(0)
    plt.xlabel('x')
    plt.ylabel('residual (yhat - y)')
    plt.title("Baseline residuals")

    plt.subplot(223)
    plt.scatter(x=y, y = residuals)
    plt.axhline(0)
    plt.xlabel('y')
    plt.ylabel('residual (yhat - y)')
    plt.title("OLS Residuals")
    plt.subplot(224)
    plt.scatter(x=y, y = baseline_residuals)
    plt.axhline(0)
    plt.xlabel('y')
    plt.ylabel('residual (yhat - y)')
    plt.title("Baseline residuals")

    plt.tight_layout()

# Definitions for each element    
def regression_errors(y, yhat):
    """ Return error metrics for given y and yhat """
    residuals = yhat - y
    
    SSE = sum(residuals**2)

    ESS = sum((yhat - y.mean())**2)

    TSS = SSE + ESS

    MSE = mean_squared_error(y, yhat)

    RMSE = mean_squared_error(y, yhat, squared = False)
    
    R2 = r2_score(y, yhat)

    return {'SSE':SSE,'ESS':ESS,'TSS':TSS,'MSE':MSE, 'RMSE':RMSE, 'R2': R2}

# Baseline
def baseline_mean_errors(y):
    """ Compute the SSE, MSE, and RMSE for the baseline model (mean) """
    
    baseline = np.full(len(y),y.mean())
    
    residuals = baseline - y
    
    SSE = sum(residuals**2)
    
    MSE = mean_squared_error(y, baseline)
    
    RMSE = mean_squared_error(y, baseline, squared = False)
    
    return {'SSE':SSE, 'MSE':MSE, 'RMSE':RMSE}

# Compare Regression to baselin
def better_than_baseline(y, yhat):
    """ Returns True if the model performs better than baseline based on RMSE """
    
    return regression_errors(y,yhat)['RMSE'] < baseline_mean_errors(y)['RMSE']

