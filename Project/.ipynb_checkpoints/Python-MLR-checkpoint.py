# -*- coding: utf-8 -*-
"""

@author: Haroon
"""

# Import required python libraries
 
import pandas as pd # data structure and the operations
import numpy as np # arrays and the operations

import sklearn.linear_model as skl_lm # for regression

from sklearn.metrics import mean_squared_error, r2_score # for metrics

import matplotlib.pyplot as plt # plotting library

import seaborn as sns # another plotting library


#%% 
# Dataset used: Advertising.csv
# Dataset can be downloaded from here-  https://www.statlearning.com/resources-first-edition

file_path = '/Users/haroon/OneDrive - Indian Institute of Science Education and Research Bhopal/IoT-Fall21/python-codes/Advertising.csv'

ad = pd.read_csv(file_path, usecols=[1,2,3,4])

#%% Dataset overview
ad.head() # see top elements

ad.info() # get dataset  details

ad.describe() # get dataset stats

ad.corr() # understand correlation
#%% UNIVARIATE REGRESSION
# Regression coefficients (Ordinary Least Squares) with Scikit Learn library

reg_mod1 = skl_lm.LinearRegression()

X = ad.TV.values.reshape(-1,1)   # converts this into column vector

y = ad.sales

reg_mod1.fit(X,y)   # y = b0 + b1 * x1

print(reg_mod1.intercept_) 

print(reg_mod1.coef_)


# write the equation in terms of determined coefficients

#sales = 7.03 + 0.04 * TV + epsilon


#% Predict sales with the created model
sales_pred1 = reg_mod1.predict(X) 

r2_score(y, sales_pred1) # Coefficient of determination, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score

mean_squared_error(y,sales_pred1)  # mean squared error metric, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error

resid = y - sales_pred1

sse_mod1 = sum(resid**2) # sum of squares

res_df1 = pd.DataFrame({'actual':y, 'predicted':sales_pred1})

res_df1.index = range(1,len(y)+1)

res_df1.plot()

#%% 
# MULTIPLE LINEAR REGRESSION - Using more than one predictor

reg_mod2 = skl_lm.LinearRegression()

X = ad[['radio', 'TV']]  

y = ad.sales

reg_mod2.fit(X,y)   # y = b0 + b1 * x1 (radio) + b2 * x2 (TV) + epsion

print(reg_mod2.coef_)

print(reg_mod2.intercept_)


#sales  =  2.92 + 0.18799423*radio + 0.04575482*TV + epsilon

sales_pred2 = reg_mod2.predict(X) 
r2_score(y, sales_pred2)  
mean_squared_error(y,sales_pred2) 

resid = y - sales_pred2
sse_mod2 = sum(resid**2) # sum of squares

res_df2 = pd.DataFrame({'actual':y, 'predicted':sales_pred2})
res_df2.index = range(1,len(y)+1)
res_df2.plot()

#%%
reg_mod3 = skl_lm.LinearRegression()

X = ad[['radio', 'TV', 'newspaper']] # converts dataframe to matrix

y = ad.sales

reg_mod3.fit(X,y)

print(reg_mod3.coef_)

print(reg_mod3.intercept_)


sales_pred3 = reg_mod3.predict(X) 
r2_score(y, sales_pred3)  
mean_squared_error(y,sales_pred3)

resid = y - sales_pred3
sse_mod3 = sum(resid**2) # sum of squares

res_df3 = pd.DataFrame({'actual':y, 'predicted':sales_pred3})
res_df3.index = range(1,len(y)+1)
res_df3.plot()
#%%
#  COMPARING DIFFERENT MODELS

#AIC= 2k - 2ln(sse)
AIC_mod1 = 2*1 - 2*np.log(sse_mod1)

AIC_mod2 = 2*2 - 2*np.log(sse_mod2)

AIC_mod3 = 2*3 - 2*np.log(sse_mod3)

print('AIC of model 1, model 2 and model 3 are', AIC_mod1, AIC_mod2, AIC_mod3)

print('AIC of model 1, model 2, and model 3 are', np.round(AIC_mod1,2), np.round(AIC_mod2,2), np.round(AIC_mod3,2))
#%%
# Grid plot all lines
pd.plotting.scatter_matrix(ad, diagonal='kde')

sns.pairplot(ad)


#%% Plot all predictions along with actual values side by side
#%matplotlib auto # This prints the plot outside the spyder IDE

fig, axes = plt.subplots(1,3,figsize=(12,6))

res_df1.plot(ax = axes[0],subplots=False)

res_df2.plot(ax = axes[1],subplots=False)

res_df3.plot(ax = axes[2],subplots=False)

plt.show()


