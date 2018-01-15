#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:03:05 2017

@author: meysamhashemi
"""
#https://people.duke.edu/~ccc14/sta-663/PyStan.html
#Estimating mean and standard deviation of normal distribution X∼N(μ,σ2)
import pystan
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

norm_code = """
data {
    int<lower=0> n;
    real y[n];
}
transformed data {}
parameters {
    real<lower=0, upper=100> mu;
    real<lower=0, upper=10> sigma;
}
transformed parameters {}
model {
    y ~ normal(mu, sigma);
}
generated quantities {}
"""

norm_data = {
             'n': 100,
             'y': np.random.normal(10, 2, 100),
            }

fit = pystan.stan(model_code=norm_code, data=norm_data, iter=1000, chains=1)

print fit

trace = fit.extract()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
plt.hist(trace['mu'][:], 25, histtype='step');
plt.subplot(1,2,2);
plt.hist(trace['sigma'][:], 25, histtype='step');
        
sm = pystan.StanModel(model_code=norm_code)
op = sm.optimizing(data=norm_data)



def save(obj, filename):
    """Save compiled models for reuse."""
    import pickle
    with open(filename, 'w') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        
 
        
def load(filename):
    """Reload compiled models for reuse."""
    import pickle
    return pickle.load(open(filename, 'r'))

model = pystan.StanModel(model_code=norm_code)
save(model, 'norm_model.pic')

new_dat = {
             'n': 100,
             'y': np.random.normal(10, 2, 100),
            }

new_model = load('norm_model.pic')
fit2 = new_model.sampling(new_dat, chains=1)
print fit2
       
trace2 = fit2.extract()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
plt.hist(trace2['mu'][:], 25, histtype='step');
plt.subplot(1,2,2);
plt.hist(trace2['sigma'][:], 25, histtype='step');