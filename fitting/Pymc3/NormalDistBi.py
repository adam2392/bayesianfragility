#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 23:46:48 2017

@author: meysamhashemi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:18:54 2016

@author: meysamhashemi
"""


#http://people.duke.edu/~ccc14/sta-663-2016/16C_PyMC3.html
#https://pymc-devs.github.io/pymc3/notebooks/LKJ.html
import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import NUTS, find_MAP, sample, Slice, traceplot, summary
from scipy import optimize
import pymc3 as pm

#import theano 
#import theano.tensor as tt
#from theano import tensor, function, grad
#theano.config.optimizer='None'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'ignore'
#theano.config.compute_test_value = 'warn'
#('off', 'ignore', 'warn', 'raise', 'pdb')


# generate observed data
N = 1000



mu1 =-5
mu2=5
a=1/3
b=2/3 
sigma1 =1
sigma2 =3





y1= np.random.normal(mu1, sigma1,N)
y2= np.random.normal(mu2, sigma2,N)
y=np.r_[y1,y2]

plt.hist(y,30)
plt.xlim(-15, 15)
#_mu = np.array([0,0,0])
#_sigma = np.array([1,5,5])
#y=np.c_[y1,y2,y1]


niter = 5000

with pm.Model() as model:
    # define priors
    mu = pm.Uniform('mu', lower=-30, upper=30, shape=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10, shape=1)


    # define likelihood
    y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)

    # inference
    start = pm.find_MAP()
    #step = pm.Metropolis()
    #trace = pm.sample(niter, step, progressbar=True)
    #step = pm.Slice()
    #trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)
    step = pm.NUTS()
    trace = pm.sample(niter, step, start, progressbar=True)
    trace_mu=trace['mu']
    trace_sig=trace['sigma']


        
     
    ax = traceplot(trace[-niter:], figsize=(8,5),  
     lines={k: v['mean'] for k, v in pm.df_summary(trace[-niter:]).iterrows()})

     
    Summary=pm.df_summary(trace[-niter:])
    
    

