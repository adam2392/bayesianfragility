#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:18:54 2016

@author: meysamhashemi
"""

import numpy as np
import matplotlib.pyplot as plt
from pymc3 import Model, Normal, HalfNormal
from pymc3 import NUTS, find_MAP, sample, Slice, traceplot, summary
from scipy import optimize
import pymc3 as pm

import theano 
import theano.tensor as tt
from theano import tensor, function, grad
#theano.config.optimizer='None'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'ignore'
#theano.config.compute_test_value = 'warn'
#('off', 'ignore', 'warn', 'raise', 'pdb')


# generate observed data
N = 100

nn=5

_mu = range(1,nn+1)*np.ones((1,nn))
_sigma = range(1,nn+1)*np.ones((1,nn))


y=np.empty((nn,N))

for i in range(0, nn):
    y[i,:]= np.random.normal(_mu[0,i], _sigma[0,i], N)

#_mu = np.array([0,0,0])
#_sigma = np.array([1,5,5])
#y=np.c_[y1,y2,y1]

y=y.T

niter = 10000

with pm.Model() as model:
    # define priors
    mu = pm.Uniform('mu', lower=-100, upper=100, shape=_mu.shape)
    sigma = pm.Uniform('sigma', lower=0, upper=10, shape=_sigma.shape)

    # define likelihood
    y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)

    # inference
    start = pm.find_MAP()
    #step = pm.Metropolis()
    #trace = pm.sample(niter, step, progressbar=True)
    step = pm.Slice()
    trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)
    #step = pm.NUTS()
    #trace = pm.sample(niter, step, start, progressbar=True)
    trace_mu=trace['mu']
    trace_sig=trace['sigma']


        
     
    ax = traceplot(trace[-niter:], figsize=(8,5),  
     lines={k: v['mean'] for k, v in pm.df_summary(trace[-niter:]).iterrows()})

     
    Summary=pm.df_summary(trace[-niter:])
    
    
        
plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
plt.hist(trace['mu'][-niter/2:,0], 20);
plt.subplot(1,2,2);
plt.hist(trace['sigma'][-niter/2:,0], 20);
        
plt.figure(figsize=(4,4))        
plt.plot(trace_mu[-niter*3/4:,0,0],trace_mu[-niter*3/4:,0,nn-1],'r*')