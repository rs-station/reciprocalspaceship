#!/usr/bin/env python

import numpy as np
import pymc3 as pm
import pandas as pd



inFN = "fw_test_data.csv"
outFN = "fw_mcmc_data.csv"
nproc=7
chain_length = 30_000
burnin = 15_000

df = pd.read_csv(inFN)
I,SigI,Sigma,J,SigJ,F,SigF,Centric = df.to_numpy(np.float64).T
Centric = Centric.astype(np.bool)
df

#Gamma prior params
a = np.where(Centric, 0.5, 1.)
scale = 1./np.where(Centric, 2.*Sigma, Sigma)

with pm.Model() as model:
    Wilson = pm.distributions.Gamma('Wilson', a, scale, shape=len(a))
    likelihood = pm.distributions.Normal('Likelihood', mu=Wilson, sigma=SigI, observed=I)
    trace = pm.sample(draws=chain_length, tune=burnin, cores=nproc)

samples = trace.get_values('Wilson')
mc_J = samples.mean(0)
mc_SigJ = samples.std(0)
mc_F = np.sqrt(samples).mean(0)
mc_SigF = np.sqrt(samples).std(0)


df['FW-I'] = mc_J
df['FW-SigI'] = mc_SigJ
df['FW-F'] = mc_F
df['FW-SigF'] = mc_SigF

df.to_csv(outFN, index=False)
