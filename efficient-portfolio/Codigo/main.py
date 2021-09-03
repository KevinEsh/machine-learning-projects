# -*- coding: utf-8 -*-
"""
Created on Sun May 26 23:21:17 2019

@author: kevin
"""

import pandas as pd
import numpy as np
import os

import genetic_algorithm as ga
import pareto as pa
import portfolio as port
#%%

path = os.getcwd()
os.chdir(path.replace('\Codigo', '\Returns'))

#1) Data Reading: Leemos la base de datos de los returnos
# y obtenemos matriz de covarianza, correlacion y vector medias
print("Reading Data...", end=' ')
Data_stocks = pd.read_csv('stock_returns.csv', sep=';', header=None)
num_assets = len(Data_stocks.columns)
print("done!")

print("Calculating Covariance...", end=' ')
cov = Data_stocks.cov()
corr = Data_stocks.corr()
mu = Data_stocks.mean()
print("done!")

print("Executing GA procedure...", end="\n\n")
portfolios = 80
P = ga.Genetic_Alg(port.find_max_portfolio,
			    max_iter=15,
				 n_ind = portfolios,
				  r = num_assets,
				   cr = 0.8,
				    mr = 1/50,
					elit=False,
					 cov=cov, mu=mu, corr=corr )


print("Ploting Pareto...", end=' ')
#%%
returns = np.empty(int(portfolios/2))
volatility = np.empty(int(portfolios/2))
#%%
for x in range(int(portfolios/2)):
	volatility[x], returns[x] = port.find_max_portfolio(P[x], cov, mu, corr, sharpe=True)
sharpe = returns/volatility

for x in returns[::-1]:
	print(x)
print()
for x in volatility[::-1]:
	print(x)
#%%
pa.print_efficient_front(cov, mu, 5000, retorno=returns, volatil=volatility, sharpe=sharpe)
print("done!")
