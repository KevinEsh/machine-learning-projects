# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def print_efficient_front(cov, mu, num_portf=500, p=1e2, **kwargs):

    num_assets = mu.shape[0]

    try:
        retorno = kwargs['retorno']
        volatil = kwargs['volatil']
        sharpe = kwargs['sharpe']
        argmax = sharpe.argmax()
        max_sharpe = (volatil[argmax], retorno[argmax])
    except:
        pass

    # Random portfolios
    weights = np.random.uniform(0,1,size=(num_portf, num_assets))
    weights = weights/weights.sum(axis=1)[:,None]

    # Return, Volatility and Sharpe arrays
    ret_arr = np.empty(num_portf)
    vol_arr = np.empty(num_portf)
    sharpe_arr = np.empty(num_portf)
#    utility = np.empty(num_portf)

    for x in range(num_portf):
        # Expected return
        ret_arr[x] =  mu @ weights[x]
        # Expected volatility
        vol_arr[x] = np.sqrt( weights[x] @ (cov @ weights[x]) )
        # Sharpe Ratio
        sharpe_arr[x] = ret_arr[x]/vol_arr[x]
        # Utility
#        utility[x] = ret_arr[x] - 0.5*p*vol_arr[x]**2

#    plt.bar(range(num_assets), weights[argmax], color='red' )
    plt.figure(figsize=(12,8))
    plt.grid(True)
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='magma', alpha=0.7)

    try:
        plt.scatter(volatil, retorno, c=sharpe, marker="v", cmap='magma')
        plt.scatter(volatil[max_sharpe[0]], retorno[max_sharpe[1]], c='black', marker='*')
    except:
        pass
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.xlim(0.9,1.4)

    plt.show()

#    print(utility.max())
#    print('retorno =', ret_arr[argmax])
#    print('volatilidad =', vol_arr[argmax])
#    print('sharpe =', sharpe_arr[argmax])
#    print('utilidad =', utility[argmax])

def main():

    path = os.getcwd()
    os.chdir(path.replace('\Codigo', '\Returns'))
    stocks = pd.read_csv('stock_returns.csv', sep=';', header=None)

    # Covariance matrix
    cov = stocks.cov()
    mu = stocks.mean()

    print_efficient_front(cov, mu, 500)


if __name__=="__main__":
    main()
