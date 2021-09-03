# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:57:17 2019
@author: kevin esh
"""

# On 20151227 by MLdP <lopezdeprado@lbl.gov>
# Hierarchical Risk Parit
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist as euclidean

import opti
import pareto

#------------------------------------------------------------------------------
def Utility(ret, var, p = 1e2):
    return ret - 0.5*p*var
#------------------------------------------------------------------------------
def weights(cov, mu, p = 1e4):
    """ 
    Calcula el vector de pesos para el problema media-varianza 
    """
    
    diag = np.diag(cov)
    a1 = diag @ mu
    a2 = diag.sum()
    
    return (1/p) * diag * ((p-a1)/a2 + mu)
#------------------------------------------------------------------------------
def var_cluster(cov,indx_assets, mu):
    """
    Calcula la sub-varianza del cluster de activos con alta correlacion
    """
    
    _cov_ = cov.loc[indx_assets,indx_assets] # matrix slice
    _mu_ = mu.loc[indx_assets]
    _w_ = weights(_cov_, _mu_) #.reshape(-1,1)  
    
    return _w_@(_cov_@_w_) #[0,0]
#------------------------------------------------------------------------------
def QuasiDiag(link):
    
# Sort clustered items by distance
    link = link.astype(int)
    sorted_indx = pd.Series([link[-1,0],link[-1,1]])
    numItems = link[-1,3] # number of original items
    
    while sorted_indx.max() >= numItems:
        sorted_indx.index=range(0,sorted_indx.shape[0]*2,2) # make space
        df0=sorted_indx[sorted_indx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sorted_indx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sorted_indx=sorted_indx.append(df0) # item 2
        sorted_indx=sorted_indx.sort_index() # re-sort
        sorted_indx.index=range(sorted_indx.shape[0]) # re-index
    return sorted_indx.tolist()

#------------------------------------------------------------------------------
def first_bisection(cov, mu, indx_assets0, indx_assets1, num_assets, p=1e-4):
    
    # Unica inversion de la covarianza
    inv = np.linalg.inv(cov)
    a1 = np.sum( inv@mu )
    a2 = np.sum(np.sum(inv))
    lamb = (p-a1)/a2
    
    #Aplicamos gradiente conjugado para obtener los mejores pesos
    w = opti.gradient_conjugated(np.array(cov), np.array(mu)+lamb, dim = num_assets )
    
    cov0 = cov.loc[indx_assets0,indx_assets0] # matrix slice
    cov1 = cov.loc[indx_assets1,indx_assets1] # matrix slice
    
    w0 = w[:num_assets//2]
    w1 = w[num_assets//2:]
    
    var0 = w0 @ (cov0 @ w0)
    var1 = w1 @ (cov1 @ w1)
    
    return var1/(var0+var1)
#------------------------------------------------------------------------------
def recursion_bisection(cov, sorted_indx, mu):
    
    num_assets = len(sorted_indx)
    indx_assets = [sorted_indx[:num_assets//2], sorted_indx[num_assets//2:]]
    
    #Inicializacion de los pesos del portafolio
    w = pd.Series(1,index=sorted_indx)
    
    alpha = first_bisection(cov, mu, indx_assets[0], indx_assets[1], num_assets)
    
    w[indx_assets[0]] *= alpha # weight 1
    w[indx_assets[1]] *= 1-alpha # weight 2
    
    while len(indx_assets)>0:
        
        indx_assets = [ cluster[i:j] for cluster in indx_assets for i,j in \
                       ( (0,len(cluster)//2), (len(cluster)//2,len(cluster)) ) \
                       if len(cluster)>1] # bi-section
        
    
        for i in range(0,len(indx_assets),2): # parse in pairs
            indx_assets0=indx_assets[i] # cluster 1
            indx_assets1=indx_assets[i+1] # cluster 2
            
            var0 = var_cluster(cov,indx_assets0, mu)
            var1 = var_cluster(cov,indx_assets1, mu)
            alpha = var1/(var0+var1)
            
            w[indx_assets0] *= alpha # weight 1
            w[indx_assets1] *= 1-alpha # weight 2
            
    return w.sort_index()
#------------------------------------------------------------------------------
def metric(corr):
    """
    Metrica de distancia basada en la correlacion, donde cada elemento se 
    encuentra en el rango 0<=d[i,j]<=1
    
    ---------------------------------------------------------------------------
    Parámetros
    corr  : 2-D numpy array (matriz de correlación)
    ---------------------------------------------------------------------------
    Regresa: 
    2-D numpy array (la matriz de metricas)
    ---------------------------------------------------------------------------
    """
    
    return np.sqrt((1.0-corr)/2.0)
#------------------------------------------------------------------------------
def generateData(nObs,size0,size1,sigma1):
# Time series of correlated variables
#1) generating some uncorrelated data
    np.random.seed(seed=12345);random.seed(12345)
    x=np.random.normal(0,1,size=(nObs,size0)) # each row is a variable
    #2) creating correlation between the variables
    cols=[random.randint(0,size0-1) for i in range(size1)]
    y=x[:,cols]+np.random.normal(0,sigma1,size=(nObs,len(cols)))
    x=np.append(x,y,axis=1)
    x=pd.DataFrame(x,columns=range(1,x.shape[1]+1))
    return x,cols
#------------------------------------------------------------------------------
def plotCorrMatrix(path,corr,labels=None):
# Heatmap of the correlation matrix
    if labels is None:labels=[]
    plt.pcolor(corr)
    plt.colorbar()
    plt.yticks(np.arange(.5,corr.shape[0]+.5),labels)
    plt.xticks(np.arange(.5,corr.shape[0]+.5),labels)
    plt.savefig(path)
    plt.clf();plt.close() # reset pylab
    return

#------------------------------------------------------------------------------
def hrp(cov, mu, corr, dendogram=False):
    
    #2) Clustering: En esta fase se ordenan los activos de acuerdo al algoritmo 
    # Tree clustering que toma en cuenta la correlacion entre los activos
    num_assets = len(corr.columns)
    
    #matriz de distancia basada en la correlacion
    corr_dist = metric(corr) 
    corr_dist = euclidean(corr_dist)

    #metodo de clusterizacion. Creacion del arbol
    sch.is_valid_im(corr_dist, warning=False, throw=False) 
    link = sch.linkage(corr_dist, method='single') 
    
    #obtenemos los indices de los activos de acuerdo al tree clustering
    indices_ord = QuasiDiag(link) 
    
    
    if dendogram: 
        sch.dendrogram(link) #creamos el dendrograma
        plt.savefig("dendograma.png") 
        plt.close()
        
    
    
    #4) Asignacion de los pesos a traves de metodo de biseccion
    w = recursion_bisection(cov,indices_ord, mu) 
    
    ret = w @ mu
    var = w @ (cov @ w)
    vol = np.sqrt(var)

    
    print('retorno =', ret)
    print('volatilidad =', vol)
    print('sharpe =', ret/vol)
    print('utilidad =', Utility(ret,var))
    
#    plt.bar(range(num_assets), w, color='red' )
    
    return

#------------------------------------------------------------------------------
def find_max_portfolio(selected, cov, mu, corr):
    
    num_assets = np.sum(selected)
    reindexing = list(range(num_assets))
    
    new_cov = cov.iloc[selected,selected]
    new_cov.columns = reindexing
    new_cov.index = reindexing

    new_corr = corr.iloc[selected,selected]
    new_corr.columns = reindexing
    new_corr.index = reindexing

    new_mu = mu.iloc[selected]
    new_mu.index = reindexing    
    
    return hrp(new_cov, new_mu, new_corr)
    
#------------------------------------------------------------------------------
def main():
        
    path = os.getcwd()
    os.chdir(path.replace('\Codigo', '\Returns'))
    
    #1) Data Reading: Leemos la base de datos de los returnos
    # y obtenemos matriz de covarianza, correlacion y vector medias
    Data_stocks = pd.read_csv('stock_returns.csv', sep=';', header=None)
    num_assets = len(Data_stocks.columns)
    
    cov = Data_stocks.cov()
    corr = Data_stocks.corr()    
    mu = Data_stocks.mean()
    
    selected = np.random.randint(0,2, size=num_assets, dtype=bool)
        
    return find_max_portfolio(selected, cov, mu, corr)


if __name__=='__main__': main()