# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:59:48 2019

@author: kevin
"""
import numpy as np
norm = np.linalg.norm

#------------------------------------------------------------------------------
def gradient_conjugated(Q, b, x = None, tol_r=1e-5, max_iter=100, **kwargs):
    
    try: n = kwargs['dim']
    except: raise KeyError("Dimension 'dim' del vector x no fue dada")
        
    if np.any(x) == None: x = np.random.random(n)
    if Q.shape[1] != x.shape[0]: raise Warning("Dimensiones de Q {} y x {} no coinciden".format(Q.shape, x.shape))
    
    #Inicializacion del residuo
    res = Q@x - b
    pk = -np.copy(res)
    
    #Historial de optimización
    itera = 1
    while norm(res) > tol_r or itera < max_iter:
        Qpk = Q@pk
        alp = (res@res)/(pk@Qpk)
        x += alp*pk
        res2 = res + alp*Qpk
        beta = (res2@res2)/(res@res)
        res = np.copy(res2)
        pk = -res + beta*pk
        
        itera+=1
    return x
#------------------------------------------------------------------------------