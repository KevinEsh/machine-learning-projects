# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:38:28 2019

@author: kevin
"""

import numpy as np
import random

#===============================================================================================================================
def function(individual, r=18):
    return np.sum(individual)
#===============================================================================================================================
def mutate(child, mr=0.055555, r=18):
    mutation_indx = np.random.random(r) < mr
    return np.logical_xor(child, mutation_indx)
#===============================================================================================================================
def crossover(parent1, parent2, r=18, mr=0.055555, method='two'):

    if method == 'two':
        #Corte de 2 puntos
        rank = range(1,r)
        cut = sorted(random.sample(rank,2))
        child1 = np.concatenate( (parent1[:cut[0]], parent2[cut[0]:cut[1]], parent1[cut[1]:]) )
        child2 = np.concatenate( (parent2[:cut[0]], parent1[cut[0]:cut[1]], parent2[cut[1]:]) )
        return child1, child2

    elif method == 'one':
        #Corte de 1 punto
        cut = np.random.randint(1, r)
        child1 = np.concatenate( (parent1[:cut], parent2[cut:]) )
        child2 = np.concatenate( (parent2[:cut], parent1[cut:]) )
        return  child1, child2

    elif method == 'uni':
        #Cruza Uniforme
        uni = np.random.randint(0,2, size=r, dtype=bool)
        temp = np.copy(parent1)
        parent1[uni] = parent2[uni]
        parent2[uni] = temp[uni]
        return parent1, parent2

    return None
#===============================================================================================================================
def new_generation(P, gbest1, r=18, cr=0.8, mr=0.055555, elit=False):
    n_ind, r = P.shape
    P_new = P[:2]

    #Selection
    #La mejor mitad de la poblacion producirá la nueva generacion.
    if elit:

        gbest2 = P[np.random.randint(1,n_ind//2)]
        child1, child2 = crossover(gbest1, gbest2, r, mr)
        P_new = np.vstack( (P_new, child1, child2) )

        parents_indx = np.random.randint(0, n_ind//2, size=(n_ind//2 -2, 2) )
    else:
        parents_indx = np.random.randint( 0,n_ind//2, size=(n_ind//2 -1, 2) )

    for i1, i2 in parents_indx:

        #Crossover
        child1, child2 = crossover(P[i1], P[i2], r, mr)

        if random.random() > cr:
            #Mutation
            child1 = mutate(child1, mr, r)
            child2 = mutate(child2, mr, r)

        P_new = np.vstack( (P_new, child2, child1) )

    return P_new
#===============================================================================================================================
def Genetic_Alg(function, max_iter, n_ind=200, r=18, cr=0.8, mr=1/18, elit=False, tol=1e-6, **kwarg):

    #Generacion de la poblacion inicial
    P = np.random.randint(0,2, size=(n_ind, r), dtype=bool)
    F = np.array([function(x, kwarg['cov'], kwarg['mu'], kwarg['corr']) for x in P ])

    #Ordenamos los individuos por su aptitud
    sorted_ind = F.argsort()
    P = P[sorted_ind[::-1]]
    F = F[sorted_ind[::-1]]

    #Seleccion del mejor individuo
    gbest = np.copy(P[0])
    fbest = F[0]

    #Genetic Algorithm - Binary Representation
    evals = 0
    while evals < max_iter:

        #Selection + Crossover + Mutation
        P = new_generation(P, gbest, r, cr, mr, elit)
        F = np.array([function(x, kwarg['cov'], kwarg['mu'], kwarg['corr']) for x in P])

        #Ordenamos los individuos por su aptitud
        sorted_ind = F.argsort()
#        print(F[sorted_ind[::-1]])
        P = P[sorted_ind[::-1]]
        F = F[sorted_ind[::-1]]

        #Actualizacion del mejor individuo
        if F[0] > fbest:
            gbest = np.copy(P[0])
            fbest = F[0]

        evals+=1
        print('gen:', evals, 'max_best:', fbest)
        if abs(fbest-r) < tol: return evals, fbest

    return P


if __name__ == '__main__':
	Genetic_Alg(function, max_iter=100000, n_ind=80, r=55, cr=0.8, mr=1/50)
