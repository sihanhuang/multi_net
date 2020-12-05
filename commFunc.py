# Community Related functions

import itertools
import numpy as np

def position(e):
    pos=[]
    for i in np.arange(e.max()+1):
        pos.append(np.flatnonzero(e==i))
    return pos
    
def summatrix(O):
    Ov = sum(O)
    return Ov[:,None]*Ov

def num(e):
    _, num = np.unique(e, return_counts=True)
    return num

def O(e,Ab):
    O = np.zeros((e.max()+1,e.max()+1))
    for i in np.arange(e.max()+1):
        for j in np.arange(e.max()+1):
            posi=position(e)
            O[i,j]=sum([Ab[index] for index in list(itertools.product(posi[i],posi[j]))])
    return O

def E(e,gamma,Z):
    E = np.zeros((e.max()+1,e.max()+1))
    for i in np.arange(e.max()+1):
        for j in np.arange(e.max()+1):
            posi=position(e)
            if i==j:
                E[i,j]=sum([np.exp(np.dot(Z[index[0],index[1],:],gamma)) for index in list(itertools.permutations(posi[i],2))])
            else:
                E[i,j]=sum([np.exp(np.dot(Z[index[0],index[1],:],gamma)) for index in list(itertools.product(posi[i],posi[j]))])
    return E

def nLL(e,gamma,Ab,Z):
    A = np.tril(Ab,-1)
    n = A.shape[0]
    return (np.sum(O(e,Ab)*np.log(E(e,gamma,Z)))/2-np.nansum(O(e,Ab)*np.log(O(e,Ab))/2-num(e)*np.log(num(e)/n)))/(n**2)

def nLLGamma(gamma,e,Ab,Z):
    n = Ab.shape[0]
    A = np.tril(Ab,-1)
    return (np.sum(O(e,Ab)*np.log(E(e,gamma,Z)))/2-np.sum(A*np.dot(Z,gamma)))/(n**2)
