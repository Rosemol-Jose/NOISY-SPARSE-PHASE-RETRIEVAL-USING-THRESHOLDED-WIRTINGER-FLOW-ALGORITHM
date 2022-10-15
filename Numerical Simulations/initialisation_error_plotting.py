# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:51:02 2020

@author: Rose
"""

#varying alpha

#THRESHOLDED WIRTINGER FLOW ALGORITHM
"""Plotting varying initialisation parameter vs relative error"""
import math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def initialise(y,A,m,p,alpha):
    """
    Finding initial value for TWF equation
    """
    A1 = A
    phi_square=(1/m)*y.sum(axis=0, dtype='float')
    t =( 1+alpha*math.sqrt(math.log10(m*p)/m))*phi_square
    I_l=np.dot(np.square(np.transpose(A)),y)
    I_l=I_l/m
    W=np.zeros((p,p))
    R=[]
    for l in range(p):
        if I_l[l] <= t:
            R.append(l)
    A1[:,R]=0
    for j in range(m):
        W += (1/m)*np.asscalar(y[j])*np.matmul(A1[j,:].reshape(-1,1),A1[j,:].reshape(1,-1))
    evals, evecs = la.eig(W)
    maxcol = list(evals).index(max(evals))
    v1 = evecs[:,maxcol]
    initial_val=math.sqrt(phi_square)*v1
    initial_val=initial_val.reshape(p,1)
    return initial_val

def threshold_level(xhat,y,m,p,A):
    #determining threshold level using mathematical equation in the research paper
        z=np.square(abs(np.matmul(A,xhat)))
        s=np.dot(np.transpose(z),np.square(z-y))
        c=beta*math.log10(m*p)/(m*m)
        tau=np.sqrt(c*s)
        return tau
    
def soft_threshold(xhat,tau):
    x1 = np.zeros((p,1))
    for i in range(p):
        x1[i]=np.sign(xhat[i])*max(abs(xhat[i])-tau,0)
    return x1
    
def gradient(xhat,y,m,A):
    grad=np.zeros((1,p))
    for j in range(m):
        s=np.matmul(A[j,:],xhat)
        c=(np.square(abs(s))-y[j])*s*A[j,:]
        grad+=c
    grad=(1/m)*grad.reshape(p,1)
    return grad

# Parameters
#m->size of signal
m=7000
#p->length of signal
p=1000
kbyp=0.1
alpha_n=[]
NSR=1
beta=1
relative_error=[]
#mu->threshold gradient/tuning parameter
mu=0.01
#t=>no.of iterations
T=1000
#sparsity factor is randomly chosen
sparse_index=np.random.choice(p,p-int(kbyp*p),replace=False) 
x=10*np.random.randn(p,1)
x[sparse_index] = [[0]]

for alpha in np.arange(0,1.1,0.1):
    sigma=NSR*np.square(la.norm(x))
    #epsilon-> stochastic noise
    epsilon=np.sqrt(sigma)*np.random.randn(m,1)
    #A->design matrix
    A = np.random.randn(m,p)
    y=np.square(np.abs(np.matmul(A,x))) + epsilon
    #xhat->initial input
    xhat = initialise(y,A,m,p,alpha)
    alpha_n.append(alpha)
    for i in range(T):
        phi_square=(1/m)*y.sum(axis=0, dtype='float')
        tau=(mu/phi_square)*threshold_level(xhat,y,m,p,A)
        xhat -= (mu/phi_square)*gradient(xhat,y,m,A)
        xhat = soft_threshold(xhat,tau)
    error=min(la.norm(xhat+x),la.norm(xhat-x))/la.norm(x)
    print("Error: ",error)
    relative_error.append(error)
print("  relative_error",  relative_error)

plt.plot(alpha_n, relative_error)
plt.xlabel('alpha') 
plt.ylabel('relative_error') 



