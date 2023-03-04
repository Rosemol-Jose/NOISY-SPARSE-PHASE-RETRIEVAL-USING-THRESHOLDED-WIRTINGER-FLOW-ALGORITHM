# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:51:08 2020

@author: Rose
"""

#THRESHOLDED WIRTINGER FLOW ALGORITHm
import math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def initialise(y,A,m,p,alpha):
    A1 = A
    print("A1",A1)
    phi_square=(1/m)*y.sum(axis=0, dtype='float')
    t =( 1+alpha*math.sqrt(math.log10(m*p)/m))*phi_square
    print ("t",t)
    I_l=np.dot(np.square(np.transpose(A)),y)
    I_l=I_l/m
    print("I",I_l)
    W=np.zeros((p,p))
    R=[]
    for l in range(p):
        if I_l[l] <= t:
            R.append(l)
    A1[:,R]=0
    print("R ", R)
    print("A1 after setting zero", A1)
    for j in range(m):
        W += (1/m)*np.asscalar(y[j])*np.matmul(A1[j,:].reshape(-1,1),A1[j,:].reshape(1,-1))
    #print("W",W)
    evals, evecs = la.eig(W)
    maxcol = list(evals).index(max(evals))
    v1 = evecs[:,maxcol]
    initial_val=math.sqrt(phi_square)*v1
    initial_val=initial_val.reshape(p,1)
    return initial_val

def threshold_level(xhat,y,m,p,A):
        #y1=np.square(abs(np.dot(A,X)))  
        z=np.square(abs(np.matmul(A,xhat)))
        s=np.dot(np.transpose(z),np.square(z-y))
        c=beta*math.log10(m*p)/(m*m)
        tau=np.sqrt(c*s)
        #print("tau1",tau)
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
m=7000
p=1000

NSR=1
k1=[]
relative_error=[]
alpha=0.1
mu=0.01
T=100



#for alpha in range(alpha):
for k in np.arange(25,225,25):
    sparse_index=np.random.choice(p,p-int(k),replace=False) 
    x=10*np.random.randn(p,1)
    x[sparse_index] = [[0]]
    sigma=NSR*np.square(la.norm(x))
    epsilon=np.sqrt(sigma)*np.random.randn(m,1)
    A = np.random.randn(m,p)
    y=np.square(np.abs(np.matmul(A,x))) + epsilon
    
#print('Epsilon= ',Epsilon )

    xhat = initialise(y,A,m,p,alpha)
   
    k1.append(k)
    for i in range(T):
        print("Iteration: ", i)
        phi_square=(1/m)*y.sum(axis=0, dtype='float')
        #print('Phi_sq is', Phi_sq)
        tau=(mu/phi_square)*threshold_level(xhat,y,m,p,A)
        xhat -= (mu/phi_square)*gradient(xhat,y,m,A)
        xhat = soft_threshold(xhat,tau)
    error=min(la.norm(xhat+x),la.norm(xhat-x))/la.norm(x)
    print("Error: ",error)
    relative_error.append(error)
print("  relative_error",  relative_error)
print("k",k1)



plt.plot(k1, relative_error)
plt.xlabel('k') 

plt.ylabel('relative_error') 
  



