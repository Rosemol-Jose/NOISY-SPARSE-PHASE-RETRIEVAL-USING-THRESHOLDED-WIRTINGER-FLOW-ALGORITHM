# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:57:17 2020

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
kbyp=0.1
NSR=1
beta=0.25
alpha=0.1
mu=0.01
T=1000
iteration=[]
success_n=[]
sparse_index=np.random.choice(p,p-int(kbyp*p),replace=False) 
x=10*np.random.randn(p,1)
x[sparse_index] = [[0]]
sigma=NSR*np.square(la.norm(x))
epsilon=np.sqrt(sigma)*np.random.randn(m,1)
print("epsilon",epsilon)
A = np.random.randn(m,p)
print("A",A)
y=np.square(np.abs(np.matmul(A,x))) + epsilon
print("y",y)
#print('Epsilon= ',Epsilon )

xhat = initialise(y,A,m,p,alpha)
#print("Initial value:", xhat)
#print("After call", A)
for i in range(T):
    print("Iteration: ", i)
    iteration.append(i)
    phi_square=(1/m)*y.sum(axis=0, dtype='float')
    #print('Phi_sq is', Phi_sq)
    tau=(mu/phi_square)*threshold_level(xhat,y,m,p,A)
    xhat -= (mu/phi_square)*gradient(xhat,y,m,A)
    xhat = soft_threshold(xhat,tau)
    error=min(la.norm(xhat+x),la.norm(xhat-x))/la.norm(x)
    success=1-error
    success_n.append(success)

print("True value of x: ",x)
print("Final estimate of x: ",xhat)

print("Error: ",error)


plt.plot(iteration,success_n)
plt.title('Signal recovery using Thresholded Wirtinger Flow')
plt.xlabel('no. of iterations') 

plt.ylabel('success') 