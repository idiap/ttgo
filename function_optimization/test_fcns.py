'''
    Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
    Written by Suhan Shetty <suhan.shetty@idiap.ch>,
   
    This file is part of TTGO.

    TTGO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    TTGO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TTGO. If not, see <http://www.gnu.org/licenses/>.
'''


import torch
import numpy as np
torch.set_default_dtype(torch.float64)

def Rosenbrock_2D(a=1, b=100, alpha=1):
    '''
        a 2D version of the Rosenbrock function with fixed coefficients (a,b)
        https://en.wikipedia.org/wiki/Rosenbrock_function
    '''
    def cost(x): 
        result = b*(x[:,1]-x[:,0]**2)**2 + (x[:,0]-a)**2
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x))

    return pdf, cost


def Rosenbrock_4D(alpha=1):
    '''
        a 4D version of the Rosenbrock function with coefficients considered as variables of the function
        a=1, b=100 represents the standard 2D Rosenbrock function
        https://en.wikipedia.org/wiki/Rosenbrock_function
    '''
    def cost(x): 
        result = x[:,1]*(x[:,3]-x[:,2]**2)**2 + (x[:,2]-x[:,0])**2
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x)) 

    return pdf, cost


def Rosenbrock_nD(n=2,alpha=1):
    '''
        nD version of Rosenbrock function: https://en.wikipedia.org/wiki/Rosenbrock_function
        actual minima is at (a, a^2, a, a^2,...,a, a^2). 
        Domain: [-2,2]
    '''

    def cost(x):
        a = x[:,0]
        b = x[:,1]
        y = x[:,2:] 
        result = 0.
        for i in range(y.shape[1]-1):
            result = result+b*(y[:,i+1]-y[:,i]**2)**2 + (y[:,i]-a)**2
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x))

    return pdf, cost

def Rosenbrock_nD_2(n=2,alpha=1):
    '''
        nD version of Rosenbrock function: https://en.wikipedia.org/wiki/Rosenbrock_function
        actual minima is at (a, a^2, a, a^2,...,a, a^2). 
        Domain: [-2,2]
    '''
    assert ((n%2)==0 and n>2), 'n has to be even number greater than 2'

    def cost(x):
        a = x[:,0]
        b = x[:,1]
        y = x[:,2:] 
        result = 0.
        for i in range(int(y.shape[1]/2)):
            result = result+b*(y[:,2*i+1]-y[:,2*i]**2)**2 + (y[:,2*i]-a)**2
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x))+1e-9

    return pdf, cost

def Himmelblaue_2D(alpha=1,a=11,b=7):
    '''
        a 2D function: https://en.wikipedia.org/wiki/Himmelblau%27s_function
        cost(x,y)=(x^2+y-11)^2+(x+y^2-7)^2
        Domain: [-5,5]
    '''
    def cost(x): 
        result = (x[:,0]**2+x[:,1]-a)**2 + (x[:,0]+x[:,1]**2-b)**2
        return result

    def pdf(x): # Cost-to-PDF transformation
        return torch.exp(-alpha*cost(x)) # or use:  1/(eps+cost(x))
    
    return pdf, cost



def Himmelblaue_4D(alpha=1):
    '''
        a 4D version of the Himmelblaue2D function with coefficients considered as variables of the function
        cost(a,b,x,y)=(x^2+y-a)^2+(x+y^2-b)^2
        a=11, b=7 represents the standard 2D Himmelblaue function
    '''
    def cost(x): 
        result = (x[:,2]**2+x[:,3]-x[:,0])**2 + (x[:,2]+x[:,3]**2-x[:,1])**2 #11, 7
        return result

    def pdf(x):
        return torch.exp(-alpha*cost(x)) + 1e-9 #or use: 1/(eps+cost(x))#

    return pdf, cost


def gmm(n=2,nmix=3,L=1,mx_coef=None,mu=None,s=0.1):
    """
        Mixture of spherical Gaussians (un-normalized)
        nmix: number of mixture coefficients
        n: dimension of the domain
        s: variance
        mu: the centers assumed to be in : [-L,L]^n
    """
    n_sqrt = torch.sqrt(torch.tensor([n]))
    if mx_coef is None: # if centers and mixture coef are not given, generate them randomly
        mx_coef = torch.rand(nmix)
        mx_coef = mx_coef/torch.sum(mx_coef)
        mu = (torch.rand(nmix,n)-0.5)*2*L

    def pdf(x):
        result = torch.tensor([0])
        for k in range(nmix):
            l = torch.linalg.norm(mu[k]-x, dim=1)/n_sqrt
            result = result + mx_coef[k]*torch.exp(-(l/s)**2)
        return result 

    def cost(x):
        return 1.-pdf(x)

    return pdf, cost


def sine_nD(n=2, alpha=1):
    "an nD sinusoidal surface"
    n_sqrt = torch.sqrt(torch.tensor([n]))
    def pdf(x): 
        return 0.49*(1.001+torch.sin(4*torch.pi*torch.linalg.norm(x,dim=1)/n_sqrt))

    def cost(x):
        return 1.-pdf(x)

    return pdf, cost


