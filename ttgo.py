'''
    Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
    Written by Suhan Shetty <suhan.shetty@idiap.ch>
   
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


""" 
    This class contains the pytorch implementation of the whole pipeline of TTGO:
     - Input:
        - cost: the cost function,
        - pdf:  the corresponding density funtion (pdf),
        - domain: the discretization of the domain of the pdf,
        - max_batch: specifies the maximum batch size (decrease it if you encounter memory issues)
        - sites_task: a list containing the modes corresponding to the task parameters (optional). You can instead
                        use set_sites() at test time

     - cross_approximate:  Fit the TT-model to the given PDF using TT-Cross (Uses tntorch library)
     - Sample from the TT-Model 
        - set the sites/modes for task parameters using set_sites() before calling sample (or use set_task at initialization)
        - two different samplers are provided: based on 1-norm or 2-norm as the density function
        - prioritized sampling can be done by setting alpha parameter in sampling()
     - Choose the best sample(s)
     - Fine-tune the best sample(s) using gradient-based optimization
        - We use scipy.optimize for this

"""

import numpy as np
import torch
import tntorch as tnt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import copy
import warnings
# torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")


class TTGO:
    def __init__(self, domain, cost, pdf, sites_task=[], max_batch=10**5, device="cpu"):
        self.device = device
        self.domain = [x.to(self.device) for x in domain] # a list of  1-D torch-tensors containing the discretization points along each axis/mode 
        self.d = [len(x) for x in domain] # number of discretization points along each axis/mode
        self.dim = len(domain) # dimension of the tensor
        self.pdf = pdf # density function
        self.cost = cost # cost function
        self.canonicalized = False 
        self.sites_task=[] # sites/variables are to be conditioned from tt-model/pdf (could be fed just before calling sample method)
        self.max_batch=max_batch # maximum batch size for cross-approximation (to avoid memory overflow)
        
        # For optimization/fine-tuning
        lb = []; ub = []
        for domain_i in self.domain:
            lb.append(domain_i[0].item())
            ub.append(domain_i[-1].item())
        self.scipy_bounds = Bounds(np.array(lb),np.array(ub))


    def to(self,device='cpu'):
        self.device = device
        self.domain = [x.to(device) for x in self.domain]
        if self.tt_model:
            self.tt_model.to(device)
            
    def clone(self):
        return copy.deepcopy(self)
    

    def pdf_batched(self,x):
        ''' To avoid memorry issues with large batch processing in tt-cross, reduce computation into smaller batches '''   
        batch_size = x.shape[0]
        pdf_values = torch.zeros(batch_size).to(self.device)
        num_batch = batch_size//self.max_batch
        end_idx = 0
        for i in range(num_batch):
            start_idx = i*self.max_batch
            end_idx = (i+1)*self.max_batch
            pdf_values[start_idx:end_idx] =self.pdf(x[start_idx:end_idx,:])
        if batch_size>end_idx:          
            pdf_values[end_idx:batch_size] = self.pdf(x[end_idx:batch_size,:])
        return pdf_values 

    def cross_approximate(self, rmax=500, nswp=10, min_iter=1000, 
        eps=1e-2, kickrank=5, verbose=True):
        ''' TT-Cross Approximation
            eps: precentage change in the norm of tt per iteration of tt-cross
         '''
        self.tt_model = tnt.cross(self.pdf_batched, domain=self.domain, max_iter=nswp, min_iter=min_iter,
            eps=eps,rmax=rmax, kickrank=kickrank,function_arg='matrix',device=self.device, verbose=verbose)
        self.canonicalize(self.sites_task)

    def set_sites(self, sites_task): 
        '''
            sites/variables to be conditioned from tt-model/pdf (modes corresponding to task parameters)
            Call this before calling sample() if you have not specified this while initializing the class
        '''
        self.sites_task = sites_task
        self.canonicalize(self.sites_task)


    def round(self, eps=1e-6,rmax=500):
        ''' Compress the tt_model: speeds up sampling and improves efficiency of represention '''
        self.tt_model.round_tt(eps,rmax)
        self.canonicalize(self.sites_task)

    def __getitem__(self,idxs):
        return self.tt_model[idxs].torch()

    def choose_best_sample(self,samples):
        """Given a batch of samples (candidates for optima), find the best sample"""
        cost_values = self.cost(samples)
        idx = torch.argmax(-cost_values)
        best_sample = samples[idx,:].view(1,-1)
        return best_sample


    def choose_best_sample_from_idx(self,sample_idx):
        """
            Given the samples in the index form, find the best sample using the TT-Model of the pdf.
            Use this only if your TT-Model is accurate!!!!
        """
        pdf_values = self.tt_model[sample_idx].torch()
        idx = torch.argmax(pdf_values)
        best_sample = self.idx2domain(sample_idx[idx,:].view(1,-1))
        return best_sample


    def choose_top_k_sample(self,samples,k=1):
        '''Given the samples choose the best k samples '''
        values, idx = torch.topk(-self.cost(samples), k)
        return samples[idx,:]

    def choose_top_k_sample_from_idx(self,sample_idx,k=1):
        '''Given the sample idx choose the best k samples '''
        pdf_values = self.tt_model[sample_idx].torch()
        values, idx = torch.topk(pdf_values, k)
        samples = self.idx2domain(sample_idx[idx,:])
        return samples


    def idx2domain(self,I):
        ''' Map the index of the tensor/discretization to the domain'''
        X = torch.zeros(I.shape)
        for i in range(I.shape[1]):
            X[:,i] =  self.domain[i][I[:, i]]
        return X


    def domain2idx(self, x_task):
        ''' Map the states from the domain (a tuple of the segment) to the index of the discretization '''
        I = (x_task*0)
        for i, site in enumerate(self.sites_task):
            I[:,i] = torch.argmin(torch.abs(x_task[:,i].view(-1,1)-self.domain[site]), dim=1) 
        return I.long()


    def optimize(self, x, bound=True, method='SLSQP', tol=1e-3):
        ''' 
            Optimize from an initial guess x.
            To Do: Move it to pytorch based optimization instead of depending on scipy (slow)
            method: 'L-BFGS-B' or 'SLSQP'
            bound: if True the optimizaton (decision) variables  will be constrained to the domain provided
        '''
        sites_task = self.sites_task

        # pytorch-to-numpy interface
        @torch.enable_grad()
        def cost_fcn(x):
            return self.cost(torch.from_numpy(x).reshape(1,-1).to(self.device)).to("cpu").numpy()
        @torch.enable_grad()
        def jacobian_cost(x):
            jac= torch.autograd.functional.jacobian(self.cost,torch.from_numpy(x).reshape(1,-1)).reshape(-1)
            jac[sites_task] = 0
            return jac.numpy().reshape(-1)
        
        if bound ==True: # constrained optimization
            results = minimize(cost_fcn, x.numpy().reshape(-1), method=method,jac=jacobian_cost, tol=tol, bounds=self.scipy_bounds)
        else: # unconstrained optimization
            results = minimize(cost_fcn, x.numpy().reshape(-1), method=method,jac=jacobian_cost, tol=tol)
        return torch.from_numpy(results.x).view(1,-1), results


    def sample(self,n_samples=1, x_task=torch.tensor([0]), alpha=0., norm=1): 
        '''
             Draw samples from the TT-distribution 
            :param x_task: the task-paramters, if any (use set_sites() to set the sites/modes for conditioning)
            :param n_samples: how many samples to draw from the TT-distribution?
            :param alpha: (0,1). O filters none but 1 highly prioritized high densiy regions (a heuristic) 
        '''
        alpha = np.clip(alpha,0,1)
        x_task = x_task.to(self.device) # the task parameters 
        if norm==1: # P(x) = |R(x)|/Z, where R is the tt_model of the unnormalized pdf 
            samples, samples_idx =  self.sample_1norm(n_samples=n_samples,x_task=x_task, alpha=alpha)
        else: # P(x) = |R(x)|^2/Z, R is the tt_model of the pdf 
            samples, samples_idx =  self.sample_2norm(n_samples=n_samples,x_task=x_task, alpha=alpha)

        return samples, samples_idx

    
    def canonicalize(self,sites_task):
        ''' Canonicalize the tt-cores '''
        self.sites_task = sites_task
        if len(self.sites_task)==0:
            self.tt_model.left_orthogonalize(self.dim-2)
        else:
            self.tt_model.left_orthogonalize(max(0,self.sites_task[0]-1))
            if not sites_task[-1]==0:
                self.tt_model.right_orthogonalize(self.sites_task[-1])
        self.canonicalized = True


    def batch_sample(self,M,alpha):
        """
            Treat each row of a matrix M as a PMF and select a column per row according to it
        """
        M = torch.abs(M)

        #fi ltering low pmf samples
        M_max, _ = torch.max(M,dim=1)
        M_max  = M_max.repeat(M.shape[1],1).T
        M = M/M_max
        M = M**(1/(1e-9+1-alpha)) # higher density is given higher importance
        M /= torch.sum(M, dim=1)[:, None]  # Normalize row-wise
        rows = torch.multinomial(M,1)
        return rows.view(-1)


    def sample_1norm(self,n_samples=1, x_task=torch.tensor([0]),  alpha=0.):
        """
        Generate n_samples pointsfrom a joint PDF distribution represented by a tensor in TT-format.
        P(x) = |R(x)|/Z, where R is the tt-model of the unnormalized pdf
        """
        # Warm-up for mass sampling
        tt_cores = self.tt_model.tt().cores
      
        Xs = torch.zeros([n_samples, self.dim]).long()

        x_task = x_task.view(1,-1)
        idx_task = self.domain2idx(x_task).view(-1)
        for i,site in enumerate(self.sites_task):
            tt_cores[site] = tt_cores[site][:,idx_task[i],:].view(tt_cores[site].shape[0],1,tt_cores[site].shape[-1])    

        tt_cores_summed = [torch.sum(core,dim=1) for core in tt_cores] #tnt.sum(t, dim=np.arange(self.dim), keepdim=True).decompress_tucker_factors()
        rights = [torch.ones(1).to(self.device)]
        for core in tt_cores_summed[::-1]:
            rights.append(torch.matmul(core, rights[-1]))
        rights = rights[::-1]
        fibers = [torch.einsum('ijk,k->ij', (tt_cores[site], rights[site+1])) for site in range(self.dim)]

        lefts = torch.ones([n_samples, 1]).to(self.device)

        for site in range(self.dim):
            pmf = torch.einsum('ij,jk->ik', (lefts, fibers[site]))
            rows = self.batch_sample(pmf,alpha)
            Xs[:, site] = rows
            lefts = torch.einsum('ij,jik->ik', (lefts, tt_cores[site][:, rows, :]))

        for i, site in enumerate(self.sites_task):
            Xs[:,site] = idx_task[i]

        samples_idx = Xs
        samples = self.idx2domain(samples_idx)

        return samples, samples_idx

    


    def sample_2norm(self, n_samples=1, x_task=torch.tensor([0]).reshape(1,-1), alpha=0.):  
        ''' 
            Note: Make sure to call self.canonicalize(sites_task) to warmup before calling this method 
            if norm==2
        '''
        x_task = x_task.view(1,-1)
        if len(self.sites_task)==0: # initialize for unconditioned sampling
            l,r = (self.dim,self.dim)
        else:
            l,r = (self.sites_task[0],self.sites_task[-1])

        eps= 1e-9

        Xs = torch.zeros([n_samples, self.dim]).long()
        idx_task = self.domain2idx(x_task).view(-1)
        for i, site in enumerate(self.sites_task):
            Xs[:,site] = idx_task[i]

        current_site=0
        vec0 = torch.tensor([1.]).view(1,1).to(self.device)
        for site in self.sites_task:
            vec0 = torch.matmul(vec0, self.tt_model.cores[site][:, idx_task[site], :]) 
            vec0 /= (torch.linalg.norm(vec0)+1e-16)
            current_site+=1

        vec = vec0.reshape(1,-1).repeat(n_samples,1) # n_samples x  r
        for site in range(current_site, self.dim):
            vec = torch.einsum('ij,jkl->ikl', (vec, self.tt_model.cores[site]))
            pmf = torch.linalg.norm(vec, dim=2)**2/(torch.linalg.norm(vec,dim=(1,2))**2+eps).view(-1,1) + eps
            sample_sites = self.batch_sample(pmf,alpha)
            Xs[:,site] = sample_sites 
            vec = vec[0, sample_sites]
            vec = vec/(torch.linalg.norm(vec,dim=-1)+eps).view(-1,1)

        for site in range(l-1, -1, -1):
            vec = torch.einsum('ij,lkj->ikl', (vec,self.tt_model.cores[site])) # change the order if error 
            pmf = torch.linalg.norm(vec, dim=2)**2/(torch.linalg.norm(vec,dim=(1,2))**2+eps).view(-1,1) + eps
            sample_sites = self.batch_sample(pmf,alpha)
            Xs[:,site] = sample_sites 
            vec = vec[0, sample_sites]
            vec = vec/(torch.linalg.norm(vec,dim=-1)+eps).view(-1,1)

        samples_idx = Xs
        samples = self.idx2domain(samples_idx)
                
        return samples, samples_idx


    def sample_random(self, n_samples, x_task=torch.tensor([0]).reshape(1,-1)):
        ''' sample from the uniform distribution from the domain '''
        samples_idx = torch.zeros([n_samples, self.dim]).long().to(self.device)

        for site in range(self.dim):
            idxs = torch.multinomial(input=torch.tensor([1.]*len(self.domain[site])), num_samples=n_samples, replacement=True)
            samples_idx[:,site] = idxs

        x_task = x_task.view(1,-1).to(self.device)
        idx_task = self.domain2idx(x_task).view(-1)
        samples_idx[:,self.sites_task] = idx_task
        samples = self.idx2domain(samples_idx)
        return samples, samples_idx


