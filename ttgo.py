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
        - tt_model: corresponding to the pdf (e.g.: tt model of exp(-cost(x)))
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

"""

import numpy as np
import torch
import tntorch as tnt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import copy
import warnings
import tt_utils
# torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")


class TTGO:
    def __init__(self, domain, cost, tt_model, sites_task=[],max_batch=10**5, device="cpu"):
        self.device = device
        self.domain = [x.to(self.device) for x in domain] # a list of  1-D torch-tensors containing the discretization points along each axis/mode 
        self.min_domain = torch.tensor([x[0] for x in domain]).to(device)
        self.max_domain = torch.tensor([x[-1] for x in domain]).to(device)
        self.n = torch.tensor([len(x) for x in domain]).to(device) # number of discretization points along each axis/mode
        self.dim = len(domain) # dimension of the tensor
        self.tt_model = tt_model.to(device)
        self.canonicalize()
        self.cost = cost # the total cost function
        self.sites_task = sites_task
        
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
    
    def pdf(self,x):
        return -self.cost(x)
    

    def idx2domain(self,I):
        ''' Map the index of the tensor/discretization to the domain'''
        return tt_utils.idx2domain(I=I, domain=self.domain, device=self.device)


    def domain2idx(self, x_task):
        ''' Map the states from the domain (a tuple of the segment) to the index of the discretization '''
        return tt_utils.domain2idx(x=x_task, domain=self.domain[:x_task.shape[-1]], device=self.device, uniform=False)


    def __getitem__(self,idxs):
        return self.tt_model[idxs].torch()

    def choose_best_sample(self,samples):
        '''
        Given the samples (candidates for optima), find the best sample
        samples: batch_size x n_samples x dim (batch_size corresponds to the number of task-parameter)
        '''
        cost_values = self.cost(samples.view(-1,samples.shape[-1])).view(samples.shape[0],samples.shape[1])
        idx = torch.argmax(-cost_values, dim=-1)
        best_sample = samples[torch.arange(samples.shape[0]).unsqueeze(1),idx.view(-1,1),:]
        return best_sample.view(-1, 1, samples.shape[-1]) # batch_size x 1 x dim


    def choose_top_k_sample(self,samples,k=1):
        '''Given the samples choose the best k samples '''
        cost_values = self.cost(samples.view(-1,samples.shape[-1])).view(samples.shape[0],samples.shape[1])
        values, idx = torch.topk(-cost_values, k, dim=-1)
        return samples[torch.arange(samples.shape[0]).unsqueeze(1),idx,:]


    def optimize(self, x, bound=True, method='SLSQP', tol=1e-3):
        ''' 
            Optimize from an initial guess x.
            To Do: Move it to pytorch based optimization instead of depending on scipy (slow)
            method: 'L-BFGS-B' or 'SLSQP'
            bound: if True the optimizaton (decision) variables  will be constrained to the domain provided
        '''
        # pytorch-to-numpy interface
        @torch.enable_grad()
        def cost_fcn(x):
            return self.cost(torch.from_numpy(x).reshape(1,-1).to(self.device)).to("cpu").numpy()
        @torch.enable_grad()
        def jacobian_cost(x):
            jac= torch.autograd.functional.jacobian(self.cost,torch.from_numpy(x).reshape(1,-1).to(self.device)).reshape(-1)
            jac[self.sites_task] = 0
            return jac.cpu().numpy().reshape(-1)
        
        if bound ==True: # constrained optimization
            results = minimize(cost_fcn, x.cpu().numpy().reshape(-1), method=method,jac=jacobian_cost, tol=tol, bounds=self.scipy_bounds)
        else: # unconstrained optimization
            results = minimize(cost_fcn, x.cpu().numpy().reshape(-1), method=method,jac=jacobian_cost, tol=tol)
        return torch.from_numpy(results.x).view(1,-1).to(self.device), results


    def sample_tt(self, x_task=None, n_samples=500, deterministic=False, alpha=0.75):
        
        if x_task is None:
            n_discretization_task = None
        else:
            self.sites_task=np.arange(x_task.shape[-1])
            n_discretization_task = self.n[:x_task.shape[-1]]
        if not deterministic:
            samples = tt_utils.stochastic_top_k(tt_cores=self.tt_model.tt().cores[:], domain=self.domain, 
                         n_discretization_x=n_discretization_task , x=x_task, n_samples=n_samples, 
                         alpha=alpha, device=self.device)
        else:
            samples = tt_utils.deterministic_top_k(tt_cores=self.tt_model.tt().cores[:], domain=self.domain, 
                         n_discretization_x=n_discretization_task, x=x_task, n_samples=n_samples, 
                        device=self.device)
        return samples 

    def sample_random(self, n_samples, x_task=None):
        ''' sample from the uniform distribution from the domain '''
        samples = tt_utils.sample_random(batch_size=1, n_samples=n_samples, domain=self.domain, device=self.device)
        if x_task is not None:
            self.sites_task=np.arange(x_task.shape[-1])
            samples[0,:,:x_task.shape[-1]] = x_task
        return samples


    
    def canonicalize(self):
        ''' Canonicalize the tt-cores '''
        self.tt_model = tt_utils.tt_canonicalize(self.tt_model,site=0).to(self.device)
        

    def gradient_optimization(self,x, is_site_fixed, GN=True, lr=1e-2, n_step=10):
        '''
            Given a batch of initializations x, fine tune the solution
            is_site_fixed: a list or tensor. is_site_fixed[i]=1 if x[:,i] is fixed/constant (e.g. task variables and discrete vasiables)
            GN=True => Gauss Newton else gradient-descent/asecent with learning rate lr
            n_step: number of steps of gd or GM
        '''

        x_opt = tt_utils.gradient_optimization(x, fcn=self.pdf, is_site_fixed=is_site_fixed, 
                x_min=self.min_domain, x_max=self.max_domain,
                lr=lr, n_step=n_step, GN=GN, max_batch=10**4, device=self.device)
        return x_opt



