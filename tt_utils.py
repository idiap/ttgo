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


import torch
import tntorch as tnt
import numpy as np
import copy
torch.set_default_dtype(torch.float64)

import math 

def get_exponential_discretization(xmax=1.0,n=100,sc=1.0,flip=False,device='cpu'):
    '''
    note: applies only for symmetric bounds; (-xmax,xmax)
    Generates discretization non-uniformly: in an exponential manner
    
    sc --> inf => uniform discretization
    sc --> 0 => discretization points are more dense near 0.
    flip=True:
        discretization points are more dense near the boundary
    '''

    xmin = -1*xmax
    even_n = n%2
    n_p = int(n/2) + even_n
    xp = xmax*torch.linspace(0,1,n_p).to(device)
    if not flip:
        yp = -1+torch.exp(sc*xp.abs()/xmax)
    else:
        yp = 1-1/(1+torch.exp(sc*xp.abs()/xmax))
        yp = yp-yp.min()
    yp = xmax*yp/yp.max()
    idx_n = -1*torch.arange(n_p)[even_n:].to(device)
    xn = -1*xp[idx_n]
    yn = -1*yp[idx_n]
    y = torch.cat((yn,yp),dim=-1)
    return y

def idx2domain(I, domain, device): # for any discretization
    ''' Map the index of the tensor/discretization to the domain'''
    X = torch.zeros(I.shape).to(device)
    for i in range(I.shape[1]):
        X[:,i] =  domain[i][I[:, i]]
    return X

def domain2idx(x, domain, device, uniform=False):
    ''' 
    Map x from the domain to the index of the discretization (the nearest discretization point smaller than x)
    '''
    I = torch.zeros(x.shape).to(device)

    if uniform: # if the discretization is uniform
        for i in range(x.shape[-1]):
            min_i = domain[i][0] # 
            step_i = domain[i][1]-domain[i][0] 
            I[:,i] = torch.clip(((x[:,i] - min_i)/step_i).floor(), 0, len(domain[i])-2)
    else: 
        for i in range(x.shape[-1]):#  #
            I[:,i] = torch.clip(((x[:,i].view(-1,1)- domain[i])>=0).sum(dim=-1)-1, 0,  len(domain[i])-2)#torch.argmin(torch.abs(x[:,i].view(-1,1)- domain[i]), dim=1) 
            
    return I.long()

def get_elements_from_cores(tt_cores, idx):
    '''
    Given the tt_cores and a batch of index get the  elements
    '''
    mat_ = tt_cores[0][:,idx[:,0],:]
    for i in range(1,idx.shape[-1]):
        mat_ = torch.einsum('ijk,kjl->ijl',(mat_,tt_cores[i][:,idx[:,i]]))
    return mat_.view(-1)


def get_elements(tt_model, idx):
    '''
    Given the tt_model in tntorch format and a batch of index get the  elements
    '''
    return get_elements_from_cores(tt_model.tt().cores, idx)

def get_tt_sum(tt_model):
    '''
    Given the tt_model in tntorch format find the sum of the tensor
    '''
    return get_tt_sum_from_cores(tt_model.tt().cores)

def get_tt_sum_from_cores(tt_cores):
    '''
    Given the tt cores  find the sum of the tensor
    '''  
    sum_ = tt_cores[0].sum(dim=1)
    for core in tt_cores[1:]:
        sum_ = sum_@core.sum(dim=1)
    return sum_.item()    

def get_tt_mean(tt_model):
    '''
    Given the tt_model in tntorch format find the mean of the tt-model
    '''

    return get_tt_mean_from_cores(tt_model.tt().cores)


def get_tt_mean_from_cores(tt_cores):
    '''
        find the mean of the tt-model given its cores
    '''
    sum_ = tt_cores[0].sum(dim=1)/tt_cores[0].shape[1]
    for core in tt_cores[1:]:
        sum_ = sum_@core.sum(dim=1)/core.shape[1]
    return sum_.item()

def process_batch_tt_cores(batch_tt_cores, device='cpu'):
    '''
    Given a batch of tt-cores ( a list of tensors. size of list is same: d)
    batch_tt_cores[i]:  r_i x n_i x s_i
    return a reshaped batch of tt-cores called batch_rr_cores_new[i]: max(r_i) x n_i x max(s_i)
    Basically makes cores corresponding to i-th mode has same shape
    '''
    bs = len(batch_tt_cores)
    d = len(batch_tt_cores[0])
    batch_tt_cores_new = copy.deepcopy(batch_tt_cores)
    r = []
    for i in range(d):
        r_i = max([batch_tt_cores[j][i].shape[-1] for j in range(bs)])
        r.append(r_i)
    
    r = [1] + r  #[1,r_1,.., r_(d-1),1]
    
    for i in range(d):
        for j in range(bs):
            batch_tt_cores_new[j][i] = torch.zeros(r[i],batch_tt_cores[j][i].shape[1],r[i+1]).to(device)
            batch_tt_cores_new[j][i][:batch_tt_cores[j][i].shape[0],:,:batch_tt_cores[j][i].shape[-1]] = batch_tt_cores[j][i][:,:,:]
            
            
    tt_cores_new = []
    for i in range(d):
        cores = [batch_tt_cores_new[j][i][None,:,:,:] for j in range(bs)]
        tt_cores_new.append(torch.cat(cores,dim=0))
        
    
    return tt_cores_new

def get_value_batch_tt_cores(tt_cores, x,  domain, 
                                    n_discretization, max_batch=10**5,
                                     device="cpu", unform_discretization=True):
    '''
    Only linear interpolation. TO DO: Extend to spline interpolation
    n_tt: number of tensor trains (each having the same shape)
    x: n_tt x batch_size x dim
    tt_cores: a list of tt-cores. each core is of shape: n_tt x r_ x n_ x r
    '''
    idx_1 = domain2idx(x.view(-1,x.shape[-1]), domain=domain, device=device, uniform=unform_discretization) # find the closest/floor index of the state (w.r.t to the discretizaton)
    x_1 = idx2domain(idx_1, domain, device=device) # nearest discretization smaller than x 
    dx = (x.view(-1,x.shape[-1])-x_1)
    idx_2 = torch.clip(idx_1+torch.sign(dx),
                                n_discretization[:x.shape[-1]]*0,
                                n_discretization[:x.shape[-1]]-1).long() # next index
    x_2 = idx2domain(idx_2, domain, device=device)
    dx = dx.abs()/(1e-6+(x_2-x_1).abs())

    idx_1 = idx_1.view(*x.shape)
    idx_2 = idx_2.view(*x.shape)
    dx = dx.view(*x.shape)

    id0 = torch.arange(x.shape[0]).unsqueeze(1)
    core_1 = tt_cores[0].permute([0,2,1,3])[id0,idx_1[:,:,0]]
    core_2 = tt_cores[0].permute([0,2,1,3])[id0,idx_2[:,:,0]]
    mat_ = core_1 + (dx[:,:,0][:,:,None,None])*(core_2-core_1)
    for i in range(1,idx_1.shape[-1]):
        core_1 = tt_cores[i].permute([0,2,1,3])[id0,idx_1[:,:,i]]
        core_2 = tt_cores[i].permute([0,2,1,3])[id0,idx_2[:,:,i]]
        mat = core_1 + (dx[:,:,i][:,:,None,None])*(core_2-core_1)
        mat_ = torch.einsum('bjik,bjkl->bjil',mat_,mat)

    return mat_.view(x.shape[0],-1)




def get_value_batch_tt_cores_v2(tt_cores, x,  domain, 
                                    n_discretization=None, max_batch=10**5,
                                     device="cpu", unform_discretization=True):
    '''
    Only linear interpolation. TO DO: Extend to spline interpolation
    same as get_value_batch_tt_cores() but with batch truncated (to overcome momory shortage)
    n_tt: number of tensor trains (each having the same shape)
    x: n_tt x batch_size x dim
    tt_cores: a list of tt-cores. each core is of shape: n_tt x r_ x n_ x r
    Given 
    
    '''
    
    if n_discretization is None:
        n_discretization = torch.tensor([len(dom) for dom in domain]).to(device)
    

    def fcn(x):
        idx_1 = domain2idx(x.view(-1,x.shape[-1]), domain=domain, device=device, uniform=unform_discretization) # find the closest/floor index of the state (w.r.t to the discretizaton)
        x_1 = idx2domain(idx_1, domain, device=device) 
        dx = (x.view(-1,x.shape[-1])-x_1)
        idx_2 = torch.clip(idx_1+torch.sign(dx),
                                    n_discretization[:x.shape[-1]]*0,
                                    n_discretization[:x.shape[-1]]-1).long() # next index
        x_2 = idx2domain(idx_2, domain, device=device)
        dx = dx.abs()/(1e-6+(x_2-x_1).abs())
        
        idx_1 = idx_1.view(*x.shape)
        idx_2 = idx_2.view(*x.shape)
        dx = dx.view(*x.shape)
        id0 = torch.arange(x.shape[0]).unsqueeze(1)
        core_1 = tt_cores[0].permute([0,2,1,3])[id0 ,idx_1[:,:,0]]
        core_2 = tt_cores[0].permute([0,2,1,3])[id0 ,idx_2[:,:,0]]
        mat_ = core_1 + (dx[:,:,0][:,:,None,None])*(core_2-core_1)
        for i in range(1,idx_1.shape[-1]):
            core_1 = tt_cores[i].permute([0,2,1,3])[id0 ,idx_1[:,:,i]]
            core_2 = tt_cores[i].permute([0,2,1,3])[id0 ,idx_2[:,:,i]]
            mat = core_1 + (dx[:,:,i][:,:,None,None])*(core_2-core_1)
            mat_ = torch.einsum('bjik,bjkl->bjil',mat_,mat)
        return mat_.view(x.shape[0],-1)
    
    def fcn_batched(x):
        n_tt = x.shape[0]
        batch_size = x.shape[1]
        fcn_values = torch.empty(n_tt, batch_size).to(device)
        num_batch = batch_size//max_batch
        end_idx = 0
        for i in range(num_batch):
            start_idx = i*max_batch
            end_idx = (i+1)*max_batch
            fcn_values[:, start_idx:end_idx] = fcn(x[:,start_idx:end_idx])
        if batch_size>end_idx:          
            fcn_values[:,end_idx:batch_size] = fcn(x[:,end_idx:batch_size])
        return fcn_values
        
    return fcn_batched(x)


def get_value(tt_model, x,  domain, 
                    n_discretization=None,spline_type='linear', 
                    max_batch=10**5, uniform=False, device="cpu"):
    ''' 
    Evaluate the tt-model (in tntorch format) at the given state with Linear interpolation between the nodes. Assumes uniform discretization 
    dh_domain : a 1D tensor containing the step size of discretization for each site/mode
    n_discretization: a 1D tensor continginingthe number of discretization points along each mode
    spline_interpolation: if true use cardinal spline interpolation, otherwise linear interpolation
    s: tension value used for cardinal spline interpolation (highly-recommended to keep it to 0.5, results in catmul-rom spline)
    '''
    if n_discretization is None:
        n_discretization = torch.tensor([len(dom) for dom in domain]).to(device)
    return get_value_from_cores(tt_model.tt().cores, x,  domain, 
                                    n_discretization,max_batch, 
                                    spline_type, uniform,
                                    device)

def get_value_from_cores(tt_cores, x,  domain, 
                            n_discretization=None, max_batch=10**5,
                            spline_type='linear', uniform=False,
                            device="cpu"):
    # if spline_interpolation: use cardinal spline. s determines the tension of the spline. s=0 => hermite spline, 
    # s=0.5 => catmul-rom spline (ideal and highly recommended)
    # if spline_interpolation is false, use linear interpolation
    if n_discretization is None:
        n_discretization = torch.tensor([len(dom) for dom in domain]).to(device)
    
    def fcn(x_batch):
        idx = domain2idx(x_batch, domain=domain, device=device, uniform=uniform) # find the closest/floor index of the state (w.r.t to the discretizaton)
        x_1 = idx2domain(idx, domain, device=device) # nearest discreti
        idx_n = idx + 1
        x_2 = idx2domain(idx_n, domain, device=device)
        t = (x_batch-x_1).abs()/((x_2-x_1).abs()) # x.shape, i.e. batch x dim
        if spline_type == 'linear':
            mat_ = torch.tensor([1]).to(device).view(1,1,1).expand(-1,idx.shape[0],-1)
            for i in range(idx.shape[-1]):
                mat = tt_cores[i][:,idx[:,i],:]+\
                    t[:,i].view(1,-1,1)*(tt_cores[i][:,idx_n[:,i],:]-tt_cores[i][:,idx[:,i],:])
                mat_ = torch.einsum('ijk,kjl->ijl',mat_,mat)
            return mat_.view(-1)
        elif spline_type == 'b':
            b0 = (1-3*t+3*t**2-t**3)/6.0; b1 = (4-6*t**2+3*t**3)/6.0
            b2 = (1+3*t+3*t**2-3*t**3)/6.0; b3 = (t**3)/6.0
        else: 
            # catmul spline interpolation (cardinal spline. s=0 => hermite, s=0.5 => catmul-rom)
            s = 0.5; # specific tension value chosen to make cardinal spline a catmul spline
            b0 = s*t*(-1+2*t-t**2); b1 = (1+(s-3)*t**2+(2-s)*t**3); 
            b2 = (s*t + (3-2*s)*t**2+(s-2)*t**3); b3 = (-s*t**2 + s*t**3);
            
        idx_0 = torch.clip(idx-1, 0*n_discretization, n_discretization-1)
        idx_1 = torch.clip(idx,   0*n_discretization, n_discretization-1)
        idx_2 = torch.clip(idx+1, 0*n_discretization, n_discretization-1)
        idx_3 = torch.clip(idx+2, 0*n_discretization, n_discretization-1)    
        mat_ =  torch.tensor([1]).to(device).view(1,1,1).expand(-1,idx.shape[0],-1)
        for i in range(idx.shape[-1]):
            mat = b0[:,i].view(1,-1,1)*tt_cores[i][:,idx_0[:,i],:] + \
                  b1[:,i].view(1,-1,1)*tt_cores[i][:,idx_1[:,i],:] + \
                  b2[:,i].view(1,-1,1)*tt_cores[i][:,idx_2[:,i],:] + \
                  b3[:,i].view(1,-1,1)*tt_cores[i][:,idx_3[:,i],:]
            mat_ = torch.einsum('ijk,kjl->ijl',mat_,mat)
        return mat_.view(-1)

    return fcn_batch_limited(fcn=fcn,max_batch=max_batch, device=device)(x)


def refine_approximation(tt_model, domain, site_list=[], scale_factor=1, 
                         spline_type='catmul',device='cpu'):
    # Refine the discretization and interpolate the model
    domain_new = refine_domain(domain=[x.to(device) for x in domain], 
                                    site_list=site_list,
                                    scale_factor=scale_factor, device=device)
    tt_model_new = refine_model(tt_model=tt_model.clone(),
                                        domain_old=domain,
                                        domain_new=domain_new, 
                                        site_list=site_list,
                                        spline_type=spline_type,
                                        device=device)

    return tt_model_new, domain_new


def get_grad(x, tt_cores, domain, n_discretization=None, grad_recipe=None,
                max_batch=10**5, spline_type='linear', uniform=False,
                device="cpu"):
    ''' 
        TODO: truncate x to min_x and max_x (pass this param to the function)
        Evaluate the gradient at the given points with spline/linear interpolation between the nodes
        input: x, batch_size x dim
        s: tension in the cardinal spline interplation between values (0.5 by default)
        output: grad_x,  batch_size x dim
    '''  
    if (grad_recipe is None) and (spline_type == 'linear'): # only for linear interpolation of gradients
        grad_recipe = get_grad_recipe(tt_cores, domain)
        
    def fcn(x):
        idx = domain2idx(x, domain=domain, uniform=uniform, device=device) 
        x_1 = idx2domain(idx, domain=domain, device=device) 
        dx = (x - x_1)
        idx_n = idx + 1 # next index
        x_2 = idx2domain(idx_n, domain=domain, device=device)
        delta_x = 1/(x_2-x_1).abs()
        t = torch.clip(dx.abs()/(x_2-x_1).abs(),0,1)
        if spline_type == 'linear': # linearly interpolate the gradients between nodes       
            mat = torch.ones(x.shape[-1],1,idx.shape[0],1).to(device)
            for i in range(x.shape[-1]):
                y1 = grad_recipe[i][:,:,idx[:,i],:]
                y2 = grad_recipe[i][:,:,idx_n[:,i],:]
                mat = torch.einsum('bijk,bkjl->bijl',mat,y1+t[:,i].view(1,1,-1,1)*(y2-y1))
            grad_x = mat.view(mat.shape[0],-1).permute(1,0)
            return grad_x

        elif spline_type == 'b': # gradient corresponding to the spline interpolation (catmul
            b_db = torch.zeros(2,4,x.shape[0],x.shape[-1]).to(device)
            b_db[0,0] = (1-3*t+3*t**2-t**3); b_db[1,0] = (-3+6*t-3*t**2)
            b_db[0,1] = (4-6*t**2+3*t**3);b_db[1,1] = (-12*t+9*t**2)
            b_db[0,2] = (1+3*t+3*t**2-3*t**3); b_db[1,2]= (3+6*t-9*t**2)
            b_db[0,3]= (t**3); b_db[1,3]=(3*t**2);
            b_db = b_db/6.0
        else:
            s = 0.5 ; # makes cardinal spline a catmul spline
            b_db = torch.zeros(2,4,x.shape[0],x.shape[-1]).to(device)
            b_db[0,0] = s*t*(-1+2*t-t**2); b_db[1,0] = s*(-1+4*t-3*t**2); 
            b_db[0,1] = (1+(s-3)*t**2+(2-s)*t**3);b_db[1,1] = (2*(s-3)*t + 3*(2-s)*t**2)
            b_db[0,2] = (s*t + (3-2*s)*t**2+(s-2)*t**3); b_db[1,2]= (s+ 2*(3-2*s)*t + 3*(s-2)*t**2);
            b_db[0,3]= (-s*t**2 + s*t**3); b_db[1,3]=(-2*s*t + 3*s*t**2);
            
        b_db[1,:] = b_db[1,:]*(delta_x[None,:,:]) 
        idx_0 = torch.clip(idx-1, 0*n_discretization, n_discretization-1)
        idx_1 = torch.clip(idx,   0*n_discretization, n_discretization-1)
        idx_2 = torch.clip(idx+1, 0*n_discretization, n_discretization-1)
        idx_3 = torch.clip(idx+2, 0*n_discretization, n_discretization-1)     
        mat = torch.ones(x.shape[-1],1,idx.shape[0],1).to(device)
        for i in range(x.shape[-1]):
            mat_01 = b_db[:,0,:,i][:,None,:,None]*tt_cores[i][None,:,idx_0[:,i],:] + \
                     b_db[:,1,:,i][:,None,:,None]*tt_cores[i][None,:,idx_1[:,i],:] + \
                     b_db[:,2,:,i][:,None,:,None]*tt_cores[i][None,:,idx_2[:,i],:] + \
                     b_db[:,3,:,i][:,None,:,None]*tt_cores[i][None,:,idx_3[:,i],:]
            mat_i = mat_01[0][None,:,:,:].repeat(x.shape[-1],1,1,1)
            mat_i[i] = 1*mat_01[1]#*(delta_x[:,i].view(1,-1,1))
            mat = torch.einsum('bijk,bkjl->bijl',mat,mat_i)
        grad_x = mat.view(mat.shape[0],-1).permute(1,0)
        return grad_x # batch_size x dim

    return fcn_batch_limited(fcn=fcn,max_batch=max_batch,num_out=x.shape[-1],device=device)(x)


def get_grad_recipe(tt_cores, domain):
    '''
        tt_cores: i-th core is of shape r_i x n_i x r_(i+1)

        for linear_interpolation of gradients
            Given tt_cores return the gradient of each core at each node 
            grad_recipe: i-the core is of shape: dim x r_i x (n_i) x r_(i+1) 
                with grad_recipe[i][i] being the gradient w.r.t 
                that site and  grad_recipe[i][j] = tt_core[i] if i!=j
    '''
    dim = len(tt_cores)
    cores = tt_cores[:]
    grad_recipe = [core[None,:,:,:].repeat(dim,1,1,1) for core in cores]
    for site in range(dim):
        core = cores[site]
        diff_x = (domain[site][1:]-domain[site][:-1]).view(1,-1,1)
        diff_core = (core[:,1:,:]-core[:,:-1,:])/diff_x
        diff_core = torch.concat((diff_core[:,0,:][:,None,:],
                                    diff_core, diff_core[:,-1,:][:,None,:]),dim=1)
        diff_x = torch.concat((diff_x[:,0,:].view(1,-1,1), diff_x, diff_x[:,-1,:].view(1,-1,1)), dim=1)
        w_1 = diff_x[:,1:,:]/(diff_x[:,1:,:]+diff_x[:,:-1,:])
        grad_recipe[site][site] = w_1*diff_core[:,1:,:]+(1.0-w_1)*diff_core[:,:-1,:]

    return grad_recipe



def get_value_discrete(tt_model, x, domain, device="cpu"): # no interpolation
    '''
        Evaluate tt-model at the given point (in batch) from the domain. Assuming uniform discretization
        Input: x, batch_size x dim
    '''
    idx_state = domain2idx(x, domain, device) # find the index (w.r.t to the discretizaton)
    return get_elements(tt_model,idx_state).view(-1) #v_model[idx_state].torch() # batch_size x 1


def extend_cores(tt_cores, site, n_cores, d,  device='cpu'):
        ''' 
        Given a list of tt_cores add n_cores of identity cores starting at the given site
        d is a list containing dimension of those cores modes (size: n_cores)
        '''
        site = min(site,len(tt_cores))
        if site==0:
            r = 1
            # base_cores_left = []
            # base_cores_right = tt_cores[:]

        elif (site == (len(tt_cores))) or (site==-1):
            r = 1
            site=len(tt_cores)
            # base_cores_left = tt_cores[:]
            # base_cores_right = []

        else:
            r = tt_cores[site-1].shape[-1]
            # base_cores_left = tt_cores[:site]
            # base_cores_right = tt_cores[site:]
        base_cores_left = tt_cores[:site]
        base_cores_right = tt_cores[site:]        
        id_action = torch.eye(r)[:,None,:].to(device)
        dummy_cores = [id_action.expand(-1, d[i],-1).to(device) for i in range(n_cores)]
        cores = base_cores_left + dummy_cores + base_cores_right
        return cores

def extend_model(tt_model, site, n_cores, d, device='cpu'):
        ''' 
        Given a list of tt_cores add n_cores of identity cores starting at the given site
        d is a list containing dimension of those cores modes (size: n_cores)
        '''
        tt_cores = tt_model.tt().cores[:]
        cores =  extend_cores(tt_cores, site, n_cores, d,  device)
        return tnt.Tensor(cores).to(device)


def refine_domain(domain, site_list=[], scale_factor=2, device='cpu'):
    '''
    Given a list of discretization, add more points into the domain corresponding to the 
    sites in site_list.
    If i-th domain has n discretization points and i is in site_list, then in the resulting
    tt, the number of points in i-th domain will increase to (scale_factor-1)*(n-1)  
    '''
    domain_new = domain[:]
    for site in site_list:
        n_site = len(domain[site])
        n_site_new = n_site + (scale_factor-1)*(n_site-1)
        domain_new[site] = torch.empty(n_site_new).to(device)
        x_1 = domain[site][1:] #  (n_site-1) 
        x_0 = domain[site][:-1] # (n_site-1) 
        d_x = x_1-x_0 #  (n_site-1) 
        idx = torch.arange(n_site).to(device)
        domain_new[site][idx*scale_factor] = domain[site].clone().to(device)
        for i in range(1,scale_factor):
            idx = torch.arange(i,n_site_new,scale_factor).to(device)
            domain_new[site][idx] = x_0 + (i)*d_x/scale_factor                
    return [x.to(device) for x in domain_new]    

def refine_cores(tt_cores, domain_old, domain_new, site_list=[], 
                spline_type='linear',
                device='cpu'): # under construction
    '''
    Given a list of tt-cores, add more slices into the cores corresponding to the 
    sites in site_list corresponding to the discretization given by domain_new.
    The new slices are added by interpolating between the original domain (domain_old)
    domain_old and domain_new are both lists containing discretization old/new for each site of tt_core
    spline_type: 'linear' otherwise 'catmul spline'
    '''
    tt_cores_new = copy.deepcopy(tt_cores)
    # interpolate between the adjacent slices 
    for site in site_list:
        x = domain_new[site].view(-1,1)
        idx = domain2idx(x,[domain_old[site]],device)
        x_1 = idx2domain(idx,[domain_old[site]],device) 
        idx_n = idx+1
        x_2 = idx2domain(idx_n,[domain_old[site]],device) 
        t = ((x - x_1).abs()/(1e-9+(x_2-x_1).abs())).view(1,-1,1)
        if spline_type == 'linear':
            tt_cores_new[site] = tt_cores[site][:,idx.view(-1),:] + t*(tt_cores[site][:,idx_n.view(-1),:]-tt_cores[site][:,idx.view(-1),:])
            
        else:
            # catmul spline interpolation (cardinal spline. s=0 => hermite, s=0.5 => catmul-rom)
            s = 0.5; # specific tension value chosen to make cardinal spline a catmul spline
            b0 = s*t*(-1+2*t-t**2); b1 = (1+(s-3)*t**2+(2-s)*t**3); 
            b2 = (s*t + (3-2*s)*t**2+(s-2)*t**3); b3 = (-s*t**2 + s*t**3);
            idx_0 = torch.clip(idx-1, 0*len(domain_old[site]), len(domain_old[site])-1).view(-1)
            idx_1 = torch.clip(idx,   0*len(domain_old[site]), len(domain_old[site])-1).view(-1)
            idx_2 = torch.clip(idx+1, 0*len(domain_old[site]), len(domain_old[site])-1).view(-1)
            idx_3 = torch.clip(idx+2, 0*len(domain_old[site]), len(domain_old[site])-1).view(-1)
            tt_cores_new[site] = b0*tt_cores[site][:,idx_0,:] + \
                b1*tt_cores[site][:,idx_1,:] + \
                b2*tt_cores[site][:,idx_2,:] + \
                b3*tt_cores[site][:,idx_3,:]    
    tt_model_new = tnt.Tensor(tt_cores_new)
    tt_model_new = tt_canonicalize(tt_model_new).to(device)
    return tt_model_new.tt().cores[:]



# def refine_cores_old(tt_cores, site_list=[], scale_factor=2, device='cpu'):
#     '''
#     Given a list of tt-cores, add more slices into the cores corresponding to the 
#     sites in site_list.
#     If i-th cire has n slices and i is in site_list, then in the resulting
#     tt, the number of slices in i-th core will increase to (scale_factor-1)*(n-1)
#     '''
#     tt_cores_new = tt_cores[:]
#     for site in site_list:
#         n_site = tt_cores[site].shape[1]
#         n_site_new = n_site + (scale_factor-1)*(n_site-1)
#         tt_cores_new[site] = torch.empty(tt_cores[site].shape[0],
#                                          n_site_new,tt_cores[site].shape[-1]).to(device)
#         core_1 = tt_cores[site][:,1:,:] # -1 x (n_site-1) x -1
#         core_0 = tt_cores[site][:,:-1,:] # -1 x (n_site-1) x -1
#         d_core = core_1-core_0 # -1 x (n_site-1) x -1
#         idx = torch.arange(n_site)
#         tt_cores_new[site][:,idx*scale_factor,:] = tt_cores[site].clone()
#         for i in range(1,scale_factor):
#             idx = torch.arange(i,n_site_new,scale_factor)
#             tt_cores_new[site][:,idx,:] = core_0 + (i)*d_core/scale_factor                
#     return tt_cores_new

def refine_model(tt_model, domain_old, domain_new, site_list=[], 
                 spline_type='catmul', device='cpu'):
    '''
    Given a tt-model, add more slices into the cores corresponding to the 
    sites in site_list.
    If i-th cire has n slices and i is in site_list, then in the resulting
    tt, the number of slices in i-th core will increase to (scale_factor-1)*(n-1)
    '''
    tt_cores_new = refine_cores(tt_cores=tt_model.tt().cores[:], domain_old=domain_old, 
                                domain_new=domain_new,
                                site_list=site_list, 
                                spline_type=spline_type,
                                device=device)
    tt_model_new = tnt.Tensor(tt_cores_new).to(device)
    tt_model_new.round(1e-9)
    tt_model_new = tt_canonicalize(tt_model_new)
    return tt_model_new.to(device)




def cross_approximate(fcn,  domain,  max_batch=10**5,
                        rmax=200, nswp=20, eps=1e-4, verbose=False, 
                        kickrank=3, val_size=1e4,device="cpu"):
    ''' 
        TT-Cross Approximation using tntorch's implementation
        eps: accuracy of approximation
    '''
    tt_model = tnt.cross(fcn_batch_limited(fcn, max_batch=max_batch, device=device),
        domain=domain,
        max_iter=nswp, eps=eps, rmax=rmax, kickrank=kickrank, 
        function_arg='matrix',device=device,_minimize=False,
        val_size=val_size, verbose=verbose)
    # tt_model.to('cpu').round_tt(eps*1e-2)
    tt_model = tt_canonicalize(tt_model)
    return tt_model.to(device)


def fcn_batch_limited(fcn, max_batch=10**5, num_out=1, device="cpu"):
    ''' 
    To avoid memorry issues with large batch processing, 
    reduce computation into smaller batches 
    '''   
    def fcn_batch_truncated(x):
        batch_size = x.shape[0]
        fcn_values = torch.empty(batch_size,num_out).to(device)
        num_batch = batch_size//max_batch
        end_idx = 0
        for i in range(num_batch):
            start_idx = i*max_batch
            end_idx = (i+1)*max_batch
            fcn_values[start_idx:end_idx] = fcn(x[start_idx:end_idx].view(-1,x.shape[1])).view(-1,num_out)
        if batch_size>end_idx:          
            fcn_values[end_idx:batch_size] = fcn(x[end_idx:batch_size].view(-1,x.shape[1])).view(-1,num_out)
        if num_out==1:
            return fcn_values.view(-1)
        else:
            return fcn_values

    return fcn_batch_truncated


def sample_random(batch_size, n_samples, domain, device="cpu"):
    ''' sample from the uniform distribution from the domain '''
    samples = torch.empty((batch_size,n_samples,len(domain))).to(device)
    for i in range(len(domain)):
        samples[:,:,i] = domain[i][0] + (domain[i][-1]-domain[i][0])*torch.rand(size=(batch_size,n_samples)).to(device)
    return samples


def stochastic_choice(M, alpha=0.99, rand_state=None, device="cpu"):
    '''
        Given pmf get the prioritized samples
        M: batch_size x n_samples x n  
        Treat each row of a matrix M[:,i,:] as a PMF and select a column per row according to it
    '''
    
    #filtering low pmf samples
    if rand_state is not None:
        torch.random.manual_seed(torch.randn(1).data)
    M= torch.abs(M) # batch_size x n_samples x n_site
    M_max, _ = torch.max(M,dim=-1) # batch_size x n_samples
    M_min, _ = torch.min(M,dim=-1)
    M_mean = M.mean(dim=-1)
    
    # M_threshold = M_mean + alpha*(M_max-M_mean)
    
    M_max  = M_max[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    # M_min  = M_min[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    # M_mean  = M_mean[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    # M_threshold  = M_threshold[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    
    # M = M*(M>M_threshold)        
    M= M/(1e-9+M_max) # batch_size x n_samples x n_site

    M=M**(1/(1e-9+1-alpha))  # higher density is given higher importance
    M=M+1e-9
    M = M/(torch.sum(M, dim=-1)[:,:, None]) + 1e-9  # Normalize the pdf, batch_size x n_samples x n_site
    samples = torch.multinomial(M.view(-1,M.shape[-1]),1).view(M.shape[0],-1) # (batch_size*n_samples) x 1        
    if rand_state is not None:
        torch.random.set_rng_state(rand_state)
    return samples # batch_size x n_samples


def deterministic_choice(M,n_samples,idx_site, device="cpu"):
    """
        M: batch_size x n_samples x n_site

    """
    idx_site[:,1:] = (idx_site[:,1:]-idx_site[:,:-1]).abs()>0 
    idx_site[:,0] = 1
    
    bs = M.shape[0]
    n_site = M.shape[2]
    M = M*idx_site[:,:,None].expand(-1,-1,n_site) # make pmf corresponding to repeated indices to be zeo
    next_site = torch.zeros(bs,n_samples).to(device)
    previous_sample_id = torch.zeros(bs,n_samples).to(device)
    M2d = M.view(bs,-1) # bs x (n_samples*n_site)
    idx_k = torch.topk(M2d, k=n_samples, dim=-1)[1] # bs x n_samples
    next_site[:,:n_samples] = (idx_k).fmod(n_site) # which site next, bs x n_samples
    previous_sample_id[:,:n_samples] = (idx_k/n_site).long() # previous sample_id, b_size x n_samples
    return next_site.long(), previous_sample_id.long()


def contract_site(tt_model, site_x, p_x, device, eps=1e-6):
    '''
    Contract the cores of the tt-model given the weights for each discretization point 
    corresponding to each of the contracted site.
    p_x: a list of 1D tensor (probaility of each index of the site) 
    Return a contracted model 
    '''

    tt_cores = [core for core in tt_model.tt().cores[:]] # r_k x n_k x r_kn
    mat =  (tt_cores[site_x[0]](p_x[1+i].view(1,-1,1))).sum(dim=1) 
    for i,site in enumerate(site_x[1:]):
        mat_i = (tt_cores[site]*(p_x[1+i].view(1,-1,1))).sum(dim=1)
        mat = mat@mat_i
    tt_cores_c = tt_cores[:site_x[0]] + tt_cores[site_x[-1]:]
    if site_x[-1] < len(tt_cores):
        tt_cores_c[site_x[-1]] = torch.einsum('ij,jkl->ikl',mat,tt_cores_c[site_x[-1]])
    else:
        tt_cores_c[-1] = torch.einsum('ikj,jl->ikl',tt_cores_c[-1],mat)
    return tnt.Tensor(tt_cores_c).round_tt(eps=eps).to(device)



def condition_site(tt_cores, x, domain_x, n_discretization_x, spline_type, device):
    '''
    Condition (or slicing) the cores of the tt-model given the values corresponding to a site. 
    Assumes x: batch_size x dim_x correspond to the first few cores
    Return the conditioned model:  tt_cores of shape batch_size x r_i x n_i x r_i' 
    '''
    batch_size = x.shape[0]
    dim_x = x.shape[1]
    # interpolate to find the corresponding slice for x
    idx = domain2idx(x,domain_x,device).view(batch_size,-1) # batch_size x dim_state
    x_1 = idx2domain(idx,domain_x,device) 
    dx = (x - x_1)
    idx_n = idx+1#torch.clip(idx_x+torch.sign(dx),n_discretization_x*0,n_discretization_x-1).long() # next index (w.r.t disctretization)
    x_2 = idx2domain(idx_n,domain_x,device) 
    t = torch.abs(dx)*1.0/(1e-9+(x_2-x_1).abs())
    # interpolate between the adjacent slices 
    if spline_type == 'linear':
        for site in range(x.shape[-1]):
            tt_cores[site] = (tt_cores[site][:,idx[:,site],:]+t[:,site].view(1,-1,1)*(tt_cores[site][:,idx_n[:,site],:]-tt_cores[site][:,idx[:,site],:]))
    else:
         # catmul spline interpolation (cardinal spline. s=0 => hermite, s=0.5 => catmul-rom)
        s = 0.5; # specific tension value chosen to make cardinal spline a catmul spline
        b0 = s*t*(-1+2*t-t**2); b1 = (1+(s-3)*t**2+(2-s)*t**3); 
        b2 = (s*t + (3-2*s)*t**2+(s-2)*t**3); b3 = (-s*t**2 + s*t**3);
        idx_0 = torch.clip(idx-1, 0*n_discretization_x, n_discretization_x-1)
        idx_1 = torch.clip(idx,   0*n_discretization_x, n_discretization_x-1)
        idx_2 = torch.clip(idx+1, 0*n_discretization_x, n_discretization_x-1)
        idx_3 = torch.clip(idx+2, 0*n_discretization_x, n_discretization_x-1)    
        for i in range(x.shape[-1]):
            tt_cores[site] = b0[:,i].view(1,-1,1)*tt_cores[i][:,idx_0[:,i],:] + \
                  b1[:,i].view(1,-1,1)*tt_cores[i][:,idx_1[:,i],:] + \
                  b2[:,i].view(1,-1,1)*tt_cores[i][:,idx_2[:,i],:] + \
                  b3[:,i].view(1,-1,1)*tt_cores[i][:,idx_3[:,i],:]
    # tranform cores so that it is: batch_size x r_k x -1 x r_kn     
    tt_cores_ext = [tt_cores[site][None,:,:,:].permute(2,1,0,3) for site in range(dim_x)]+[tt_cores[site][None,:,:,:].expand(batch_size,-1,-1,-1) for site in range(dim_x,len(tt_cores))]
    # Merge the slices corresponding to x into one core of size: b_state x 1 x 1 x r and then merge it to the non-sliced core b_state x 1 x n_a x r_a
    core_state = tt_cores_ext[0]
    for site in range(1,dim_x):
        core_state = torch.einsum('bijk,bkjl->bijl',core_state,tt_cores_ext[site])
    tt_cores_ext[dim_x] = torch.einsum('bi,ijk->bjk',core_state[:,0,0,:],tt_cores[dim_x])[:,None,:,:] # b_state x 1 x n_1 x r
    tt_cores_ext = tt_cores_ext[dim_x:]
    return tt_cores_ext # each core is of shape barch_size x r_ x n_ x r and the number of cores is len(tt_cores)-dim_x

def get_rights(tt_cores_ext, device):
    batch_size = tt_cores_ext[0].shape[0]
    # batch_size x r_k x r_kn
    tt_cores_action_summed =[torch.sum(core,dim=2) for core in tt_cores_ext] # batch_size x r_k x r_kn 
    rights = [torch.ones(batch_size,1).to(device).view(-1,1)] # each element is batch_size x r_k
    for site, summed_core in enumerate(tt_cores_action_summed[::-1]):
        r_ = torch.einsum('ijk,ik->ij',summed_core, rights[-1])
        rights.append(r_) # batch_size x r_k : batch_size x (r_k x r_kn) times (batch_size x r_kn)
    rights = rights[::-1] # batch_size x r_k
    return rights


def stochastic_top_k(tt_cores, domain, spline_type='linear',
                         n_discretization_x=None, x=None, n_samples = 1, 
                         alpha=0.9, device="cpu"):
    '''
    Consider x to be continuous (linear interpolation between tt-nodes)
    state: batch_size x dim_state
    Generate n_samples points from Q-function (treated as a joint PDF distribution ) 
    '''
    dim = len(tt_cores)
    
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[1]], spline_type=spline_type,
                            n_discretization_x=n_discretization_x, 
                            device=device)

    rights = get_rights(tt_cores_ext,device=device)

    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #
    lefts = torch.ones([batch_size, n_samples, 1]).to(device) # batch_size x n_samples x 1
    for site in range(len(tt_cores_ext)):
        fiber = torch.einsum('ijkl,il->ijk', (tt_cores_ext[site], rights[site+1])) # batch_size x r_k x n_k
        pmf = torch.einsum('ijk,ikl->ijl', (lefts, fiber)) # batch_size x n_samples x n_site 
        samples_idx[:,:, site] = stochastic_choice(M=pmf, alpha=alpha, rand_state=None, device=device ) # batch_size x n_samples
        core_sliced = (tt_cores_ext[site].permute([0,2,1,3])[torch.arange(tt_cores_ext[site].shape[0]).unsqueeze(1),samples_idx[:,:, site]]).permute([0,2,1,3])
        lefts = torch.einsum('ijk,ikjl->ijl', (lefts, core_sliced))

        
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples
    return samples_concat


def deterministic_top_k(tt_cores, domain=[], spline_type='linear',
                x=None, n_samples=100, 
                n_discretization_x=None, 
                device="cpu"):
    '''
    Consider the states to be continuous (linear interpolation between tt-nodes)
    x: batch_size x dim_x (task variables)
    Generate n_samples points from tt-model (treated as a joint PDF distribution ) corresponding to top-k max values
    The tt_cores are not assumed to be right orthogonalized (orthogonal model).
    If not, call canonlicalize(tt_model) prior to calling this method 
    This will speed up the process 
    '''
    dim = len(tt_cores)
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, spline_type=spline_type,
                            domain_x=domain[:x.shape[-1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)


    # rights = get_rights(tt_cores_ext, device=device)
    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #

    # pmf:  batch_size x 1 x n
    pmf = torch.linalg.norm(tt_cores_ext[0],dim=-1) # tt_cores_ext[0]: batch_size X 1 X n X r_1
    # pmf = torch.einsum('ijkr,ir->ijk',tt_cores_ext[0],rights[1]).abs()

    n_site_0 = tt_cores_ext[0].shape[-2]
    # samples_site:  batch_size x min(n_samples,n_site) 
    idx_k = torch.topk(pmf.view(batch_size,-1),k=min(n_samples,n_site_0),dim=-1)[1].fmod(n_site_0).long()
    if n_site_0 < n_samples: 
        samples_idx[:,:,0] = idx_k.repeat(1,int(n_samples/n_site_0)+1)[:,:n_samples] #batch_size x n_samples
    else:
        samples_idx[:,:,0] = idx_k
    # p_cum: batch_size x n_samples x r_1
    p_cum = (tt_cores_ext[0].permute([0,2,1,3])[torch.arange(batch_size).unsqueeze(1),idx_k]).permute([0,2,1,3])[:,0,:,:]

    for site in range(1,len(tt_cores_ext)):
        n_sites = tt_cores_ext[site].shape[-2]

        pmf_pre = torch.einsum('ijk,iklm->ijlm', (p_cum, tt_cores_ext[site])).flatten(1,2)#.view(batch_size,-1,tt_cores_ext[site].shape[-1]) # batch x n_site*n_samples x r_site
        pmf = torch.linalg.norm(pmf_pre,dim=-1) # batch x (n_site*n_samples) 
        # pmf = torch.einsum('ijr,ir->ij',pmf_pre,rights[site+1]).abs()
        idx_k = torch.topk(pmf, k=n_samples, dim=-1)[1].long() # bs x n_samples
        
        samples_idx[:,:,site] = idx_k.fmod(n_sites).long()#((idx_k)/n_samples).floor().long()#( # top-k indices from the site, bs x n_samples
        samples_prev_id  = (idx_k/n_sites).long()#(idx_k).fmod(n_samples).long()#idx_k - samples_idx[:,:,site]*n_sites # ((idx_k-1)/n_sites).long() # update previous site index
        samples_idx[:,:,:site]= samples_idx[:,:,:site][torch.arange(batch_size).unsqueeze(1),samples_prev_id]
        # p_cum: batch_size x  n_samples  x r_site 
        p_cum = pmf_pre[torch.arange(batch_size).unsqueeze(1),idx_k]

    
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples

    return samples_concat


def get_tt_max(tt_model, domain, n_samples=100, deterministic=True, alpha=0.9, device="cpu"):
    '''
    Note: max is w.r.t the absolute value
    find the pseudo-max and argmax of a tt-model (absolute max) in a stochastic way
    '''
    tt_model_o =  tt_canonicalize(tt_model)
    tt_cores = tt_model_o.tt().cores[:]
    # Warm-up for mass sampling
    if deterministic:
        samples = deterministic_top_k(tt_cores=tt_cores, 
                        n_samples=n_samples, 
                        domain=domain, 
                        device=device)
    else:
        samples = stochastic_top_k(tt_cores=tt_cores, 
                        n_samples=n_samples, alpha=alpha,
                        domain=domain, device=device)
    samples_idx = domain2idx(samples.flatten(0,1),domain,device)
    values = get_elements(tt_model_o,samples_idx)
    idx = torch.argmax(torch.abs(values)) # batch_size 
    best_value = values[idx]

    return best_value, samples_idx[idx].view(-1) # max, argmax



def get_tt_bounds(tt_model,domain,device="cpu"):
    tt_model_1 = tt_model.clone()
    bound_1, idx_1 = get_tt_max(tt_model, domain, device=device)
    bound_1  =  get_elements(tt_model,idx_1.view(1,-1)).item()
    tt_model_2 = tt_model_1-bound_1
    tt_model_2.round_tt(eps=1e-9)
    bound_2, idx_2 = get_tt_max(tt_model_2.to(device),domain, device=device)
    bound_2  = get_elements(tt_model,idx_2.view(1,-1)).item()
    upper_bound = bound_1 if (bound_1>bound_2) else bound_2
    lower_bound = bound_1 if (bound_1<bound_2) else bound_2
    return (lower_bound,upper_bound)


def normalize_tt(tt_model, domain, lb=1., ub=100., 
                    auto_bound=True,canonicalize=True,
                    device="cpu"):
    lower_bound, upper_bound  = get_tt_bounds(tt_model, domain, device=device)
    if auto_bound:
        lb = 1 + upper_bound - lower_bound
        tt_model_out = lb + (tt_model.to("cpu")-lower_bound)
    else:
        tt_model_out = lb + (tt_model.to("cpu")-lower_bound)*((ub-lb)/(upper_bound-lower_bound))
    if canonicalize:
        tt_model_out = tt_canonicalize(tt_model_out)
    else:
        tt_model_out.round_tt(eps=1e-9) # not necessary
    return tt_model_out.to(device)

def tt_canonicalize(tt_model,site=0):
    ''' 
    Return an  orthogonalized tt-model at site. 
    For i>site, torch.einsum('ijk,ljk->il',Core[i],Core[i]) will be identity matrix
    '''
    tt_model_o = tt_model.clone()
    tt_model_o.orthogonalize(site)
    return tt_model_o

def gradient_optimization(x, fcn, is_site_fixed, x_min=-torch.inf, x_max=torch.inf,
                lr=1e-2, n_step=10, GN=True, max_batch=10**4, device='cpu'):
    '''
        fcn: function to be optimized/maximized (scalar output)
        x: initial guess: batch_size x dim
        x_min, x_max: bounds on x
        lr: learning rate for gradient ascent
        is_site_fixed: tensor of sixe 1 x dim. is_site_fixed[i]=0 if the site i is a variable
                        otherwise it is 1
        GN=True: Gauss-Newton type optimization, else gradient ascent 
    '''
    def sol_batch(x):
        for i in range(n_step):
            idx = torch.arange(x.shape[0]).to(device)
            jac = torch.autograd.functional.jacobian(fcn,x)[idx,idx]
            if GN:
                H = torch.einsum('ij,ik->ijk', jac,jac) 
                invH = torch.linalg.pinv(H)
                dx = -torch.einsum('ijk, ik -> ij', invH, jac)*fcn(x).view(-1,1)
            else:
                dx = lr*jac*(1-is_site_fixed.view(1,-1))
            x += dx
            x = torch.clip(x, x_min, x_max)
        return x
    return fcn_batch_limited(fcn=sol_batch, max_batch=max_batch,num_out=x.shape[-1],device=device)(x)