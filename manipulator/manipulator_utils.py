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
from roma import rotmat_to_unitquat as tfm
from utils import test_ttgo

def dist_orientation(Rd_0,v_d,Ra_0):
    '''
    Cost on orientation (flexible orientation)
    Rd_0: a 3x3 rotation matrix corresponding to the desired orientation (w.r.t world frame)
    v_d: 1x3 vector w.r.t. Rd frame  w.r.t. which rotation is allowed
    Ra_0: ..x3x3 batch of rotation matrices w.r.t world frame
    returns distance in range (0,1) 
    '''
    v_d = (v_d/torch.linalg.norm(v_d)).view(-1) # normalize the axis vector
    Rd_0 = Rd_0.view(3,3)
    Ra_d = torch.matmul(Ra_0,Rd_0.T) # Ra w.r.t. Rd frame    
 
    qa_d = tfm(Ra_d) # corresponding quarternion (imaginary_vector,real)
    va_d = qa_d[:,:-1]
    va_d = va_d/(torch.linalg.vector_norm(va_d,dim=1).view(-1,1)+1e-9) # axis vector w.r.t Rd frame to get Ra_d 

    d_orient = 1-torch.einsum('ij,j->i',va_d,v_d)**2
    return d_orient 

def dist_orientation_fixed(Rd_0, Ra_0,device='cpu'):
    '''
    distance between two orientations: Rd_0 (fixed desired orientation), Ra_0 (actual orientation)
    '''
    Rd_0 = Rd_0.view(3,3)
    qa_d = tfm(torch.matmul(Ra_0,Rd_0.T))
    q0 = torch.tensor([0.,0.,0.,1.]).to(device)
    dist_orient = 1-torch.einsum('ij,j->i',qa_d,q0)**2
    
    return dist_orient 



def exp_space(xmin=-1,xmax=1.,d=100):
    '''
        discretization of an interval with exponential sepration from the center
    '''
    d1 = int(d/2)
    d2 = d - d1
    xmid = 0.5*(xmin+xmax)
    t1 = np.logspace(0., 1, d1);
    t2 = np.logspace(0., 1, d2);
    t1 = t1 - t1[0]; t1 = t1/t1[-1]
    t2 = t2 - t2[0] +t1[1];t2 = t2/t2[-1]; 
    t1 = xmid + (xmax-xmid)*t1
    t2 = xmid + (xmin-xmid)*np.flip(t2)
    t = np.concatenate((t2,t1))
    return torch.from_numpy(t)

def get_latex_str(results, alphas):
    '''
    Used for generating tables for latex
    '''
    latex_str_tt = []
    for i in range(results.shape[1]-1): # over aphas
        latex_str_tt.append("& "+ "$" +str(round(alphas[i],2)) + "$")
        for j in range(results.shape[0]): # over sample_set
            for item_ in results[j,i,:]:
                latex_str_tt.append( " & " + "$" + str(round(item_.item(),2)) + "$")
        latex_str_tt.append(" \\\ ")     
    latex_str_rand = []
    latex_str_rand.append("&-")
    for i in range(results.shape[0]):
        for item_ in results[i,-1,:]:
            latex_str_rand.append( " & " + "$" +str(round(item_.item(),2))+ "$")
    latex_str_rand.append(" \\\ ")
    
    latex_tt = ""
    latex_rand = ""
    for str_ in latex_str_tt:
        latex_tt+=str_
    for str_ in latex_str_rand:
        latex_rand+=str_
    return latex_tt, latex_rand


def test_robotics_task(ttgo, cost_all, test_task, alphas, sample_set, cut_total=0.25,device='cpu'):
    # for latex
    norm = 1 
    results_union = torch.empty((len(sample_set),len(alphas)+1,3)).to(device) #n_samples x (alpha,rand)  x (raw,opt,sucess)
    results_intersection = torch.empty((len(sample_set),len(alphas)+1,3)).to(device) #n_samples x (alpha,rand) x  (raw,opt,sucess)
    for i, n_samples in enumerate(sample_set):
        results_rand_union = torch.empty(len(alphas),3).to(device)
        results_rand_intersection = torch.empty(len(alphas),3).to(device)
        for j, alpha in enumerate(alphas):
            costs_tt,costs_tt_opt,costs_rand,costs_rand_opt,tt_nit,rand_nit = test_ttgo(ttgo=ttgo.clone(), cost=cost_all, 
                            test_task=test_task, n_samples_tt=n_samples,
                            alpha=alpha, norm=norm, device=device, test_rand=True, cut_total=cut_total)
            n_test = costs_tt.shape[0]
            idx_tt = costs_tt_opt[:,0]<cut_total
            idx_rand = costs_rand_opt[:,0]<cut_total 
            
            # union
            idx = torch.logical_or(idx_tt,idx_rand) 
            tt_success = (100*torch.sum(idx_tt).item()/n_test); rand_success = (100*torch.sum(idx_rand).item()/n_test)
            tt_nit_mean = torch.mean(tt_nit[idx]).item(); rand_nit_mean=torch.mean(rand_nit[idx]).item()
            costs_tt_mean = torch.mean(costs_tt[idx,0],dim=0).item()
            costs_tt_opt_mean = torch.mean(costs_tt_opt[idx,0]).item()
            costs_rand_mean = torch.mean(costs_rand[idx,0]).item()
            costs_rand_opt_mean = torch.mean(costs_rand_opt[idx,0]).item()
            results_union[i,j,:] = torch.tensor([costs_tt_mean,costs_tt_opt_mean,tt_success])
            results_rand_union[j,:]=torch.tensor([costs_rand_mean, costs_rand_opt_mean,rand_success])  
            
            # intersection
            idx = torch.logical_and(idx_tt,idx_rand)
            tt_success = (100*torch.sum(idx_tt).item()/n_test); rand_success = (100*torch.sum(idx_rand).item()/n_test)
            tt_nit_mean = torch.mean(tt_nit[idx]).item(); rand_nit_mean=torch.mean(rand_nit[idx]).item()
            costs_tt_mean = torch.mean(costs_tt[idx,0],dim=0).item()
            costs_tt_opt_mean = torch.mean(costs_tt_opt[idx,0]).item()
            costs_rand_mean = torch.mean(costs_rand[idx,0]).item()
            costs_rand_opt_mean = torch.mean(costs_rand_opt[idx,0]).item()
            results_intersection[i,j,:] = torch.tensor([costs_tt_mean,costs_tt_opt_mean,tt_success])
            results_rand_intersection[j,:]=torch.tensor([costs_rand_mean, costs_rand_opt_mean,rand_success])  
        
        results_union[i,-1,:]=torch.mean(results_rand_union,dim=0)
        results_intersection[i,-1,:]=torch.mean(results_rand_intersection,dim=0)

        print("For Latex: ")

    print("Union: ")
    latex_tt, latex_rand = get_latex_str(results_union,alphas)
    print("tt: \n ",latex_tt)
    print("rand: \n",latex_rand)
                
    print("Intersection: ")
    latex_tt, latex_rand = get_latex_str(results_union,alphas)
    print("tt: \n ",latex_tt)
    print("rand: \n",latex_rand)
