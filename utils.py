'''
    Copyright (c) 2008 Idiap Research Institute, http://www.idiap.ch/
    Written by Suhan Shetty <suhan.shetty@idiap.ch>
'''

import torch 
import numpy as np
np.set_printoptions(2, suppress=True)
torch.set_printoptions(2, sci_mode=False)

def test_ttgo(ttgo, cost, test_task, n_samples_tt, 
              deterministic=True, alpha=0, device='cpu', 
              test_rand=False, robotics=True, cut_total=0.33):
    '''
        Test TTGO for a given application
        test_task: a batch of test set of task paramters
        n_samples_tt: number of samplesfrom tt-model considered in ttgo from the tt-model 
        n_samples_rand: number of samples from uniform distribution for random initialization
        alpha: choose a value between (0,1) for prioritized sampling
        norm: choose the type of sampling method 1 or 2 (chekc the paper)
        cost: the cost function
    '''
    import time
    test_task = test_task.to(device)
    n=ttgo.dim
    n_samples_rand = 1*n_samples_tt
    n_test = test_task.shape[0]

    state_tt = torch.zeros(n_test,n).to(device);  state_tt_opt = state_tt.clone()

    state_rand = state_tt.clone();  state_rand_opt = state_tt.clone()

    tt_t = torch.zeros(n_test).to(device); rand_t = tt_t.clone()
    tt_nit = tt_t.clone(); rand_nit = tt_t.clone()

    for i,sample_task in enumerate(test_task):
        t1 = time.time()
        # sample from tt
        samples = ttgo.sample_tt(n_samples=n_samples_tt, 
            x_task=sample_task.reshape(1,-1),alpha=alpha,deterministic=deterministic)
        # choose the best solution
        state = ttgo.choose_best_sample(samples)
        t2= time.time()
        # optimize
        state_opt, results = ttgo.optimize(state)
        t3 = time.time()
        tt_nit_i = results.nit
        state_tt[i,:]= 1*state
        state_tt_opt[i,:]= 1*state_opt

        t4 = time.time()
        # sample from uniform distribution
        samples_rand = ttgo.sample_random(n_samples=n_samples_rand,
            x_task=sample_task.reshape(1,-1))
        # choose the best sample
        state = ttgo.choose_best_sample(samples_rand)
        t5=time.time()
        # optimize
        state_opt, results = ttgo.optimize(state)
        t6=time.time()
        rand_nit_i = results.nit
        
        state_rand[i,:]= 1*state
        state_rand_opt[i,:]= 1*state_opt 
                    
        tt_t[i]=(t2-t1);rand_t[i]=(t5-t4);
        tt_nit[i] = tt_nit_i; rand_nit[i] = rand_nit_i

    costs_tt = cost(state_tt);costs_tt_opt = cost(state_tt_opt)
    costs_rand = cost(state_rand);costs_rand_opt = cost(state_rand_opt)
 
    print("################################################################")
    print("################################################################")
    print("deterministic:{}  |  alpha:{}  |  n_samples_tt:{}  |  n_samples_rand:{} | ".format(deterministic,
        alpha,n_samples_tt,n_samples_rand))
    print('################################################################')
    print("################################################################")
    
    print("Cost TT (raw)           : ", torch.mean(costs_tt,dim=0))
    print("Cost TT (optimized)     : ", torch.mean(costs_tt_opt,dim=0))

    if test_rand==True:
        print("Cost rand (raw)           : ", torch.mean(costs_rand,dim=0))
        print("Cost rand (optimized)     : ", torch.mean(costs_rand_opt,dim=0))

    if robotics==True:

        n_test = costs_tt.shape[0]
        idx_tt = costs_tt_opt[:,0]<cut_total #torch.logical_and(costs_tt_opt[:,0]<cut_total,costs_tt_opt[:,1]<cut_goal)
        idx_rand = costs_rand_opt[:,0]<cut_total #torch.logical_and(costs_rand_opt[:,0]<cut_total,costs_rand_opt[:,1]<cut_goal)
        idx1 = torch.logical_and(idx_tt,idx_rand)
        idx2 = torch.logical_or(idx_tt,idx_rand)
        to_print = [" (intersection)", " (union)"]
        for i, idx in enumerate([idx1,idx2]):
            print('-------------------------------------')
            print('Performance, c_total < ',cut_total, to_print[i])
            print("-------------------------------------")
            print("Success-rate (tt vs rand) : ",torch.sum(idx_tt).item()/n_test,torch.sum(idx_rand).item()/n_test)
            print("# iterations (tt vs rand) : ", torch.mean(tt_nit[idx]).item(),torch.mean(rand_nit[idx]).item())

            print("Cost-mean-tt-raw:",torch.mean(costs_tt[idx],dim=0))
            print("Cost-mean-tt-opt:",torch.mean(costs_tt_opt[idx],dim=0))

            print("Cost-mean-rand-raw:",torch.mean(costs_rand[idx],dim=0))
            print("Cost-mean-rand-opt:",torch.mean(costs_rand_opt[idx],dim=0))


    return costs_tt, costs_tt_opt, costs_rand, costs_rand_opt, tt_nit, rand_nit

##########################################################################################
##########################################################################################
##########################################################################################


class Point2PointMotion:
    '''
    Generates point to point motion satisfying the boundary conditions while maintaining:
        - the velocity at the intial and final step zero,
        - the bounds on the trajectory (Ex: joint limits)

    The generated trajectory trajectory represents the phase of the movement t in (0,1).
    params: 
        - dt: time/phase step (assumin t in (0,1))
        - K: number of basis functions 
        - basis: {"rbf", "rbf2", "bs"} where "rbf2" is the inverse rbf, "bs" is bernstein polynomial
        - n: number of variables/states
    '''
    def __init__(self, n,  dt=0.01, K=3, basis="rbf", bounds=None, device="cpu"):
        self.device = device
        self.n = n # number of variables/coordinates
        self.T = int(1/dt) # number of time steps
        self.t = torch.linspace(0,1,self.T).to(device) # phase
        self.K = K # number of basis functions
        self.basis = basis
        if basis == "rbf":
            self.Phi = self.Phi_rbf().to(device)
        elif basis == "rbf2":
            self.Phi = self.Phi_rbf2().to(device)
        elif basis == "bs":
            self.Phi = self.Phi_Bs().to(device)
        self.set_bound(bounds) # bounds is either None (no limit) or a list containing lower and upper bound

    def set_device(self,device):
        self.device=device

    def set_bound(self, bounds):
        if bounds is None:
            bounds=[]
            bounds.append(torch.tensor([-10**5]*self.n).to(device)) # lower bound
            bounds.append(-1*bounds[0])
        self.lower_bound = bounds[0].reshape(1,1,-1)  # lower limit on the trajectory
        self.upper_bound = bounds[1].reshape(1,1,-1) # upper limit on the trajectory


    def Phi_rbf(self): #RBF
        t = torch.linspace(0,1,self.T).to(self.device)
        r_rbf = 0.5/(self.K) # radius
        c_rbf = torch.linspace(0,1,self.K+2).to(self.device)[1:-1] # centers
        Phi = torch.empty((self.T,self.K)).to(self.device)
        for k in range(self.K):
            Phi[:,k]=torch.exp(-(t-c_rbf[k])**2/r_rbf**2)
        return Phi

    def Phi_rbf2(self): # Inverse RBF
        t = torch.linspace(0,1,self.T).to(self.device)
        r_rbf = 0.5/self.K
        c_rbf = torch.linspace(0,1,self.K+2).to(self.device)[1:-1]   
        Phi = torch.empty((self.T,self.K)).to(self.device)
        for k in range(self.K):
            Phi[:,k] = (1/(1+torch.exp((t-c_rbf[k])**2/r_rbf**2)))
        return Phi


    def Phi_Bs(self): # Bernstein Polynomial
        t = torch.linspace(0,1,self.T)
        Phi = torch.zeros((self.T,self.K))
        for k in range(self.K):
            b = np.math.factorial(self.K)/(np.math.factorial(self.K-k)*np.math.factorial(k))
            Phi[:,k]=(b*((1-t)**(self.K-k))*(t**k))
        Phi = Phi - Phi[0,:]+ 1e-9 
        return Phi

    def gen_traj(self,x_0, w):
        '''
            Given the initial state (batch x n ) and the weights (batch x K*n)
            generate trajectories with only initial condition satisfied
        '''
        batch_size = w.shape[0]
        x_0 = x_0[:,None,:].repeat(1,self.T,1) #batch x time x n, initial condition
        w = w.reshape(batch_size,self.K,self.n) #weights
        z_t = torch.einsum('jk,ikl->ijl',self.Phi,w) # batch x time x n
        z_t = z_t - z_t[:,0,:][:,None,:] # so that z(0) = 0
        x_t = x_0 + z_t
        x_t_bounded = self.bound_traj(x_t) # clip the trajectory to maintain the upper and lower limits
        return x_t_bounded #.reshape(batch_size,self.T,self.n)

    def gen_traj_p2p(self,x_0, x_f, w):
        ''' 
            generate trajectory with boundary conditions satisfied
            x_0: batch x n, initial state
            x_f: batc x n, final state
            w: batch x (K*n), weights of basis function, the 
        '''
        batch_size = w.shape[0]
        x_0 = x_0.reshape(batch_size,1,self.n).repeat(1,self.T,1) #batch x time x n
        x_f = x_f.reshape(batch_size,1,self.n).repeat(1,self.T,1) #batch x time x n
        w = w.reshape(batch_size,self.K,self.n)
        z_t = torch.einsum('jk,ikl->ijl',self.Phi,w) # batch x time x n
        z_0 = z_t[:,0,:][:,None,:]
        z_f = z_t[:,-1,:][:,None,:]
        x_t = x_0 + z_t - z_0 + torch.einsum('j,ijk->ijk',self.t,x_f-x_0+z_0-z_f) # x(t) = x(0)+ z(t)-z(0)+t*(x(1)-x(0)+z(0)-z(1))
        x_t_bounded = self.bound_traj(x_t)  # clip the trajectory to maintain the upper and lower limits
        return x_t_bounded # (batch_size,self.T,self.n)

 
    def bound_traj(self,x):
        ''' 
            clip the given trajectories (batch x T x n)
            within the limits and smoothen it and maintain the boundary conditions
        '''
        delta = self.upper_bound-self.lower_bound
        lower_x = self.lower_bound + delta*0.01
        upper_x = self.upper_bound - delta*0.01
        x = torch.clip(x, lower_x, upper_x ) # clip it
        
        # running average for filtering (also ensures zero velocity at the boundaries)
        k = 4 # set (k>0)
        x = torch.cat((x[:,0,:][:,None,:].repeat(1,2*k,1), x, 
            x[:,-1,:][:,None,:].repeat(1,2*k,1)),dim=1)
        
        cum_l = x[:,k:-k,:].shape[1]
        cum_x = 8*x[:,k:k+cum_l,:]+3*(x[:,(k-1):(k-1+cum_l),:]+
            x[:,(k+1):(k+1+cum_l),:])+2*(x[:,(k-2):(k-2+cum_l),:]+
            x[:,(k+2):(k+2+cum_l),:])+1*(x[:,(k-3):(k-3+cum_l),:]+
            x[:,(k+3):(k+3+cum_l),:])
        cum_w = 2*(4+3+2+1)

        x_transformed = cum_x/cum_w

        return x_transformed
