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
np.set_printoptions(4, suppress=True)
torch.set_printoptions(4, sci_mode=False)
import sys
sys.path.append('../')
from ttgo import TTGO
from utils import test_ttgo
from ur10_kinematics import Ur10Kinematics
from manipulator_utils import dist_orientation_fixed
import argparse
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    ############################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--dh_x', type=float, default=0.01)
    parser.add_argument('--d0_theta',type=int, default=60)
    parser.add_argument('--rmax', type=int, default=500)  # max tt-rank for tt-cross
    parser.add_argument('--nswp', type=int, default=30)  # number of sweeps in tt-cross
    parser.add_argument('--b_goal', type=float, default=0.05)
    parser.add_argument('--b_orient', type=float, default=0.2) 
    parser.add_argument('--kr', type=float, default=3)
    parser.add_argument('--d_type', type=str, default='uniform') # or {'log', 'uniform'} disctretization of joint angles
    parser.add_argument('--name', type=str)  # file nazme for saving
    args = parser.parse_args()
    file_name = args.name if args.name else "ur10-ik-d0_theta-{}-kr-{}-nswp-{}-rmax-{}-dh-{}-b-{}-{}.pickle".format(args.d0_theta,args.kr, args.nswp, args.rmax, args.dh_x, args.b_orient, args.b_goal)
    print(file_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ############################################################

     # Setup the robot and the environment
   
    ur10 = Ur10Kinematics(device=device)
    n_joints= ur10.n_joints
    
    ############################################################

    # Desired orientation (fixed orientation)
    Rd_0 = torch.eye(3).to(device)

    def cost_all(x): # For inverse kinematics
        x = x.to(device)
        batch_size = x.shape[0]
        goal_loc = x[:,:3]
        q = x[:,3:]  # batch x joint angles
        _, end_loc, end_R = ur10.forward_kin(q) # batch x joint x keys x positions
        
        # cost on error in end-effector position
        d_goal = torch.linalg.norm(end_loc-goal_loc, dim=1)
        
        # cost on error in end-effector orientation
        d_orient = dist_orientation_fixed(Rd_0,end_R,device=device)

        c_total = 0.5*(d_goal/args.b_goal + d_orient/args.b_orient)
        
        c_return = torch.cat((c_total.view(-1,1), d_goal.view(-1,1), d_orient.view(-1,1)),dim=-1)
        return c_return

    def cost(x):
        return cost_all(x)[:,0]


    def pdf(x):
        x = x.to(device)
        pdf_ = torch.exp(-cost(x)**2) 
        return pdf_

    #####################################################################

    # Define the domain
    d_theta_all = [args.d0_theta]*n_joints
    d_theta = [int(d_theta_all[joint]) for joint in range(n_joints)]
    if args.d_type == 'uniform':
        domain_decision = [0.5*torch.linspace(ur10.theta_min[i],0.5*ur10.theta_max[i],d_theta[i]).to(device) for i in range(n_joints)]
    else: # logarithmic scaling
        domain_decision = [exp_space(0.5*ur10.theta_min[i].to('cpu'),0.5*ur10.theta_max[i].to('cpu'),d_theta[i]).to(device) for i in range(n_joints)]

    # Find the work-space
    n_test = 1000
    test_theta = torch.zeros(n_test,n_joints).to(device)
    for i in range(n_joints):
        unif = torch.distributions.uniform.Uniform(low = domain_decision[i][0],high=domain_decision[i][-1])
        test_theta[:,i]= torch.tensor([unif.sample() for i in range(n_test)]).to(device)
    _, test_xpos, _ = ur10.forward_kin(test_theta)
    x_min,_ = torch.min(test_xpos, dim=0)
    x_max,_ = torch.max(test_xpos,dim=0)
    x_min[-1] = 0.1
    idx_select = test_xpos[:,-1]>x_min[-1]
    test_task = test_xpos[idx_select,:]

    # discretize the domain
    domain_task = [torch.linspace(x_min[i], x_max[i], int((x_max[i]-x_min[i])/args.dh_x)).to(device) for i in range(3)]
    domain = domain_task + domain_decision
    print("Discretization: ",[len(x) for x in domain])

    ###############################################################################
    # Fit TT-model

    ttgo = TTGO(domain=domain,pdf=pdf, cost=cost,device=device)
    ttgo.cross_approximate(rmax=args.rmax, nswp=args.nswp,kickrank=args.kr)
    ttgo.round(1e-4)
    # Save
    torch.save({
    'tt_model':ttgo.tt_model,
    'b': (args.b_goal,args.b_orient),
    'd0':args.d0_theta,
    'dh_x': args.dh_x,
    'domain': domain,
    'Rd_0':Rd_0,
    'test_task':test_task
    }, file_name)

    ##############################################################
    # Prepare for test

    sites_task = list(range(len(domain_task)))
    ttgo.set_sites(sites_task)

   # Test the model
    sample_set = [1,10,100,1000]
    alphas = [0.9,0.75]
    cut_total=0.33
    test_robotics_task(ttgo.clone(), cost_all, test_task, alphas, sample_set, cut_total,device)