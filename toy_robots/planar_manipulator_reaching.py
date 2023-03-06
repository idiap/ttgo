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
from planar_manipulator import PlanarManipulator
from plot_utils import plot_chain
np.set_printoptions(3, suppress=True)
torch.set_printoptions(3, sci_mode=False)
import sys
sys.path.append('../')
from ttgo import TTGO
from cost_utils import PlanarManipulatorCost
from utils import Point2PointMotion
from utils import test_ttgo
import tt_utils

import warnings

warnings.filterwarnings('ignore')
#####################################################################
import argparse

if __name__ == '__main__':

    ############################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--d0_x', type=float, default=50)
    parser.add_argument('--d0_theta',type=int, default=50)
    parser.add_argument('--d0_w',type=int, default=50)
    parser.add_argument('--rmax', type=int, default=500)  # max tt-rank for tt-cross
    parser.add_argument('--nswp', type=int, default=30)  # number of sweeps in tt-cross
    parser.add_argument('--margin',type=float, default=0.02)
    parser.add_argument('--b_goal', type=float, default=0.1)
    parser.add_argument('--b_obst', type=float, default=1.)
    parser.add_argument('--b_ee', type=float, default=1.)
    parser.add_argument('--b_control', type=float, default=1.)
    parser.add_argument('--w_goal',type=float, default=1.)
    parser.add_argument('--w_obst',type=float, default=1.)
    parser.add_argument('--w_ee',type=float, default=1.)
    parser.add_argument('--w_control',type=float, default=0.)
    parser.add_argument('--K',type=int, default=2)
    parser.add_argument('--dt',type=float, default=0.02)
    parser.add_argument('--kr', type=int, default=5)
    parser.add_argument('--n_joints',type=int, default=2)
    parser.add_argument('--n_kp',type=int, default=5)
    parser.add_argument('--mp',type=int, default=3) #0: reach-target, 1: one via-point, 2: two via points, 3: two via and return 
    args = parser.parse_args()
   
    file_name = "planar-ik-mp-{}-n_joints-{}-margin-{}-d0_x-{}-d0_theta-{}-d0_w-{}-nswp-{}-rmax-{}-kr-{}-b-{}-{}-{}-{}.pickle".format(args.mp, 
        args.n_joints,args.margin,args.d0_x, args.d0_theta,args.d0_w, args.nswp,
        args.rmax, args.kr, args.b_goal, args.b_obst,args.b_ee,args.b_control)
    print(file_name)

    ##################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is ", device)



    ########################################################    
    # Define the robot
    n_joints = args.n_joints
    link_lengths = torch.tensor([1./n_joints]*n_joints).to(device)
    max_theta = np.pi/1.1
    min_theta = -1*max_theta
    robot = PlanarManipulator(n_joints=n_joints,link_lengths=link_lengths,
        max_theta=max_theta,n_kp=args.n_kp, device=device)

    ########################################################
    # Define the environment and the task (Cost function)
    x_obst = [torch.tensor([0.,-0.75]).to(device)]
    r_obst = [0.25]
    margin=0.02
    
    bounds = [robot.min_config, robot.max_config]
    p2p_motion = Point2PointMotion(n=n_joints,dt=args.dt,K=args.K,basis='rbf',
                                    bounds=bounds, device=device)

    costPlanarManipulator = PlanarManipulatorCost(robot,p2p_motion=p2p_motion,x_obst=x_obst,
                                                  r_obst=r_obst, margin=args.margin,
                                                  w_goal=args.w_goal,w_obst=args.w_obst,
                                                  w_ee=args.w_ee, w_control=args.w_control,
                                                  b_goal=args.b_goal, b_obst=args.b_obst,
                                                  b_ee=args.b_ee, b_control=args.b_control,
                                                  device=device)

    # Initial and final configuration
    theta_0 = torch.tensor([2.1*torch.pi/4,-1.5*torch.pi/4]).view(1,-1).to(device)
    theta_3 = 1*theta_0


    # Pick and place location (via-points: x_1 and x_2)
    x_min_place = -0.75; x_max_place = -0.5;
    y_min_place = -0.5; y_max_place = 1.;

    x_min_pick =  0.5; x_max_pick = 0.75;
    y_min_pick = -0.5; y_max_pick = 1.;

    d0_y = int(args.d0_x/5);
    domain_x1 = [torch.linspace(x_min_pick,x_max_pick,args.d0_x),
                torch.linspace(y_min_pick,y_max_pick,d0_y)]
    domain_x2= [torch.linspace(x_min_place,x_max_place,args.d0_x),
                torch.linspace(y_min_place,y_max_place,d0_y)]

    
    domain_theta = [torch.linspace(min_theta, max_theta,args.d0_theta)]*n_joints
    domain_w = [torch.linspace(min_theta,max_theta,args.d0_w)]*(args.K*n_joints)

    if args.mp==3: # 2-via points and initial and final config given
        def cost(x):
            return costPlanarManipulator.cost_j2p2p2j(x,theta_0,theta_3)[:,0]

        def cost_to_print(x): # for printing results
            return costPlanarManipulator.cost_j2p2p2j(x,theta_0,theta_3)

        domain_task = domain_x1 + domain_x2 
        domain_decision =  domain_theta*2+ domain_w*3
        
    elif args.mp==2: # 2-via-points only initial configuration is given
        def cost(x):
            return costPlanarManipulator.cost_j2p2p(x,theta_0)[:,0]

        def cost_to_print(x): # for printing results
            return costPlanarManipulator.cost_j2p2p(x,theta_0)


        domain_task = domain_x1 + domain_x2 
        domain_decision =  domain_theta*2 + domain_w*2

    elif args.mp==1: # one via point with initial and final config given 
        def cost(x):
            return costPlanarManipulator.cost_j2p2j(x,theta_0,theta_3)[:,0]

        def cost_to_print(x): # for printing results
            return costPlanarManipulator.cost_j2p2j(x,theta_0,theta_3)

        domain_task = domain_x1 
        domain_decision =  domain_theta + domain_w*2

    elif args.mp==0: # only target point is given
        def cost(x):
            x = x.to(device)
            return costPlanarManipulator.cost_j2p(x,theta_0)[:,0]

        def cost_to_print(x): # for printing results
            return costPlanarManipulator.cost_j2p(x,theta_0)

        domain_task = domain_x1
        domain_decision = domain_theta + domain_w



    def pdf(x):
        return torch.exp(-cost(x)**2)


    domain = domain_task+domain_decision
    #########################################################
    # Fit TT-Model
    tt_model = tt_utils.cross_approximate(fcn=pdf,  domain=domain, 
                            rmax=100, nswp=20, eps=1e-3, verbose=True, 
                            kickrank=5, device=device)
    ttgo = TTGO(domain=domain,tt_mod=tt_model.to(device),cost=cost, device=device)
    ########################################################
    # generate test set
    ns = 50
    test_task = torch.zeros(ns,len(domain_task)).to(device)
    for i in range(len(domain_task)):
        unif = torch.distributions.uniform.Uniform(low=domain_task[i][0],high=domain_task[i][-1])
        test_task[:,i]= torch.tensor([unif.sample() for i in range(ns)]).to(device)

    ########################################################

    # Save the model
    torch.save({
        'mp':args.mp,
        'tt_model':ttgo.tt_model,
        'w': (args.w_goal,args.w_obst,args.w_ee,args.w_control),
        'b': (args.b_goal,args.b_obst,args.b_ee,args.b_control),
        'margin': args.margin,
        'domain': domain,
        'test_task': test_task,
        'x_obst':x_obst,
        'r_obst':r_obst,
        'n_joints':args.n_joints,
        'n_kp':args.n_kp,
        'dt':args.dt,
        'theta_0':theta_0,
        'theta_3':theta_3
    }, file_name)

    ########################################################
    # Test the model
    
    norm=1
    print("total-cost | goal | collidion | end-effector | control ")
    for alpha in [0.99,0.9,0.8,0.5]:
        for n_samples_tt in [10,50,100,1000]:
            _ test_ttgo(ttgo=ttgo.clone(), cost=cost_to_print, 
                    test_task=test_task, n_samples_tt=n_samples_tt,
                    alpha=alpha, device=device, test_rand=True)


