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
torch.set_default_dtype(torch.float64)

import numpy as np
np.set_printoptions(2, suppress=True)
torch.set_printoptions(2, sci_mode=False)

import sys
sys.path.append('../')
from panda_kinematics import PandaKinematics
from ttgo import TTGO
from manipulator_utils import exp_space,test_robotics_task
from utils import Point2PointMotion, test_ttgo
from panda_cost_utils import PandaCost,SDF_Cost 
import argparse
import warnings
warnings.filterwarnings("ignore")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # step size in end-effector
    parser.add_argument('--dh_x', type=float, default=0.01)
    parser.add_argument('--rmax', type=int, default=100)  # max tt-rank for tt-cross
    parser.add_argument('--nswp', type=int, default=30)  # number of sweeps in tt-cross
    parser.add_argument('--margin',type=float, default=0.1)
    parser.add_argument('--d0_w',type=int, default=30)
    parser.add_argument('--d0_theta',type=int, default=30)
    parser.add_argument('--kr', type=float, default=3)
    parser.add_argument('--b_orient',type=float, default=0.05)
    parser.add_argument('--b_goal', type=float, default=0.05)
    parser.add_argument('--b_obst', type=float, default=0.05)
    parser.add_argument('--b_control', type=float, default=0.5)
    parser.add_argument('--b_ee', type=float, default=1.5)
    parser.add_argument('--d_type', type=str, default='uniform') # or {'log', 'uniform'} disctretization of joint angles
    parser.add_argument('--basis',type=str, default='rbf')
    parser.add_argument('--K',type=int, default=2)
    parser.add_argument('--dt',type=float, default=0.1)
    parser.add_argument('--max_batch',type=int, default=20000)
    parser.add_argument('--mp', type=int, default=0) 
    parser.add_argument('--pick',type=str,default="table")
    args = parser.parse_args()
    

    file_name = "panda-reach-mp-{}-b-{}-{}-{}-{}-{}-d0_theta-{}-d0_w-{}-nswp-{}-rmax-{}-kr-{}-basis-{}-margin-{}-dt-{}.pickle".format(args.mp,
        args.b_goal, args.b_obst, args.b_orient, args.b_control, 
        args.b_ee, args.d0_theta, args.d0_w, args.nswp, args.rmax,
        args.kr, args.basis, args.margin,args.dt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################################################################
    # Define the environment (sdf cost) and the robot
    with torch.no_grad():
        data_sdf = np.load('./data/sdf.npy', allow_pickle=True)[()]
        sdf_matr = data_sdf['sdf_matr']  # SDF tensor
        bounds = torch.tensor(data_sdf['bounds']).to(device) # Bound of the environment
        env_bound = data_sdf['env_bound']  
        shelf_bound = data_sdf['shelf_bound'] 
        box_bound = data_sdf['box_bound'] 

        sdf_tensor = torch.from_numpy(sdf_matr).to(device)
        sdf_cost = SDF_Cost(sdf_tensor=sdf_tensor, domain=bounds, device=device)

        data_keys = np.load('./data/sphere_setting.npy', allow_pickle=True)[()]# key_points
        status_array = data_keys['status_array']
        body_radius = data_keys['body_radius']
        relative_pos = data_keys['relative_pos']

        key_points_weight = torch.from_numpy(status_array).to(device)>0 # 8xMx1
        key_points_weight[-1] = True # activates thesphere on the gripper (inactie by default)
        key_points_margin = torch.from_numpy(body_radius).to(device) #
        key_points_pos = torch.from_numpy(relative_pos).to(device)

        key_points_margin[-1] = args.margin  # end-effector-upper part
        key_points_margin[-1][-1]= 0.03 # gripper
        
        key_points = [key_points_pos, key_points_weight, key_points_margin]
        panda = PandaKinematics(device=device, key_points_data=key_points)

        ##############################################################################
        # Define the domain  and cost function

        # domain of joint angles
        n_joints=7
        d_theta_all = [args.d0_theta]*n_joints
        d_theta = [int(d_theta_all[joint]) for joint in range(n_joints)]
        if args.d_type == 'uniform':
            domain_theta_ik = [torch.linspace(panda.theta_min[i],panda.theta_max[i],d_theta[i]).to(device) for i in range(n_joints)]
            domain_theta_conf = [domain_theta_ik[i]/2 for i in range(n_joints)]
        else: # logarithmic scaling
            domain_theta_ik = [exp_space(panda.theta_min[i].to('cpu'),panda.theta_max[i].to('cpu'),d_theta[i]).to(device) for i in range(n_joints)]
            domain_theta_conf = [domain_theta_ik[i]/2 for i in range(n_joints)]


        # task space for placing in box
        x_min = [-0.25,0.5,0.35]
        x_max = [-0.1,0.7,0.45]
        domain_x_place = [torch.linspace(x_min[i], x_max[i], int((x_max[i]-x_min[i])/args.dh_x)) for i in range(3)]

        if args.pick == "shelf":
            # task space for picking as shelf
            env_bounds = torch.from_numpy(shelf_bound)
            x_min = env_bounds[:,0]; x_max = env_bounds[:,1]
            x_max[0] = 0.7; x_min[0] = 0.45; #x_max[0]-0.15
            x_max[1] = x_max[1]-0.1; x_min[1] = x_min[1]+0.1
            x_max[2] = 0.75; x_min[2] = 0
            domain_x_pick = [torch.linspace(x_min[i], x_max[i], int((x_max[i]-x_min[i])/args.dh_x)) for i in range(3)]
   
            # Parallel (shelf pick)
            v_d = torch.tensor([0.,0.,1.])
            Rd_0 = torch.tensor([[ 0.7071,0.7071,0.], [0.,0.,1],[0.7071, -0.7071, 0.]]) # desired orientation
           
        else:
            # task space forpicking: table
            x_max = [-0.2,0.35,0.2]
            x_min = [-0.8,-0.1,0.03]
            domain_x_pick = [torch.linspace(x_min[i], x_max[i], int((x_max[i]-x_min[i])/args.dh_x)) for i in range(3)]
            # pick orientation
            v_d = torch.tensor([0.,0.,1.]).to(device)
            Rd_0=torch.tensor([[ 0.7071,  0.7071,  0.],
                                [ 0.7071, -0.7071,  0.],
                                [ 0.    ,  0.    , -1.]])



        # Discretize weights
        max_w= 1*panda.theta_max
        min_w = 1*panda.theta_min

        d_w_all =  [args.d0_w]*n_joints# np.linspace(50,100,n_joints)
        d_w = [int(d_w_all[joint]) for joint in range(n_joints)]
        domain_w = [torch.linspace(min_w[i],max_w[i],d_w[i]) for i in range(n_joints)]*args.K


        # Desired orientation of ee at the via-point
        theta_limits = [panda.theta_min, panda.theta_max]
        p2p = Point2PointMotion(dt=args.dt, K=args.K, n=n_joints, basis=args.basis,
                                    bounds=theta_limits, device=device)
        pandaCost = PandaCost(p2p_motion=p2p,robot=panda, sdf_cost=sdf_cost, 
                            Rd_0=Rd_0, v_d=v_d,
                            b_obst=args.b_obst,b_goal=args.b_goal, b_ee = args.b_ee,
                            b_control=args.b_control, b_orient=args.b_orient, 
                            device=device)  

        # For the via-point problem only (intial and final configurations)
        # theta_0 = torch.tensor([ -0.5,  -1.,  0.5, -2.,  0.4,  2.,  1.]).reshape(1,-1).to(device) # nominal pose for picking from shelf
        theta_0 = torch.tensor([0.98, -0.27,  0.8, -1.9, 0.16, 1.7, 0.66]).reshape(1,-1).to(device)
        # theta_0 = torch.tensor([ 0.17, -0.99,  0.99, -1.86,  0.83,  1.15, -1.49]).reshape(1,-1).to(device)
        theta_2 = torch.tensor([-1.06, -0.49,  1.76, -1.67,  0.5 ,  1.82, -1.6 ]).reshape(1,-1).to(device) # on top of the box for dropping the object
        theta_3 = 1*theta_0
        if args.mp==0:
            theta_0 = 1*theta_2

        if args.mp == 0: # target-reaching-from-fixed-initial-configuration
            print("Target reaching from a fixed initial configuration")
            domain_task = domain_x_pick
            domain_decision = domain_theta_ik + domain_w
            domain = domain_task + domain_decision
            def cost_all(x):
                return pandaCost.cost_j2p(x,theta_0)
        
        elif args.mp == 1: # via-point problem between fixed initial and final configurations
            print("Via-point-1 problem between fixed initial and final configurations")
            domain_task = domain_x_pick
            domain_decision = domain_theta_ik + domain_w*2 
            domain = domain_task + domain_decision
            def cost_all(x):
                return pandaCost.cost_j2p2j(x, theta_0, theta_2)

        elif args.mp == 2: # fixed initial, given two target points to reach in sequence
            print("Reach two target points in sequence")
            domain_task = domain_x_pick + domain_x_place
            domain_decision = domain_theta_ik*2 + domain_w*2 
            domain = domain_task + domain_decision
            def cost_all(x):
                return pandaCost.cost_j2p2p(x, theta_0)

        elif args.mp == 3: # fixed initial and , given two target points to reach in sequence
            print("Via-point-2 problem between fixed initial and final configurations")
            domain_task = domain_x_pick + domain_x_place
            domain_decision = domain_theta_ik*2 + domain_w*3
            domain = domain_task + domain_decision
            def cost_all(x):
                return pandaCost.cost_j2p2p2j(x, theta_0, theta_3)


        domain = [x.to(device) for x in domain]

        print("Discretization: ",[len(x) for x in domain])


        def cost(x):
            return cost_all(x)[:,0]
            
        def pdf(x):
            return torch.exp(-cost(x)**2)

        ################################################################################
        # Fit the TT-model
        # with torch.no_grad():
        tt_model = tt_utils.cross_approximate(fcn=pdf,  domain=[x.to(device) for x in domain], 
                                rmax=200, nswp=20, eps=1e-3, verbose=True, 
        # Refine the discretization and interpolate the model
        scale_factor = 10
        site_list = torch.arange(len(domain))#len(domain_task)+torch.arange(len(domain_decision))
        domain_new = tt_utils.refine_domain(domain=domain, 
                                            site_list=site_list,
                                            scale_factor=scale_factor, device=device)
        tt_model_new = tt_utils.refine_model(tt_model=tt_model.to(device), 
                                            site_list=site_list,
                                            scale_factor=scale_factor, device=device)                        kickrank=5, device=device)


        ttgo = TTGO(tt_model=tt_model_new, domain=domain_new, cost=cost,device=device)

    
        ################################################################################

        print("Preparing test set")

        # generate test set for target point 
        ns = 100

        ns_ = 10000
        test_x_1 = torch.zeros(ns_,3).to(device)
        test_x_2 = torch.zeros(ns_,3).to(device)
        for i in range(3):
            unif = torch.distributions.uniform.Uniform(low=domain_x_pick[i][0],high=domain_x_pick[i][-1])
            test_x_1[:,i]= torch.tensor([unif.sample() for i in range(ns_)]).to(device)
            unif = torch.distributions.uniform.Uniform(low=domain_x_place[i][0],high=domain_x_place[i][-1])
            test_x_2[:,i]= torch.tensor([unif.sample() for i in range(ns_)]).to(device)

        test_x_1  = test_x_1[(sdf_cost.sdf(test_x_1)-0.09)>0][:ns] # 
        test_x_2  = test_x_2[(sdf_cost.sdf(test_x_2)-0.09)>0][:ns] # 


        if args.mp==0 or args.mp==1: 
            test_task = test_x_1
        if args.mp==2 or args.mp==3:
            test_task = torch.cat((test_x_1, test_x_2),dim=-1)

        print("Saving the model......")

        torch.save({
            'tt_model':ttgo.tt_model,
            'p2p':p2p,
            'panda': panda,
            'pandaCost':pandaCost,
            'sdf_cost':sdf_cost,
            'w': (pandaCost.w_goal,pandaCost.w_obst,pandaCost.w_orient, pandaCost.w_ee, pandaCost.w_control),
            'b': (args.b_goal,args.b_obst,args.b_orient,args.b_ee, args.b_control),
            'd0':(args.d0_theta, args.d0_w),
            'K': args.K,
            'margin': args.margin,
            'key_points_weight':key_points_weight,
            'key_points_margin':key_points_margin,
            'domains': [domain, domain_task,domain_x_pick, domain_x_place],  
            'theta_0': theta_0,
            'theta_2': theta_2,
            'theta_3': theta_3,
            'dt':args.dt,
            'basis':args.basis,
            'Rd_0': Rd_0,
            'v_d':v_d,
            'mp':args.mp,
            'test_task': test_task,
        }, file_name)


        ############################################################ 
        print("############################")
        print("Test the model")
        print("############################")

        # change dt for testing (even smoother trajectory)
        p2p = Point2PointMotion(dt=0.01, K=args.K, n=n_joints, basis=args.basis,
                                    bounds=theta_limits, device=device)
        pandaCost = PandaCost(p2p_motion=p2p,robot=panda, sdf_cost=sdf_cost, 
                            Rd_0=Rd_0, v_d=v_d,
                            b_obst=args.b_obst,b_goal=args.b_goal, b_ee = args.b_ee,
                            b_control=args.b_control, b_orient=args.b_orient, 
                            device=device)
        ttgo.cost = cost  


    # Test the model
    sample_set = [1,10,100,1000]
    alphas = [0.9,0.75,0.5,0]
    cut_total=0.25
    test_robotics_task(ttgo.clone(), cost_all, test_task, alphas, sample_set, cut_total,device)
