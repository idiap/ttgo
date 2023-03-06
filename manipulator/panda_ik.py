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

# Note: Prefer using jupyter-notebook version as it is more up to date

import torch
import numpy as np
np.set_printoptions(2, suppress=True)
torch.set_printoptions(2, sci_mode=False)

import sys
sys.path.append('../')
from ttgo import TTGO
import tt_utils
from utils import test_ttgo
from manipulator_utils import test_robotics_task
from panda_cost_utils import SDF_Cost, PandaCost
from panda_kinematics import PandaKinematics
import argparse


import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    ############################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--dh_x', type=float, default=0.01)
    parser.add_argument('--d0_theta',type=int, default=50)
    parser.add_argument('--rmax', type=int, default=500)  # max tt-rank for tt-cross
    parser.add_argument('--nswp', type=int, default=50)  # number of sweeps in tt-cross
    parser.add_argument('--kr', type=float, default=5) # kickrank param for tt-cross
    parser.add_argument('--b_goal', type=float, default=0.05) #nominal distance of goal
    parser.add_argument('--b_obst', type=float, default=0.01) # nominal collision 
    parser.add_argument('--b_orient', type=float, default=0.2) #nominal error in orientation
    parser.add_argument('--margin', type=float, default=0.0) #safety margin of collision for end-effector
    parser.add_argument('--d_type', type=str, default='uniform') # or {'log', 'uniform'} disctretization of joint angles
    parser.add_argument('--name', type=str)  # file nazme for saving
    args = parser.parse_args()
    file_name = args.name if args.name else "panda-ik-margin-{}-dh-{}-d0_theta-{}-nswp-{}-rmax-{}-kr-{}-b-{}-{}-{}.pickle".format(args.margin, args.dh_x, args.d0_theta, args.nswp, args.rmax, args.kr, args.b_goal, args.b_obst, args.b_orient)
    print(file_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ############################################################

    
    with torch.no_grad():
        # Setup the robot and the environment

        data_sdf = np.load('./data/sdf.npy', allow_pickle=True)[()]
        sdf_matr = data_sdf['sdf_matr']  
        bounds = torch.tensor(data_sdf['bounds']).float().to(device) # Bound of the environment
        sdf_tensor = torch.from_numpy(sdf_matr).float().to(device)
        sdf_cost = SDF_Cost(sdf_tensor=sdf_tensor, domain=bounds, device=device)
        env_bound = data_sdf['env_bound']  
        shelf_bound = data_sdf['shelf_bound'] 
        box_bound = data_sdf['box_bound'] 

        # key-points on the body of the robot for collision check
        data_keys = np.load('./data/sphere_setting.npy', allow_pickle=True)[()]# key_points
        status_array = data_keys['status_array']
        body_radius = data_keys['body_radius']
        relative_pos = data_keys['relative_pos']
        key_points_weight = torch.from_numpy(status_array).float().to(device) >0 # 8xMx1
        key_points_weight[-1] = 1*args.margin
        key_points_margin = torch.from_numpy(body_radius).float().to(device)#
        key_points_pos = torch.from_numpy(relative_pos).float().to(device)
        key_points = [key_points_pos, key_points_weight, key_points_margin]
        
        # define the robot
        panda = PandaKinematics(device=device, key_points_data=key_points)

        ############################################################
        
        # Define the cost function

        # Specify the doesired orientation
        Rd_0 = torch.tensor([[ 0.7071,0.7071,0.], [0.,0.,1],[0.7071, -0.7071, 0.]]).to(device) # desired orientation
        v_d = torch.tensor([0.,0.,1.]).to(device)
        # Rd = torch.tensor([[ 0,0.,0.], [0.,0.,1],[0., 0., 0.]])

        pandaCost = PandaCost(robot=panda, sdf_cost=sdf_cost,
                            Rd_0=Rd_0, v_d=v_d,b_obst=args.b_obst, 
                            b_goal=args.b_goal,b_orient=args.b_orient,device=device)  


        def cost(x): # For inverse kinematics
            return pandaCost.cost_ik(x)[:,0]

        def cost_all(x): # For inverse kinematics
            return pandaCost.cost_ik(x)

        def pdf(x):
            x = x.to(device)
            pdf_ = torch.exp(-cost(x)**2) 
            return pdf_

        ############################################################################

        # Define the domain for discretization

        n_joints=7
        d_theta_all = [args.d0_theta]*n_joints
        d_theta = [int(d_theta_all[joint]) for joint in range(n_joints)]

        # type of discretization of intervals of decision variables
        if args.d_type == 'uniform':
            domain_decision = [torch.linspace(panda.theta_min[i],panda.theta_max[i],d_theta[i]).to(device) for i in range(n_joints)]
        else: # logarithmic scaling
            domain_decision = [exp_space(panda.theta_min[i],panda.theta_max[i],d_theta[i]).to(device) for i in range(n_joints)]

        # task space of the manipulator (the shelf)
        env_bounds = torch.from_numpy(shelf_bound)
        x_min = env_bounds[:,0]
        x_max = env_bounds[:,1]
        x_max[0] = 0.75; x_min[0]=0.45
        x_max[1] = x_max[1]-0.1
        x_min[1] = x_min[1]+0.1
        x_max[-1] = 0.75; x_min[-1] = 0.
    

        domain_task = [torch.linspace(x_min[i], x_max[i], int((x_max[i]-x_min[i])/args.dh_x)) for i in range(3)]
        domain = domain_task + domain_decision
        print("Discretization: ",[len(x) for x in domain])

        #######################################################################################
        # Fit the TT-Model
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



        ############################################################ 
        print("############################")
        print("Test the model")
        print("############################")

        # generate test set
        ns = 100
        test_task = torch.zeros(ns,len(domain_task)).to(device)
        for i in range(len(domain_task)):
            unif = torch.distributions.uniform.Uniform(low=domain_task[i][0],high=domain_task[i][-1])
            test_task[:,i]= torch.tensor([unif.sample() for i in range(ns)]).to(device)


        torch.save({
            'tt_model':ttgo.tt_model,
            'panda': panda,
            'pandaCost':pandaCost,
            'sdf_cost':sdf_cost,
            'w': (pandaCost.w_goal,pandaCost.w_obst,pandaCost.w_orient),
            'b': (args.b_goal,args.b_obst,args.b_orient),
            'margin': args.margin,
            'key_points_weight':key_points_weight,
            'key_points_margin':key_points_margin,
            'domains': domain,  
            'Rd_0': Rd_0,
            'v_d':v_d,
            'test_task': test_task,
        }, file_name)


        # Test the model
        sample_set = [1,10,100,1000]
        alphas = [0.9,0.75,0.5,0]
        cut_total=0.33
        test_robotics_task(ttgo.clone(), cost_all, test_task, alphas, sample_set, cut_total,device)