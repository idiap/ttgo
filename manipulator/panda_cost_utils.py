
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
from roma import rotmat_to_unitquat as tfm
import time

class SDF_Cost:
    ''' 
    Compute the cost based on SDF at a batch of input coordiates (X: Nx3 array) 
    Assumption: uniform discretization of each axis x,y,z
    domain (3x2 array) : the domain of x,y and z axis
    sdf_tensor: (n1 x n2 x n3) tensor of sdf values 
    margin (for cost computation): represents the threshold on the distance away from the obstacle 
    (determined from the sdf value) which is considered to be safe for the robot from collision        
    '''
    def __init__(self, sdf_tensor, domain, device="cpu"):
        self.device=device
        self.domain = domain.to(self.device) # a list of three elements (bounds on x,y and z axis)
        self.n = torch.tensor(sdf_tensor.shape).to(self.device) # number of discretization points on x,y,z axis
        length_xyz = torch.tensor([domain[i][-1]-domain[i][0] for i in range(3)]).to(self.device)
        self.dh = length_xyz/self.n # discretization width (uniform)
        self.lower_bounds = torch.tensor([domain[i][0] for i in range(3)]).to(self.device)
        self.sdf_tensor = sdf_tensor.to(self.device)
        self.grad_sdf_tensor = torch.gradient(self.sdf_tensor,edge_order=1, 
            spacing=[dx.item()for dx in self.dh]) # get the numerical gradient
        self.grad_sdf_tensor = [g_tensor.to(device) for g_tensor in self.grad_sdf_tensor]
        self.sdf_min = torch.min(self.sdf_tensor)
        self.sdf_max = torch.max(self.sdf_tensor)
        self.batch_size = 10**6

    def set_device(self,device):
        self.device=device

    def domain2idx(self,X):
        ''' 
        get the index of the sdf tensor corresponding to the coordinate
        X: Nx3 array (batch of (x,y,z) coordinates )
        '''
        X0 = X-self.lower_bounds
        I = torch.clip(torch.round((X0/self.dh)), 0*self.n, self.n-1)
        return I.long()

    def idx2domain(self,I):
        '''
        Get the coordinates given the index
        I: Nx3 array
        '''
        X = self.lower_bounds + I*self.dh
        return X 
    
    def sdf(self,X):
        ''' 
        compute the sdf values given a mini batch of coordinates X: Nx3 tensor
        '''
        I = self.domain2idx(X) # Nx3
        dX = X-self.idx2domain(I)
        sdf_interp = self.sdf_tensor[I[:,0],I[:,1],I[:,2]] # Nx1
        for i in range(3):
            sdf_interp +=  dX[:,i]*self.grad_sdf_tensor[i][I[:,0],I[:,1],I[:,2]] 
        return sdf_interp





##########################################################################################
##########################################################################################

class PandaCost:
    '''
        Cost functions for various motion planning tasks for Panda manipulator and 
        collision cost is computed using an SDF
        params:
        p2p: object of class Point2PointMotion
        robot: object with method forward_kin() that gives the positions of various points
               of interest on the robot (key_points, end-effector position and orientation)
               given a batch of joint angles (of shape: batch_size x n_joints) and has 
               properties max_config and min_config (the bounds on the states)
        sdf_cost: object of class SDF_Cost
        w_<task>: weightage for each sub-objective in the cost (will be normalized to sum to 1)       
    '''
    def __init__(self,robot, p2p_motion=None, sdf_cost=None, Rd_0=torch.eye(3), v_d=torch.ones(3), 
        b_obst=0.05, b_goal=0.05, b_orient=0.05, b_ee=1.5, b_control=1.5,
        w_obst=1., w_goal=1., w_orient=1., w_ee=0., w_control=1., device="cpu"):

        self.device = device
        self.robot = robot
        self.sdf_cost = sdf_cost 
        self.n_kp = self.robot.n_kp # get the number of key points

        # joint angle limits
        self.max_x = self.robot.max_config.to(self.device); 
        self.min_x = self.robot.min_config.to(self.device); 
     
        # weights for the sub-objectives
        w_control=w_control; w_ee=w_ee; w_goal=w_goal; w_obst=w_obst; w_orient=w_orient
        if p2p_motion==None:
            w_control=0.; w_ee = 0.; 
        else:
            self.p2p = p2p_motion
            self.n = self.p2p.n
            self.K = self.p2p.K
            self.T = self.p2p.T
            self.p2p.set_bound([self.min_x, self.max_x])
            self.t = torch.linspace(0,1,self.T).to(self.device)

        # normalize the weights
        w_sum = w_obst+w_goal+w_orient+w_control+w_ee
        self.w_obst = w_obst/w_sum # weight on obstacle avoidance
        self.w_ee = w_ee/w_sum # weight on end-effector trajectory to be straight line or joint sweep (you choose)
        self.w_goal = w_goal/w_sum # weight on reaching a target state (could be via pose or final pose depending on the problem formulaiton)
        self.w_orient = w_orient/w_sum
        self.w_control = w_control/w_sum 

        # The following  represent nominal/base/acceptable costs
        self.b_obst = b_obst; self.b_goal = b_goal; self.b_orient=b_orient
        self.b_ee = b_ee; self.b_control = b_control; 

        self.Rd_0 = Rd_0.to(device) # desired orientation
        self.v_d = v_d.to(device)  # allowed axis of rotation

    def set_device(self,device):
        self.device=device


    def dist_goal(self, x_actual, x_target):
        ''' 
            Quantify error in end-effector position from the desired position
        '''    
        d_goal= torch.linalg.norm(x_actual.view(-1,3)-x_target.view(-1,3), dim=1)
        return d_goal 

    def dist_orientation(self, Ra_0):
        '''
        Quantify error in orientation (flexible orientation)
        Rd_0: a 3x3 rotation matrix corresponding to the desired orientation (w.r.t world frame)
        v_d: 1x3 vector w.r.t. Rd frame  w.r.t. which rotation is allowed
        Ra_0: ..x3x3 batch of rotation matrices w.r.t world frame
        returns distance in range (0,1) 
        '''
        v_d = (self.v_d/torch.linalg.norm(self.v_d)).view(-1) # normalize the axis vector
        Rd_0 = self.Rd_0.view(3,3)
        Ra_d = torch.matmul(Ra_0,Rd_0.T) # Ra w.r.t. Rd frame    
        qa_d = tfm(Ra_d+1e-6) # corresponding quarternion (imaginary_vector,real)
        va_d = qa_d[:,:-1]
        va_d = va_d/(torch.linalg.vector_norm(va_d,dim=1).view(-1,1)+1e-9) # axis vector w.r.t Rd frame to get Ra_d
        d_orient = 1-torch.sum(va_d*v_d,dim=1)**2          
        return d_orient 


    def dist_orient_traj(self,end_R_t):
        '''
            Impose cost on the orientation to be fixed
        '''
        v_d = (self.v_d/torch.linalg.norm(self.v_d)).view(-1) # normalize the axis vector
        Rd_0 = self.Rd_0.T.view(1,1,3,3)
        Ra_d = torch.matmul(end_R_t,Rd_0.repeat(end_R_t.shape[0],end_R_t.shape[1],1,1)) # Ra w.r.t. Rd frame    
        qa_d = tfm(Ra_d+1e-6) # corresponding quarternion (imaginary_vector,real)
        va_d = qa_d[:,:,:-1]
        va_d = va_d/(torch.linalg.vector_norm(va_d,dim=-1).view(end_R_t.shape[0],end_R_t.shape[1],1)+1e-9) # axis vector w.r.t Rd frame to get Ra_d
        d_orient = 1-torch.sum(va_d*v_d,dim=-1)**2  
        return torch.mean(d_orient,dim=-1) 
       
        

    
    def dist_obst(self,key_loc_t):
        ''' 
        Quantify collision with obstacles
        key_position_t:  # batch x time x joint x keys x positions
        '''
        b_size, n_time, n_joints, n_keys, _ = key_loc_t.shape
        sdf_values = (self.sdf_cost.sdf(key_loc_t.reshape(-1,3))) # (batch x time x n_joints x n_keys) x 1
        sdf_values = sdf_values.view(b_size,n_time,n_joints,n_keys,1) # batch x time x joint x keys x 1
        sdf2cost = -1*torch.min(0*sdf_values,
            sdf_values-self.robot.key_points_margin.expand(1,1,-1,-1,-1))
        d_obst = torch.sum(sdf2cost*self.robot.key_points_weight.expand(1,1,-1,-1,-1), 
            dim=(1,2,3,4))
        return d_obst

    def dist_ee(self,ee_loc_t):
        ''' 
        Quantify the straightness of the end-effector(ee) trajctory
        ee_loc_t: batch x time x ee-postion # trajectory of the end-effector
        '''        
        d_traj = torch.sum(torch.linalg.norm(ee_loc_t[:,1:,:]-ee_loc_t[:,:-1,:],dim=2),dim=1) # length of the trajectory
        d_shortest_traj = torch.linalg.norm(ee_loc_t[:,-1,:]-ee_loc_t[:,0,:],dim=1) # length of the line between initial and final point
        d_ee= torch.abs(d_traj-d_shortest_traj)/(1e-6+d_shortest_traj)
        return d_ee 

    def dist_control(self,theta_t):

        ''' 
        Quantify joint angle sweep 
        theta_t: batch x time x joint, d_theta: batch x 1 x joint
        '''
        theta_shortest = 0.5*torch.linalg.norm(self.robot.theta_max - self.robot.theta_min)
        theta_total = torch.sum(torch.linalg.norm(theta_t[:,1:,:]-theta_t[:,:-1,:],dim=2), dim=1)
        # theta_shortest = torch.linalg.norm(theta_t[:,-1,:]-theta_t[:,0,:], dim=1)
        d_control = torch.abs(theta_total-theta_shortest)/(1e-6+theta_shortest)
        return d_control
    


    def cost_j2j(self,x, theta_0, theta_f):
        ''' Given  (init_joint_angle, final_joint_angle, basis_weights) define the cost for reaching task'''
        b_size = x.shape[0]
        x = x.to(self.device)
        theta_0 = theta_0.repeat(b_size,1)
        theta_f = theta_f.repeat(b_size,1)
        w = 1*x # weights
        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_f,w) #joint angles: batch x time x joint
        _,n_time,n_joints = theta_t.shape

        key_loc_t, end_loc_t, end_R_t = self.robot.forward_kin(theta_t.reshape(b_size*n_time,n_joints)) # (batchxtime) x joint x key x positions

        _, n_frames, n_keys, _ = key_loc_t.shape
        key_loc_t = key_loc_t.view(b_size,n_time,n_frames,n_keys,3) # batch x time x joint x keys x positions
        end_loc_t = end_loc_t.view(b_size,n_time,3) # batch x time x  end-eff-positions
        end_R_t = end_R_t.view(b_size,n_time,3,3)
       # Cost due to obstacle
        d_obst = self.dist_obst(key_loc_t)

        # Cost on end-effector traj (aim to keep it straight)
        d_control = self.dist_control(theta_t)
        
        d_orient = self.dist_orient_traj(end_R_t)
        
    
        c_total =  self.w_obst*d_obst/self.b_obst+ self.w_control*d_control/self.b_control + self.w_orient*d_orient/self.b_orient

        c_return = torch.cat((c_total.view(-1,1),d_obst.view(-1,1),d_orient.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
 

        return c_return

  
    def cost_ik(self,x):
        '''
            Define the cost for IK of manipulator
        '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        goal_loc = x[:,:3]
        q = x[:,3:]  # batch x joint angles
        key_position, end_loc, end_R = self.robot.forward_kin(q) # batch x joint x keys x positions
        
        # quantify error in end-effector position
        d_goal = torch.linalg.norm(end_loc-goal_loc, dim=1)
        
        # quantify error in end-effector orientation
        d_orient = self.dist_orientation(end_R)
     
        # quantify collision due to obstacle
        d_obst = self.dist_obst(key_position[:,None,:,:,:])

        c_total = self.w_goal*(d_goal/self.b_goal)+self.w_obst*(d_obst/self.b_obst)+self.w_orient*(d_orient/self.b_orient)
        
        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
                d_obst.view(-1,1),d_orient.view(-1,1)),dim=-1) # for analysis
 
        return c_return

    def cost_goal(self,x):
        ''' compute the IK cost for target-reaching '''
        goal_loc = x[:,:3]
        theta = x[:,3:] # joint angles
        key_position, end_loc, end_R = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        # quantify error in end-effector position
        d_goal = torch.linalg.norm(end_loc-goal_loc, dim=1)
        return d_goal    
    
    def cost_orient(self,theta):
        ''' compute the IK cost for orientation '''

        key_position, end_loc, end_R = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        # quantify error in end-effector orientation
        d_orient = self.dist_orientation(end_R)
        return d_orient    
    
    def cost_obst(self,theta):
        ''' compute the IK cost for obstacles '''
        key_position, end_loc, end_R = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        # quantify collision due to obstacle
        d_obst = self.dist_obst(key_position[:,None,:,:,:])
        return d_obst    
    


   
    def cost_j2p(self, x, theta_0):
        '''
             Given the (init_joint_angle(theta_0), final position of ee, weights_of_basis_fcns) 
             define the cost for reaching task
        '''
        b_size = x.shape[0]
        x = x.to(self.device)
        xgoal_1 = x[:,:3] # of the end-effector: batch x positions
        theta_1 = x[:,3:3+self.n] # final configuration, batch x joint-angles
        w = x[:,3+self.n:] # basis weights
        theta_t = self.p2p.gen_traj_p2p(theta_0.repeat(b_size,1),theta_1,w) # batch x time x joint 
        T = theta_t.shape[1]
        _,n_time,n_joints = theta_t.shape
        key_loc_t, end_loc_t, end_R_t = self.robot.forward_kin(theta_t.view(b_size*n_time,n_joints)) # (batchxtime) x joint x key x positions
        _, n_frames, n_keys,_ = key_loc_t.shape

        key_loc_t = key_loc_t.view(b_size,n_time,n_frames,n_keys,-1) # batch x time x joint x keys x positions
        end_loc_t = end_loc_t.view(b_size,n_time,-1) # batch x time x  end-eff-positions
        end_R_t = end_R_t.view(b_size,n_time,3,-1)

        xgoal_actual_1 = end_loc_t[:,-1,:] # batch x end-effector-loc
        end_R_T = end_R_t[:,-1,:,:] # orientation matrix

        # quantify collision 
        d_obst = self.dist_obst(key_loc_t) # batch x 1
        # pickup cost: ensure that the gripper never goes below the desired pickup point during pickup
        dist2surf =xgoal_1[:,-1].view(-1,1)
        dist_pickup = torch.sum(dist2surf>end_loc_t[:,int(2*T/3):,-1], dim=-1)
        d_obst += dist_pickup


        # quantify straightness of end-eff trajectory
        d_ee =  1.*self.dist_ee(end_loc_t)

        # quantify joint angle sweep
        d_control = self.dist_control(theta_t) 

        # quantify target error (desired final ee-pose and the actual pose)
        d_goal = self.dist_goal(xgoal_actual_1, xgoal_1)

        # quantify error in desired orientation
        d_orient = self.dist_orientation(end_R_T)

        # Compute the total Cost
        c_total = self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_goal*d_goal/self.b_goal+self.w_orient*d_orient/self.b_orient+self.w_control*d_control/self.b_control
        
        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
                d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
 
        return c_return


    def cost_j2p2j(self, x, theta_0, theta_2):
        ''' 
            Given the (init_joint_angle (theta_0),  via-point, final_joint_angle(theta_2)) 
            define the cost for reaching task
        '''
        b_size = x.shape[0]
        x = x.to(self.device)

        # xgoal_1 = x[:,:3] # via-point
        # w01 = x[:,3:3+self.n*self.K] 
        # theta_1 = x[:,3+self.n*self.K:3+self.n+self.K*self.n] # via-configuration
        # w12 = x[:,3+self.n+self.K*self.n:]

        xgoal_1 = x[:,:3] # via-point
        theta_1 = x[:,3:3+self.n] # via-configuration
        w01 = x[:,3+self.n:3+self.n+self.n*self.K] # basis weights for phase 1 (theta_0 to theta_1)
        w12 = x[:,3+self.n+self.n*self.K:] # basis weights for phase 2 (theta_1 to theta_2)
    
        # joint-angle traj from initial config to via config        
        theta_01_t = self.p2p.gen_traj_p2p(theta_0.repeat(b_size,1),theta_1,w01) #batch x time/2 x joint
        
        # joint-angle traj from via-config to final config
        theta_12_t = self.p2p.gen_traj_p2p(theta_1,theta_2.repeat(b_size,1),w12) #batch x time/2 x joint

        T01 = theta_01_t.shape[1]
        T12 = theta_12_t.shape[1]
        n_joints = theta_01_t.shape[-1]

        # Find the location of the joints and end-effector
        key_loc_t_01, end_loc_t_01, end_R_t_01 = self.robot.forward_kin(theta_01_t.reshape(b_size*T01,n_joints)) # (batchxtime) x joint x key x positions
        key_loc_t_12, end_loc_t_12, end_R_t_12 = self.robot.forward_kin(theta_12_t.reshape(b_size*T12,n_joints)) # (batchxtime) x joint x key x positions
        
        _, n_frames, n_keys,_ = key_loc_t_01.shape
        
        key_loc_t_01 = key_loc_t_01.view(b_size,T01,n_frames,n_keys,3) # batch x time x joint x keys x positions
        key_loc_t_12 = key_loc_t_12.view(b_size,T12,n_frames,n_keys,3) # batch x time x joint x keys x positions

        end_loc_t_01 = end_loc_t_01.view(b_size,T01,3) # batch x time x  end-eff-positions
        end_loc_t_12 = end_loc_t_12.view(b_size,T12,3) # batch x time x  end-eff-positions

        xgoal_1_actual = end_loc_t_01[:,-1,:] # batch x end-effector-loc
        end_R_T = end_R_t_01.view(b_size,T01,3,-1)[:,self.T,:,:] # rotation matrix at via-point-1

        # quantify error in Orientation (only for picking or via-1)
        d_orient = self.dist_orientation(end_R_T)

        # quantify collision
        d_obst = 0.5*(self.dist_obst(key_loc_t_01)+self.dist_obst(key_loc_t_12))
        
        # quantify straightness of end-effector traj
        d_ee = 0.5*(self.dist_ee(end_loc_t_01)+self.dist_ee(end_loc_t_12))
        
        # quantify joint angle sweep
        d_control = 0.5*(self.dist_control(theta_01_t) + self.dist_control(theta_12_t))
        
        # quantify error in the position of end-effector from desired target points
        d_goal = self.dist_goal(xgoal_1_actual,xgoal_1)

        # Compute the total Cost
        c_total = self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_goal*d_goal/self.b_goal+self.w_orient*d_orient/self.b_orient+self.w_control*d_control/self.b_control

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
                d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
 

        return c_return

    def cost_j2p2p(self, x, theta_0):
            ''' 
                Given the init_joint_angle (theta_0) and two target points to reach in sequence 
                define the cost for reaching task
            '''
            b_size = x.shape[0]
            x = x.to(self.device)
            # xgoal_1 = x[:,:3] # via-point-1
            # xgoal_2 = x[:,3:6] # via-point-1
            # w01 = x[:,6:6+self.n*self.K] 
            # theta_1 = x[:,6+self.n*self.K:6+self.K*self.n+self.n] # via-configuration-1
            # w12 = x[:,6+self.K*self.n+self.n:6+2*self.K*self.n+self.n]
            # theta_2 = x[:,6+2*self.K*self.n+self.n:6+2*self.K*self.n+2*self.n] # via-configuration-1

            xgoal_1 = x[:,:3] # via-point-1
            xgoal_2 = x[:,3:6] # via-point-2
            theta_1 = x[:,6:6+self.n] # via-configuration-1
            theta_2 = x[:,6+self.n:6+2*self.n] # via-configuration-1
            w01 = x[:,6+2*self.n:6+2*self.n+self.n*self.K] # basis weights for phase 1 (theta_0 to theta_1)
            w12 = x[:,6+2*self.n+self.n*self.K:] # basis weights for phase 2 (theta_1 to theta_2)
        

            # joint-angle traj from initial config to via config        
            theta_01_t = self.p2p.gen_traj_p2p(theta_0.repeat(b_size,1),theta_1,w01) #batch x time/2 x joint
            
            # joint-angle traj from via-config-1 to via-config-2 (final config)
            theta_12_t = self.p2p.gen_traj_p2p(theta_1,theta_2,w12) #batch x time/2 x joint

            T01 = theta_01_t.shape[1]
            T12 = theta_12_t.shape[1]
            n_joints = theta_01_t.shape[-1]

            # Find the location of the joints and end-effector
            key_loc_t_01, end_loc_t_01, end_R_t_01 = self.robot.forward_kin(theta_01_t.reshape(b_size*T01,n_joints)) # (batchxtime) x joint x key x positions
            key_loc_t_12, end_loc_t_12, end_R_t_12 = self.robot.forward_kin(theta_12_t.reshape(b_size*T12,n_joints)) # (batchxtime) x joint x key x positions
            
            _, n_frames, n_keys,_ = key_loc_t_01.shape
            
            key_loc_t_01 = key_loc_t_01.view(b_size,T01,n_frames,n_keys,3) # batch x time x joint x keys x positions
            key_loc_t_12 = key_loc_t_12.view(b_size,T12,n_frames,n_keys,3) # batch x time x joint x keys x positions

            end_loc_t_01 = end_loc_t_01.view(b_size,T01,3) # batch x time x  end-eff-positions
            end_loc_t_12 = end_loc_t_12.view(b_size,T12,3) # batch x time x  end-eff-positions

            xgoal_1_actual = end_loc_t_01[:,-1,:] # batch x end-effector-loc
            xgoal_2_actual = end_loc_t_12[:,-1,:] # batch x end-effector-loc
            end_R_T = end_R_t_01.view(b_size,T01,3,-1)[:,self.T,:,:] # rotation matrix at via-point

            # quantify error in Orientation (only for picking or via-1)
            d_orient = self.dist_orientation(end_R_T)

            # quantify collision
            d_obst = 0.5*(self.dist_obst(key_loc_t_01)+self.dist_obst(key_loc_t_12))
            
            # quantify straightness of end-effector traj
            d_ee = 0.5*(self.dist_ee(end_loc_t_01)+self.dist_ee(end_loc_t_12))
            
            # quantify joint angle sweep
            d_control = 0.5*(self.dist_control(theta_01_t) + self.dist_control(theta_12_t))
            
            # quantify error in the position of end-effector from desired target points
            d_goal = 0.5*(self.dist_goal(xgoal_1_actual,xgoal_1)+self.dist_goal(xgoal_2_actual,xgoal_2))

            # Compute the total Cost
            c_total = self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_goal*d_goal/self.b_goal+self.w_orient*d_orient/self.b_orient+self.w_control*d_control/self.b_control

            c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
                    d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
     

            return c_return

    def cost_j2p2p2j_(self, x, theta_0, theta_3):
            ''' 
                Given the init_joint_angle (theta_0), final configuration (theta_3)
                and two via-pointsdefine the cost for reaching task
            '''
            b_size = x.shape[0]
            x = x.to(self.device)
    
            # xgoal_1 = x[:,:3] # via-point-1
            # xgoal_2 = x[:,3:6] # via-point-1
            # w01 = x[:,6:6+self.n*self.K] 
            # theta_1 = x[:,6+self.n*self.K:6+self.K*self.n+self.n] # via-configuration-1
            # w12 = x[:,6+self.K*self.n+self.n:6+2*self.K*self.n+self.n]
            # theta_2 = x[:,6+2*self.K*self.n+self.n:6+2*self.K*self.n+2*self.n] # via-configuration-1
            # w23 = x[:,6+2*self.K*self.n+2*self.n:]
    
            xgoal_1 = x[:,:3] # via-point-1
            xgoal_2 = x[:,3:6] # via-point-2
            theta_1 = x[:,6:6+self.n] # via-configuration-1
            theta_2 = x[:,6+self.n:6+2*self.n] # via-configuration-1
            w01 = x[:,6+2*self.n:6+2*self.n+self.n*self.K] # basis weights for phase 1 (theta_0 to theta_1)
            w12 = x[:,6+2*self.n+self.n*self.K:6+2*self.n+2*self.n*self.K] # basis weights for phase 2 (theta_1 to theta_2)
            w23 = x[:,6+2*self.n+2*self.n*self.K:]

            
            # joint-angle traj from initial config to via config        
            theta_01_t = self.p2p.gen_traj_p2p(theta_0.repeat(b_size,1),theta_1,w01) #batch x time/2 x joint
            
            # joint-angle traj from via-config-1 to via-config-2
            theta_12_t = self.p2p.gen_traj_p2p(theta_1,theta_2,w12) #batch x time/2 x joint
        
            # joint-angle traj from via-config-1 to via-config-2
            theta_23_t = self.p2p.gen_traj_p2p(theta_2,theta_3.repeat(b_size,1),w23) #batch x time/2 x joint

            T01 = theta_01_t.shape[1]
            T12 = theta_12_t.shape[1]
            T23 = theta_23_t.shape[1]
            n_joints = theta_01_t.shape[-1]

            # Find the location of the joints and end-effector
            key_loc_t_01, end_loc_t_01, end_R_t_01 = self.robot.forward_kin(theta_01_t.reshape(b_size*T01,n_joints)) # (batchxtime) x joint x key x positions
            key_loc_t_12, end_loc_t_12, end_R_t_12 = self.robot.forward_kin(theta_12_t.reshape(b_size*T12,n_joints)) # (batchxtime) x joint x key x positions
            key_loc_t_23, end_loc_t_23, end_R_t_23 = self.robot.forward_kin(theta_23_t.reshape(b_size*T23,n_joints)) # (batchxtime) x joint x key x positions
            
            _, n_frames, n_keys,_ = key_loc_t_01.shape
            
            key_loc_t_01 = key_loc_t_01.view(b_size,T01,n_frames,n_keys,3) # batch x time x joint x keys x positions
            key_loc_t_12 = key_loc_t_12.view(b_size,T12,n_frames,n_keys,3) # batch x time x joint x keys x positions
            key_loc_t_23 = key_loc_t_23.view(b_size,T23,n_frames,n_keys,3) # batch x time x joint x keys x positions

            end_loc_t_01 = end_loc_t_01.view(b_size,T01,3) # batch x time x  end-eff-positions
            end_loc_t_12 = end_loc_t_12.view(b_size,T12,3) # batch x time x  end-eff-positions
            end_loc_t_23 = end_loc_t_23.view(b_size,T23,3) # batch x time x  end-eff-positions

            xgoal_1_actual = end_loc_t_01[:,-1,:] # batch x end-effector-loc
            xgoal_2_actual = end_loc_t_12[:,-1,:] # batch x end-effector-loc
            end_R_T_1 = end_R_t_01.view(b_size,T01,3,-1)[:,self.T,:,:] # rotation matrix at via-point-1
            end_R_T_2 = end_R_t_12.view(b_size,T12,3,-1)[:,self.T,:,:] # rotation matrix at via-point-2

            # quantify error in Orientation (only for picking or via-1)
            d_orient = 0.5*(self.dist_orientation(end_R_T_1)+self.dist_orientation(end_R_T_2))

            # quantify collision
            d_obst = (1./3.)*(self.dist_obst(key_loc_t_01)+self.dist_obst(key_loc_t_12)+self.dist_obst(key_loc_t_23))
            
            # quantify straightness of end-effector traj
            d_ee = (1./3.)*(self.dist_ee(end_loc_t_01)+self.dist_ee(end_loc_t_12)+self.dist_ee(end_loc_t_23))
            
            # quantify joint angle sweep
            d_control = (1./3.)*(self.dist_control(theta_01_t) + self.dist_control(theta_12_t)+self.dist_control(theta_23_t))
            
            # quantify error in the position of end-effector from desired target points
            d_goal = 0.5*(self.dist_goal(xgoal_1_actual,xgoal_1)+self.dist_goal(xgoal_2_actual,xgoal_2))

            # Compute the total Cost
            c_total = self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_goal*d_goal/self.b_goal+self.w_orient*d_orient/self.b_orient+self.w_control*d_control/self.b_control

            c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
                    d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
     

            return c_return

    def cost_j2p2p2j(self, x, theta_0, theta_3):
            ''' 
                Given the init_joint_angle (theta_0), final configuration (theta_3)
                and two via-pointsdefine the cost for reaching task
            '''
            b_size = x.shape[0]
            x = x.to(self.device)
    
            # xgoal_1 = x[:,:3] # via-point-1
            # xgoal_2 = x[:,3:6] # via-point-1
            # w01 = x[:,6:6+self.n*self.K] 
            # theta_1 = x[:,6+self.n*self.K:6+self.K*self.n+self.n] # via-configuration-1
            # w12 = x[:,6+self.K*self.n+self.n:6+2*self.K*self.n+self.n]
            # theta_2 = x[:,6+2*self.K*self.n+self.n:6+2*self.K*self.n+2*self.n] # via-configuration-1
            # w23 = x[:,6+2*self.K*self.n+2*self.n:]
    
            xgoal_1 = x[:,:3] # via-point-1
            xgoal_2 = x[:,3:6] # via-point-2
            theta_1 = x[:,6:6+self.n] # via-configuration-1
            theta_2 = x[:,6+self.n:6+2*self.n] # via-configuration-1
            w01 = x[:,6+2*self.n:6+2*self.n+self.n*self.K] # basis weights for phase 1 (theta_0 to theta_1)
            w12 = x[:,6+2*self.n+self.n*self.K:6+2*self.n+2*self.n*self.K] # basis weights for phase 2 (theta_1 to theta_2)
            w23 = x[:,6+2*self.n+2*self.n*self.K:]

            
            # joint-angle traj from initial config to via config        
            theta_01_t = self.p2p.gen_traj_p2p(theta_0.repeat(b_size,1),theta_1,w01) #batch x time/2 x joint
            
            # joint-angle traj from via-config-1 to via-config-2
            theta_12_t = self.p2p.gen_traj_p2p(theta_1,theta_2,w12) #batch x time/2 x joint
        
            # joint-angle traj from via-config-1 to via-config-2
            theta_23_t = self.p2p.gen_traj_p2p(theta_2,theta_3.repeat(b_size,1),w23) #batch x time/2 x joint

            T01 = theta_01_t.shape[1]
            T12 = theta_12_t.shape[1]
            T23 = theta_23_t.shape[1]
            n_joints = theta_01_t.shape[-1]

            # Find the location of the joints and end-effector
            key_loc_t_01, end_loc_t_01, end_R_t_01 = self.robot.forward_kin(theta_01_t.reshape(b_size*T01,n_joints)) # (batchxtime) x joint x key x positions
            key_loc_t_12, end_loc_t_12, end_R_t_12 = self.robot.forward_kin(theta_12_t.reshape(b_size*T12,n_joints)) # (batchxtime) x joint x key x positions
            key_loc_t_23, end_loc_t_23, end_R_t_23 = self.robot.forward_kin(theta_23_t.reshape(b_size*T23,n_joints)) # (batchxtime) x joint x key x positions
            
            _, n_frames, n_keys,_ = key_loc_t_01.shape
            
            key_loc_t_01 = key_loc_t_01.view(b_size,T01,n_frames,n_keys,3) # batch x time x joint x keys x positions
            key_loc_t_12 = key_loc_t_12.view(b_size,T12,n_frames,n_keys,3) # batch x time x joint x keys x positions
            key_loc_t_23 = key_loc_t_23.view(b_size,T23,n_frames,n_keys,3) # batch x time x joint x keys x positions

            end_loc_t_01 = end_loc_t_01.view(b_size,T01,3) # batch x time x  end-eff-positions
            end_loc_t_12 = end_loc_t_12.view(b_size,T12,3) # batch x time x  end-eff-positions
            end_loc_t_23 = end_loc_t_23.view(b_size,T23,3) # batch x time x  end-eff-positions

            xgoal_1_actual = end_loc_t_01[:,-1,:] # batch x end-effector-loc
            xgoal_2_actual = end_loc_t_12[:,-1,:] # batch x end-effector-loc
            end_R_T_1 = end_R_t_01.view(b_size,T01,3,-1)[:,self.T,:,:] # rotation matrix at via-point-1
            end_R_T_2 = end_R_t_12.view(b_size,T12,3,-1)[:,self.T,:,:] # rotation matrix at via-point-2

            # quantify error in Orientation (only for picking or via-1)
            d_orient = 0.75*self.dist_orientation(end_R_T_1)+ 0.25*self.dist_orientation(end_R_T_2)

            # quantify collision
            d_obst = (1./3.)*(self.dist_obst(key_loc_t_01)+self.dist_obst(key_loc_t_12)+self.dist_obst(key_loc_t_23))
            
            # pickup cost: ensure that the gripper never goes below the desired pickup point during pickup
            dist2surf =xgoal_1[:,-1].view(-1,1)
            dist_pickup = torch.sum(dist2surf>end_loc_t_01[:,int(2*T01/3):,-1], dim=1)
            dist_pickup+=torch.sum(dist2surf>end_loc_t_12[:,:int(1*T12/3),-1], dim=1)
            
            d_obst += dist_pickup

            # quantify straightness of end-effector traj
            d_ee = (1./3.)*(self.dist_ee(end_loc_t_01)+self.dist_ee(end_loc_t_12)+self.dist_ee(end_loc_t_23))
            
            # quantify joint angle sweep
            d_control = (1./3.)*(self.dist_control(theta_01_t) + self.dist_control(theta_12_t)+self.dist_control(theta_23_t))
            
            # quantify error in the position of end-effector from desired target points
            d_goal = 0.5*(self.dist_goal(xgoal_1_actual,xgoal_1)+self.dist_goal(xgoal_2_actual,xgoal_2))

            # dont let the ee go below

            # Compute the total Cost
            c_total = self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_goal*d_goal/self.b_goal+self.w_orient*d_orient/self.b_orient+self.w_control*d_control/self.b_control

            c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
                    d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
     

            return c_return




    def cost_j2p_arbitrary(self, x):
        '''
          Given the (arbitrary init_joint_angle, final position of ee,
          weights_of_basis_fcns) define the cost for reaching task
        '''
        b_size = x.shape[0]
        x = x.to(self.device)
        theta_0 = x[:, :self.n] # intial configuration
        goal_desired = x[:,self.n:self.n+3] # of the end-effector: batch x positions
        theta_1 = x[:,self.n+3:2*self.n+3] # final configuration, batch x joint-angles
        w = x[:,2*self.n+3:] # basis weights
        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_1,w) # batch x time x joint 
        _,n_time,n_joints = theta_t.shape
        
        key_loc_t, end_loc_t, end_R_t = self.robot.forward_kin(theta_t.view(b_size*n_time,self.n)) # (batchxtime) x joint x key x positions

        _, n_frames, n_keys,_ = key_loc_t.shape
        key_loc_t = key_loc_t.view(b_size,n_time,n_frames,n_keys,-1) # batch x time x joint x keys x positions

        end_loc_t = end_loc_t.view(b_size,n_time,-1) # batch x time x  end-eff-positions
        end_loc_T = end_loc_t[:,-1,:] # batch x end-effector-loc
        end_R_t = end_R_t.view(b_size,n_time,3,-1)
        end_R_T = end_R_t[:,-1,:,:]

        # Cost due to obstacle
        d_obst = self.dist_obst(key_loc_t) # batch x 1
      
        # Keep end-eff trajectory straight
        d_ee =  1.*self.dist_ee(end_loc_t)

        # Minimize joint angle sweep
        d_control = self.dist_control(theta_t) 
        
        
        d_orient_traj = self.dist_orient_traj(end_R_t)


        # Cost due to target error (desired final ee-pose and the actual pose)
        d_goal = self.dist_goal(end_loc_T, goal_desired)

        d_orient = self.dist_orientation(end_R_T)
        # Total Cost
        c_total = self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_goal*d_goal/self.b_goal+self.w_orient*d_orient/self.b_orient+self.w_control*d_control/self.b_control
        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
                d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
 

        return c_return


 
    def cost_j2j_arbitrary(self,x):
        ''' (Arbitrary init_joint_angle, Arbirary final_joint_angle, basis_weights) define the cost for reaching task'''
        b_size = x.shape[0]
        x = x.to(self.device)
        theta_0 = x[:,:self.n]
        w = x[:,self.n:self.n+self.K*self.n]
        theta_f = x[:,self.n+self.K*self.n:]
        
        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_f,w) #joint angles: batch x time x joint
        _,n_time,n_joints = theta_t.shape

        key_loc_t, end_loc_t, end_R_t = self.robot.forward_kin(theta_t.reshape(b_size*n_time,n_joints)) # (batchxtime) x joint x key x positions

        _, n_frames, n_keys, _ = key_loc_t.shape
        key_loc_t = key_loc_t.view(b_size,n_time,n_frames,n_keys,3) # batch x time x joint x keys x positions
        end_loc_t = end_loc_t.view(b_size,n_time,3) # batch x time x  end-eff-positions


       # Cost due to obstacle
        d_obst = self.dist_obst(key_loc_t)
        
        # Cost on end-effector traj (aim to keep it straight)
        d_ee = self.dist_ee(end_loc_t)
        d_control = self.dist_control(theta_t)
    
    
        c_total =  self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee + self.w_control*d_control/self.b_control

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
                d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
 

        return c_return


