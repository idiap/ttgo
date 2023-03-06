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
class PlanarManipulatorCost:
    ''' 
    Cost functions for various operations with planar manipulator
    Assumes obstacles to be spheres in a plane
    '''
    def __init__(self, robot,p2p_motion=None, x_obst=[], r_obst=[],margin=0.02,
     w_goal=0., w_obst=0.7, w_orient=0., w_ee=0., w_control=0.3,
     b_goal=0.2, b_obst=0.2,b_orient=0.5, b_ee=1., b_control=1., device='cpu'):
        self.device=device
        self.robot=robot # an object of class PlanarManipulator
        self.x_obst=[x.to(device) for x in x_obst] # centers of the obstacles/spheres
        self.r_obst=r_obst # radius of the spheres/spheres
        self.margin=margin # safety margin from the surface (considers the width of the links)
        # Define the nominal cost and weights for each individal cost
        self.b_goal=b_goal; self.b_obst=b_obst; self.b_orient=b_orient; self.b_ee=b_ee; self.b_control=b_control;
        self.w_goal=w_goal; self.w_obst=w_obst; self.w_orient=w_orient; self.w_ee=w_ee; self.w_control=w_control;
        self.p2p = p2p_motion
        self.n_joints = robot.n_joints

    def dist_goal(self, x_goal, ee_loc):
        '''
        metric for error in end-effector pose from the desired
        x_goal: desired position of the end-effector, batch x 2
        ee_loc: actual position/location of the end-effector, batch x 2
        '''
        d_goal = torch.linalg.norm(ee_loc-x_goal, dim=-1)
        return d_goal

    def dist_orient(self, theta_ee_goal, theta_ee_actual):
        '''
            Orientation error
        '''
        d_err = torch.abs(theta_ee_goal-theta_ee_actual)
        d_orient = torch.min(d_err,2*torch.pi-d_err)
        return d_orient

    def dist_traj(self,x_t):
        '''
            metric for length of a trajectory (to seek minimum length trajectory)
            x_t: batch x time x coordinates
        '''
        d_shortest = torch.linalg.norm(x_t[:,-1,:]-x_t[:,0,:], dim=-1)
        d_traj = torch.sum(torch.linalg.norm(x_t[:,1:,:]-x_t[:,:-1,:],dim=-1),dim=-1)
        d_straight = torch.abs(d_traj-d_shortest)/(d_shortest+1e-6)
        return d_straight

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


    def dist_obst(self, kp_loc):
        ''' 
            metric for obstacle avoidance 
            kp_loc: batch x time x joint x key-point x coordinate
        '''
        batch_size=kp_loc.shape[0]
        d_collisions = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
            dist2centre = torch.linalg.norm(kp_loc-self.x_obst[i].view(1,1,1,1,-1), dim=-1).view(batch_size,-1)
            dist_in = (dist2centre/(self.r_obst[i]+self.margin))
            dist_in = (1-dist_in)*(dist_in<1)
            d_collisions += torch.sum(dist_in,dim=-1)
        return d_collisions
        
    def cost_ik_2(self,x):
        ''' 
        Cost for inverse kinematics. 
        task-param: ee position and orientation
        decision variables: joint angles
        '''
        x = x.to(self.device)
        x_goal = x[:,:2] # desired position of the end-effector
        theta_ee_goal = x[:,2] # desired orientation of ee
        theta = x[:,3:] # joint angles
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        
        d_goal = self.dist_goal(x_goal, ee_loc)
        d_obst = self.dist_obst(kp_loc[:,None,:,:,:])
        d_orient = self.dist_orient(theta_ee_goal, theta_ee)

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient
        c_return = torch.cat((c_total.view(-1,1), d_goal.view(-1,1), d_obst.view(-1,1),d_orient.view(-1,1)),dim=1)
        return c_return

    def cost_ik(self,x):
        ''' 
        Cost for inverse kinematics. 
        task-param: ee position 
        decision variables: joint angles
        '''
        x = x.to(self.device)
        x_goal = x[:,:2] # desired position of the end-effector
        theta = x[:,2:] # joint angles
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        
        d_goal = self.dist_goal(x_goal, ee_loc)
        d_obst = self.dist_obst(kp_loc[:,None,:,:,:])

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst
        c_return = torch.cat((c_total.view(-1,1), d_goal.view(-1,1), d_obst.view(-1,1)),dim=1)
        return c_return

    def cost_goal(self,x):
        ''' compute the IK cost without obstacles '''
        x_goal = x[:,:2] # desired position of the end-effector
        theta = x[:,2:] # joint angles
        kp_loc, joint_loc, ee_loc, theta_ee = self.robot.forward_kin(theta) # get position of key-points and the end-effector
        d_goal = self.dist_goal(x_goal, ee_loc)
        return d_goal    
    

    def cost_obst(self,x,theta_0):
        ''' 
        Obstacle avoidance cost for motion planning from a fixed joint configuration (theta_0) 
        to a final configuration (theta_1)
        task-param: position of end-effector 
        '''
        batch_size = x.shape[0]
        theta_1 = x[:,:self.n_joints] #final configuration
        w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_1,w)#batchxtimexjoint_angle
        T01 = theta_t.shape[1]
        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        d_obst = self.dist_obst(kp_loc_t_01)
        
        return d_obst

    def cost_control(self,x,theta_0):
        ''' 
        Control cost for motion planning from a fixed joint configuration (theta_0) 
        to a final configuration (theta_1)
        task-param: position of end-effector 
        '''
        batch_size = x.shape[0]
        theta_1 = x[:,:self.n_joints] #final configuration
        w = x[:,self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
        if theta_0.shape[0]!=batch_size:
            theta_0 = theta_0.view(1,-1).repeat(batch_size,1)

        
        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_1,w)#batchxtimexjoint_angle

        d_control = self.dist_traj(theta_t)

        return d_control

    def cost_j2p(self,x,theta_0):
        ''' 
        Cost for motion planning from a fixed joint configuration (theta_0) 
        to a given target point for end-effector (x_goal) 
        task-param: position of end-effector 
        '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal = x[:,:2] # desired position of ee
        theta_1 = x[:,2:2+self.n_joints] #final configuration
        w = x[:,2+self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
       
        theta_t = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w)#batchxtimexjoint_angle
        T01 = theta_t.shape[1]
        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
     
        x_actual_1 = ee_loc_t_01[:,-1,:]

        d_goal = self.dist_goal(x_goal,x_actual_1)
        d_obst = self.dist_obst(kp_loc_t_01)
        d_ee = self.dist_traj(ee_loc_t_01)
        d_control = self.dist_traj(theta_t)

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
        
        return c_return 


    def cost_j2p_2(self,x,theta_0):
        ''' 
        Cost for motion planning from a fixed joint configuration (theta_0) 
        to a given target point for end-effector (x_goal) 
        task-param: position of end-effector and the orientation
        '''

        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal = x[:,:2] # desired position of ee
        theta_ee_goal = x[:,2]
        theta_1 = x[:,3:3+self.n_joints] #final configuration
        w = x[:,3+self.n_joints:] # weights of the basis function for motion between theta_0 to theta_1
       
        theta_t = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w)#batchxtimexjoint_angle
        T01 = theta_t.shape[1]
        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
     
        theta_ee_1 = theta_ee_t_01.view(batch_size,T01)[:,-1]

        x_actual_1 = e_loc_t_01[:,-1,:]

        d_goal = self.dist_goal(x_goal,x_actual_1)
        d_obst = self.dist_obst(kp_loc_t)
        d_orient = self.dist_orient(theta_ee_goal, theta_ee_1)
        d_ee = self.dist_traj(ee_loc_t)
        d_control = self.dist_traj(theta_t)

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
        
        return c_return 

    def cost_j2p2j(self,x,theta_0, theta_2): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to
            a fixed final configuration (theta_2) via a target point for end-effector (x_goal) 
            task-param: position of end-effector
         '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal = x[:,:2] # desired position of ee (via-point)

        theta_1 = x[:,2:2+self.n_joints] # intermediate configuration at the via-point
        w = x[:,2+self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/2)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/2):] # for motion between theta_1 to theta_2

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2.view(1,-1).repeat(batch_size,1),w12)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)


        x_actual_1 = ee_loc_t_01[:,-1,:]
        
        d_goal = self.dist_goal(x_goal,x_actual_1) # for via point
        d_obst = 0.5*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12))
        d_ee = 0.5*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12))
        d_control = 0.5*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12))


        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return


    def cost_j2p2j_2(self,x,theta_0, theta_2): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to
            a fixed final configuration (theta_2) via a target point for end-effector (x_goal) 
            task-param: position and orientation of end-effector
         '''

        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal = x[:,:2] # desired position of ee (via-point)
        theta_ee_goal = x[:,2]

        theta_1 = x[:,3:3+self.n_joints] # intermediate configuration at the via-point
        w = x[:,3+self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/2)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/2):] # for motion between theta_1 to theta_2

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2.view(1,-1).repeat(batch_size,1),w12)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)


        theta_ee_1 = theta_ee_t_01.view(batch_size,T01)[:,-1]
        x_actual_1 = ee_loc_t_01[:,-1,:]
        
        d_goal = self.dist_goal(x_goal,x_actual_1) # for via point
        d_obst = 0.5*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12))
        d_ee = 0.5*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12))
        d_control = 0.5*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12))
        d_orient = self.dist_orient(theta_ee_goal, theta_ee_1)


        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return

    def cost_j2p2p(self,x,theta_0): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to
            a fixed final configuration (theta_2) via a target point for end-effector (x_goal) 
            task-param: position of ee at  via points
        '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal_1 = x[:,:2] # desired position of ee at via point
        x_goal_2 = x[:,2:4] # desired position of ee at the final point
        theta_1 = x[:,4:4+self.n_joints] # via configuration
        theta_2 = x[:,4+self.n_joints:4+2*self.n_joints] # via configuration
        w = x[:,4+2*self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/2)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/2):] # for motion between theta_1 to theta_2

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2,w12)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates
    
        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)

    
        x_actual_1 = ee_loc_t_01[:,-1,:]
        x_actual_2 = ee_loc_t_12[:,-1,:]


        d_goal = 0.5*(self.dist_goal(x_goal_1, x_actual_1)+self.dist_goal(x_goal_2,x_actual_2)) # for via point
        d_obst = 0.5*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12))
        d_ee = 0.5*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12))
        d_control = 0.5*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12))

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return



    def cost_j2p2p_2(self,x,theta_0): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to
            a fixed final configuration (theta_2) via a target point for end-effector (x_goal) 
            task-param: position and orientation of ee at  via points
        '''

        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal_1 = x[:,:2] # desired position of ee at via point
        x_goal_2 = x[:,2:4] # desired position of ee at the final point
        theta_ee_goal_1 = x[:,4]
        theta_ee_goal_1 = x[:,5]
        theta_1 = x[:,5:5+self.n_joints] # via configuration
        theta_2 = x[:,5+self.n_joints:5+2*self.n_joints] # via configuration
        w = x[:,5+2*self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/2)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/2):] # for motion between theta_1 to theta_2

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2,w12)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates
    
        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)
 
        theta_ee_1 = theta_ee_t_01.view(batch_size,T01)[:,-1]
        theta_ee_2 = theta_ee_t_12.view(batch_size,T12)[:,-1]

    
        x_actual_1 = ee_loc_t_01[:,-1,:]
        x_actual_2 = ee_loc_t_12[:,-1,:]


        d_goal = 0.5*(self.dist_goal(x_goal_1, x_actual_1)+self.dist_goal(x_goal_2,x_actual_2)) # for via point
        d_obst = 0.5*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12))
        d_ee = 0.5*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12))
        d_control = 0.5*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12))
        d_orient = 0.5*(self.dist_orient(theta_ee_goal_1, theta_ee_1),self.dist_orient(theta_ee_goal_2, theta_ee_2))


        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return

    def cost_j2p2p2j(self,x,theta_0, theta_3): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to the final configuration (theta_3)
            but via two intermediary points for the end-effector: x_goal_1 and x_goal_2 
            task-param: position of ee at the via points
         '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal_1 = x[:,:2] # desired position of ee at via point
        x_goal_2 = x[:,2:4] # desired position of ee at the final point
        theta_1 = x[:,4:4+self.n_joints] # via configuration
        theta_2 = x[:,4+self.n_joints:4+2*self.n_joints] # via configuration
        theta_1 = x[:,4:4+self.n_joints] # via configuration
        theta_2 = x[:,4+self.n_joints:4+2*self.n_joints] # via configuration
        w = x[:,4+2*self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/3)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/3):2*int(w.shape[-1]/3)] # for motion between theta_1 to theta_2
        w23 = w[:,2*int(w.shape[-1]/3):] # for motion between theta_2 to theta_0

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2,w12)#batchxtimexjoint_angle
        theta_t_23 = self.p2p.gen_traj_p2p(theta_2,theta_3.view(1,-1).repeat(batch_size,1),w23)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]
        T23 = theta_t_23.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        kp_loc_t_23, joint_loc_t_23, ee_loc_t_23, theta_ee_t_23 = self.robot.forward_kin(theta_t_23.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        kp_loc_t_23 = kp_loc_t_23.view(batch_size,T23,*kp_loc_t_23.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)
        ee_loc_t_23 = ee_loc_t_23.view(batch_size,T23,-1)
        
   
        x_actual_1 = ee_loc_t_01[:,-1,:]
        x_actual_2 = ee_loc_t_12[:,-1,:]

    
        d_goal = 0.5*(self.dist_goal(x_goal_1,x_actual_1)+self.dist_goal(x_goal_2,x_actual_2)) # for via point
        d_obst = (1./3.)*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12)+self.dist_obst(kp_loc_t_23))
        d_ee = (1./3.)*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12)+self.dist_traj(ee_loc_t_23))
        d_control = (1./3.)*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12)+self.dist_traj(theta_t_23))

        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return




    def cost_j2p2p2j_2(self,x,theta_0, theta_3): 
        ''' 
            Cost for motion planning from a fixed joint configuration (theta_0) to the final configuration (theta_3)
            but via two intermediary points for the end-effector: x_goal_1 and x_goal_2 
            task-param: position and orientation of ee at the via points
         '''
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_goal_1 = x[:,:2] # desired position of ee at via point
        x_goal_2 = x[:,2:4] # desired position of ee at the final point
        theta_ee_goal_1 = x[:,4]
        theta_ee_goal_1 = x[:,5]
        theta_1 = x[:,5:5+self.n_joints] # via configuration
        theta_2 = x[:,5+self.n_joints:5+2*self.n_joints] # via configuration
        theta_1 = x[:,5:5+self.n_joints] # via configuration
        theta_2 = x[:,5+self.n_joints:5+2*self.n_joints] # via configuration
        w = x[:,5+2*self.n_joints:] # weights of the basis function
        w01 = w[:,:int(w.shape[-1]/3)] # weights for the first part of the motion: theta_0 to theta_1
        w12 = w[:,int(w.shape[-1]/3):2*int(w.shape[-1]/3)] # for motion between theta_1 to theta_2
        w23 = w[:,2*int(w.shape[-1]/3):] # for motion between theta_2 to theta_0

        theta_t_01 = self.p2p.gen_traj_p2p(theta_0.view(1,-1).repeat(batch_size,1),theta_1,w01)#batchxtimexjoint_angle
        theta_t_12 = self.p2p.gen_traj_p2p(theta_1,theta_2,w12)#batchxtimexjoint_angle
        theta_t_23 = self.p2p.gen_traj_p2p(theta_2,theta_3.view(1,-1).repeat(batch_size,1),w23)#batchxtimexjoint_angle
         
        T01 = theta_t_01.shape[1]
        T12 = theta_t_12.shape[1]
        T23 = theta_t_23.shape[1]

        kp_loc_t_01, joint_loc_t_01, ee_loc_t_01, theta_ee_t_01 = self.robot.forward_kin(theta_t_01.view(-1,self.n_joints))
        kp_loc_t_12, joint_loc_t_12, ee_loc_t_12, theta_ee_t_12 = self.robot.forward_kin(theta_t_12.view(-1,self.n_joints))
        kp_loc_t_23, joint_loc_t_23, ee_loc_t_23, theta_ee_t_23 = self.robot.forward_kin(theta_t_23.view(-1,self.n_joints))
        #kp_loc_t: (batch x time) x joint x kp x coordinates
        #ee_loc_t: (batch x time) x coordinates

        kp_loc_t_01 = kp_loc_t_01.view(batch_size,T01,*kp_loc_t_01.shape[1:])
        kp_loc_t_12 = kp_loc_t_12.view(batch_size,T12,*kp_loc_t_12.shape[1:])
        kp_loc_t_23 = kp_loc_t_23.view(batch_size,T23,*kp_loc_t_23.shape[1:])
        ee_loc_t_01 = ee_loc_t_01.view(batch_size,T01,-1)
        ee_loc_t_12 = ee_loc_t_12.view(batch_size,T12,-1)
        ee_loc_t_23 = ee_loc_t_23.view(batch_size,T23,-1)
        
        theta_ee_1 = theta_ee_t_01.view(batch_size,T01)[:,-1]
        theta_ee_2 = theta_ee_t_12.view(batch_size,T12)[:,-1]

        x_actual_1 = ee_loc_t_01[:,-1,:]
        x_actual_2 = ee_loc_t_12[:,-1,:]

    
        d_goal = 0.5*(self.dist_goal(x_goal_1,x_actual_1)+self.dist_goal(x_goal_2,x_actual_2)) # for via point
        d_obst = (1./3.)*(self.dist_obst(kp_loc_t_01)+self.dist_obst(kp_loc_t_12)+self.dist_obst(kp_loc_t_23))
        d_ee = (1./3.)*(self.dist_traj(ee_loc_t_01)+self.dist_traj(ee_loc_t_12)+self.dist_traj(ee_loc_t_23))
        d_control = (1./3.)*(self.dist_traj(theta_t_01)+self.dist_traj(theta_t_12)+self.dist_traj(theta_t_23))
        d_orient = 0.5*(self.dist_orient(theta_ee_goal_1, theta_ee_1),self.dist_orient(theta_ee_goal_2, theta_ee_2))



        c_total = self.w_goal*d_goal/self.b_goal+self.w_obst*d_obst/self.b_obst+self.w_orient*d_orient/self.b_orient +self.w_ee*d_ee/self.b_ee+self.w_control*d_control/self.b_control # total cost 

        c_return = torch.cat((c_total.view(-1,1),d_goal.view(-1,1),
            d_obst.view(-1,1),d_orient.view(-1,1), d_ee.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis

        return c_return


    def cost_j2j(self,x, theta_0, theta_f):
        ''' Given  (init_joint_angle, final_joint_angle, basis_weights) define the cost for reaching task'''
        batch_size = x.shape[0]
        x = x.to(self.device)
        theta_0 = theta_0.repeat(batch_size,1)
        theta_f = theta_f.repeat(batch_size,1)
        w = 1*x # weights
        theta_t = self.p2p.gen_traj_p2p(theta_0,theta_f,w) #joint angles: batch x time x joint
        T = theta_t.shape[1]

        key_loc_t,joint_loc_t, ee_loc_t, theta_ee_t = self.robot.forward_kin(theta_t.view(-1,self.n_joints)) # (batchxtime) x joint x key x positions

        key_loc_t = key_loc_t.view(batch_size,T,*key_loc_t.shape[1:])
        ee_loc_t = ee_loc_t.view(batch_size,T,-1)

        # Cost due to obstacle
        d_obst = self.dist_obst(key_loc_t)

        
        # Cost on end-effector traj (aim to keep it straight)
        d_control = self.dist_control(theta_t)
    
        c_total =  self.w_obst*d_obst/self.b_obst + self.w_control*d_control/self.b_control

        c_return = torch.cat((c_total.view(-1,1), d_obst.view(-1,1),d_control.view(-1,1)),dim=-1) # for analysis
 

        return c_return



class PointMassCost:
    ''' Point-to-Point Motion of a point mass (motion planning) '''
    def __init__(self, p2p_motion, x_obst=[], r_obst=[], margin=0.01,
     w_obst=1., w_straight=1.,b_obst=1., b_straight=1., device='cpu'):
        '''
            p2p_motion: object of class Point2PointMotion (for generating trajectory)
            x_obst: a list of center of spherical obstacles in the plane
            r_obst: a list containing radius of the spherical obstscles
            margin: safety distance from the edge of the obstacle
            w_obst: weight for obstacle avoidance
            w_straight: weight for keeping the trajectory as staight as possible
            b_* are nominal values for the corresponding distances (see the cost function below)
        '''
        self.device=device
        self.x_obst=x_obst # centers of the obstacles/spheres
        self.r_obst=r_obst # radius of the spheres/spheres
        self.margin=margin # safety margin from the surface (considers the width of the links)
        self.b_obst=b_obst;self.b_straight=b_straight 
        self.w_obst=w_obst; self.w_straight=w_straight 
        self.p2p = p2p_motion

    def dist_obst(self, x_t):
        ''' A matric for obstacle collision '''
        batch_size=x_t.shape[0]
        d_collisions = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
            dist2centre = torch.linalg.norm(x_t-self.x_obst[i].view(1,1,-1), dim=-1).view(batch_size,-1)
            dist_in = (dist2centre/(self.r_obst[i]+self.margin))
            dist_in = (1-dist_in)*(dist_in<1)
            d_collisions += torch.sum(dist_in,dim=-1)
        return d_collisions

    def dist_straight(self,x_t):
        ''' metric for how straight is a trajectory (to seek minimal length) '''
        d_shortest = torch.linalg.norm(x_t[:,-1,:]-x_t[:,0,:], dim=-1)
        d_traj = torch.sum(torch.linalg.norm(x_t[:,1:,:]-x_t[:,:-1,:],dim=-1),dim=-1)
        d_straight = torch.abs(d_traj-d_shortest)/(d_shortest+1e-6)
        return d_straight
        
    def cost_motion(self,x, x0):
        '''move from left end of the plane to the right end '''
        x=x.to(self.device)
        x0 = x0.view(1,2).repeat(x.shape[0],1) # initial point
        xf = torch.cat((0.95+0*x[:,0],x[:,0]),dim=-1)
        w = x[:,1:] # weights of the basis function

        x_t = self.p2p.gen_traj_p2p(x0,xf,w)
        d_obst = self.dist_obst(x_t)
        d_straight = self.dist_straight(x_t)

        c_total = (self.w_straight*d_straight/self.b_straight+self.w_obst*d_obst/self.b_obst)
        c_return = torch.cat((c_total.view(-1,1),d_obst.view(-1,1),d_straight.view(-1,1)),dim=1)
        return c_return


