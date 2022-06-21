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

class PlanarManipulator:
    def __init__(self, n_joints=2, link_lengths=[], max_theta=torch.pi/1.1, n_kp=3, device="cpu"):
        ''' 
            n_joints: number of joints in the planar manipulator
            max_theta: max joint angle (same for al joints)
            link_lengths: a list containing length of each link
            n_kp: number of key-points on each link (for collision check)
        '''
        self.device = device
        self.n_joints = n_joints
        if link_lengths is None:
            self.link_lengths = torch.tensor([1./n_joints]*n_joints).to(self.device)
        else:
            self.link_lengths = link_lengths.to(device) 
        assert n_joints== link_lengths.shape[0], 'The length of the list containing link_lengths should match n_joints'
        
        self.max_config = torch.tensor([max_theta]*n_joints).to(self.device)
        self.min_config = -1*self.max_config
        self.theta_max = self.max_config
        self.theta_min = self.min_config

        self.n_kp = n_kp
        assert self.n_kp>=2, 'number of key points should be at least two'
        self.key_points = torch.empty(self.n_joints,self.n_kp).to(device)
        for i in range(n_joints):
            self.key_points[i] = torch.arange(0,self.n_kp)/(self.n_kp-1)


    # forward kinematics
    def forward_kin(self, q):
        ''' Given a batch of joint angles find the position of all the key-points and the end-effector  '''
        batch_size = q.shape[0]
        q = torch.clip(q,self.min_config, self.max_config)
        q_cumsum = torch.zeros(batch_size,self.n_joints).to(self.device)
        for joint in range(self.n_joints):
            q_cumsum[:,joint] = torch.sum(q[:,:joint+1],dim=1)

        cq = torch.cos(q_cumsum).view(batch_size,-1,1)
        sq = torch.sin(q_cumsum).view(batch_size,-1,1)
        cq_sq = torch.cat((cq,sq),dim=2)

        joint_loc = torch.zeros((batch_size, self.n_joints+1, 2)).to(self.device)
        key_loc = torch.empty((batch_size, self.n_joints,self.n_kp,2)).to(self.device)
        for i in range(self.n_joints):
            joint_loc[:,i+1,:] = joint_loc[:,i,:]+self.link_lengths[i]*cq_sq[:,i,:]
            key_loc[:,i,:,:] = joint_loc[:,i,:][:,None,:] + (joint_loc[:,i+1,:]-joint_loc[:,i,:])[:,None,:]*self.key_points[i].reshape(1,-1,1)
    
        end_loc = joint_loc[:,-1,:]
        # find the orientation of end-effector in range (0,2*pi)
        theta_orient = torch.fmod(q_cumsum[:,-1],2*torch.pi)
        theta_orient[theta_orient<0] = 2*torch.pi+theta_orient[theta_orient<0]

        return key_loc, joint_loc, end_loc, theta_orient # output shape: batch_size x (n_joints) x n_kp x 2, batch_size x 2






















