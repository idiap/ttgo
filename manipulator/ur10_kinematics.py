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

class Ur10Kinematics:

    def __init__(self, device="cpu", key_points_data=None):
        self.device = device
        self.n_joints = 6

        # joint limits
        self.theta_max = torch.tensor([ 2.*torch.pi]*self.n_joints).to(self.device)
        self.theta_min = torch.tensor([-2*torch.pi]*self.n_joints).to(self.device)
        self.max_config = self.theta_max.reshape(1,-1).to(device)
        self.min_config = self.theta_min.reshape(1,-1).to(device)
        
        # DH Params
        self.dh_a = torch.tensor([0,0.612,0.5723,0.,-0.,0, 0.]).to(self.device)
        self.dh_d = torch.tensor([0.1273,0,0.,0.163941,0.1157,0.0922,0.]).to(self.device) # 0.103 is added for the tip of the flange/ee
        self.dh_alpha = (torch.pi/2)*torch.tensor([-1,0,0,-1,1,0,0]).to(self.device)
        
        # key-points on the body of robot for collision check
        if key_points_data is None: # choose the joint locations as the key-points
            self.key_points = torch.empty(7,1,3).fill_(0.).to(self.device)
            self.key_points_weight = torch.empty(7,1,1).fill_(1./7.).to(self.device)
            self.key_points_margin = torch.empty(7,1,1).fill_(0.1).to(self.device)
        else:
            self.key_points = key_points_data[0].to(self.device) # 8xMx3 tensor
            self.key_points_weight = key_points_data[1].to(self.device) # 8xMx3 tensor
            self.key_points_margin = key_points_data[2].to(self.device) # 8xMX3 tensor

        # prepare trasfomation matrices
        ca = torch.cos(self.dh_alpha)
        sa = torch.sin(self.dh_alpha)
        Talpha = torch.eye(4,4).reshape(1,4,4).to(self.device)
        Talpha = Talpha.repeat(len(self.dh_alpha),1,1)
        Talpha[:,1,1] = ca
        Talpha[:,1,2] = -sa
        Talpha[:,2,1] = sa
        Talpha[:,2,2] = ca

        Ta = torch.eye(4,4).reshape(1,4,4).to(self.device)
        Ta = Ta.repeat(len(self.dh_a),1,1)
        Td = Ta.clone()
        Ta[:,0,-1] = self.dh_a
        Td[:,2,-1] = self.dh_d
        self.Td = Td.to(self.device)
        self.T_alpha_a = torch.einsum('ijk,ikl->ijl',Talpha, Ta).to(self.device)


    def forward_kin(self,q):
        self.T = self.computeTransformation(q) # 4D-tensor: batch x joint x (2D-Transformation-matrix) 
        self.key_positions = self.getKeyPosition() # position of key-points
        self.ee_position = self.key_positions[:,-1,-1,:] # end-effector position
        self.ee_orientation = self.getEndPose() 
        return  self.key_positions, self.ee_position, self.ee_orientation  

    def getKeyPosition(self):
        ''' returns position of all the key points given a batch of joint angles q'''
        key_position = self.TransformationToKeyPosition(self.T)  # 3D-tensor: batch x keys x position
        # Note: key_position[:,-1,:] gives end-effector position
        return key_position # 3D array: batch x joint x keys x position

    def getEndPoseEuler(self):
        ''' returns pose of end-effetor given a batch of joint angles q'''
        end_pose, end_R = self.TransformationToEndPoseEuler(self.T)  # 3D-tensor: batch x pose
        return end_pose, end_R # 2D array: batch x pose and 3D array of rotation matrices

    def getEndPose(self):
        ''' returns pose of end-effetor given a batch of joint angles q'''
        _, end_R = self.TransformationToEndPose(self.T)  # 3D-tensor: batch x pose
        return end_R # 2D array: batch x pose and 3D array of rotation matrices

    def TransformationToKeyPosition(self, T):
        ''' 
            Given a batch of the Transformation matrices (4D array: batch x joint x Tranform-matrix ) get the 3D positions of the key-points
            output key position (3D array): batch x keys x position
        '''
        x_joint = T[:, :, :3, -1].to(self.device) # 3D array of position of joints: batch x joint x position
        R_joint = T[:,:,:3,:3] # batch x joint x rot_matrix
        x_key = x_joint.view(x_joint.shape[0],x_joint.shape[1],1,x_joint.shape[2]) + torch.einsum('ijpr,jkr->ijkp',R_joint,self.key_points) # batch x joint x key x position

        return x_key # batch x joint x keyx x position



    def TransformationToEndPoseEuler(self,T):
        ''' 
            Given a batch of the Transformation matrices (4D array: batch x joint x Tranform-matrix ) get the 6D pose (position and euler angles)
            of the end-effector
            output pose (2D array): batch x pose
        '''
        x = T[:, -1, :3, -1].to(self.device) # 2D array of position of end-effector: batch x pose
        R = T[:, -1 , :3, :3].to(self.device) # 3D array containing rotation matrices: batch x rotation_matrix

        sy = torch.sqrt(R[:,0, 0] * R[:,0, 0] + R[:,1, 0] * R[:,1, 0])
        
        t1 = torch.stack((torch.atan2(R[:,2, 1], R[:,2, 2]), torch.atan2(-R[:,2, 0], sy), 
             torch.atan2(R[:,1, 0], R[:,0, 0])), dim=1)
        t2 = torch.stack((torch.atan2(-R[:,1, 2], R[:,1, 1]),torch.atan2(-R[:,2, 0], sy), 
             torch.zeros(T.shape[0]).to(self.device)), dim=1)

        sy = (sy.reshape(sy.shape[0],1)).repeat(1,3)
        singular = sy < 1e-6
        ts = torch.where(singular, t2, t1) # 2D array of orientation: batch x orientation
        return torch.cat((x, ts), dim=1), R 

    def TransformationToEndPose(self,T):
        ''' 
            Given a batch of the Transformation matrices (4D array: batch x joint x Tranform-matrix ) get the 3D position
            of the end-effector and the orientation matrix 
        '''
        x = T[:, -1, :3, -1].to(self.device) # 2D array of position of end-effector: batch x pose
        R = T[:, -1 , :3, :3].to(self.device) # 3D array containing rotation matrices: batch x rotation_matrix
        return x,R

    def computeTransformation(self,q):
        ''' Returns transformation matrices: batch x joint x tranformation-matrix  '''
        q = q.to(self.device)
    
        q = torch.cat((q, torch.tensor([0]*q.shape[0]).to(self.device).view(-1,1)),dim=1) 
        cq = torch.cos(q)
        sq = torch.sin(q)        

        T0 = torch.eye(4,4).reshape(1,1,4,4).to(self.device)
        Tz = T0.repeat(q.shape[0],q.shape[1],1,1)

        T = T0.repeat(q.shape[0],q.shape[1]+1,1,1)

        Tz[:,:,0,0] = cq
        Tz[:,:,1,1] = cq 
        Tz[:,:,0,1] = -1*sq 
        Tz[:,:,1,0] = sq

        T_joints = torch.einsum('ijk,nikl,ilm->nijm',self.Td, Tz, self.T_alpha_a)

        for i in range(1,q.shape[1]+1): 
            T[:,i,:,:] = torch.einsum('ijk,ikl->ijl', T[:,i-1,:,:].clone(), T_joints[:,i-1,:,:].clone()) 
        out_T = T[:,1:,:,:].clone()
        return  out_T



