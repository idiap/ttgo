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


import numpy as np
import pybullet as p
import time

def plot_traj(xs, obj_id, dt = 0.01):
    for x in xs:
        p.resetBasePositionAndOrientation(obj_id, x[:3], (0,0,0,1))
        time.sleep(dt)
        
def create_obstacles(obstacles, colors = None):
    obstacle_ids = []
    if colors is None: 
        colors = [[1.,1.,0.,1.],[0.,1.,1.,1.], [1.,0.,1.,1.], [.7,.7,.7,1.], [.3,.3,0.3,1.]]
    for i,obstacle in enumerate(obstacles):
        if 'obs_type' in list(obstacle.keys()):
            obs_type = obstacle['obs_type']
        else:
            obs_type = p.GEOM_SPHERE
        
        if obs_type == p.GEOM_SPHERE:
            _,_,obs_id = create_primitives(obs_type, radius =obstacle['rad'], basePosition=np.zeros(3), rgbaColor=colors[i])
        elif obs_type == p.GEOM_BOX:
            _,_,obs_id = create_primitives(obs_type, halfExtents=obstacle['halfExtents'], basePosition=np.zeros(3), rgbaColor=colors[i])
        elif obs_type == p.GEOM_CAPSULE:
            _,_,obs_id = create_primitives(obs_type, radius=obstacle['rad'], length=obstacle['length'], basePosition=np.zeros(3), rgbaColor=colors[i])
            
        p.resetBasePositionAndOrientation(obs_id, obstacle['pos'], (0,0,0,1))    
        obstacle['obs_id'] = obs_id
        obstacle_ids.append(obs_id)
    return obstacle_ids
        
def init_pybullet(x0, x_target, obstacles, model='pointmass', globalScaling=0.3, colors = None):
    '''
    Return: obj_id, init_id, target_id, border_id, obstacle_ids
    '''
    p.resetSimulation()

    if model == 'pointmass':
        _,_,obj_id = create_primitives(rgbaColor=[0,0,1,1],radius = 0.04)
    elif model == 'quadcopter':
        obj_id = p.loadURDF("../data/urdf/quadrotor.urdf",[0,0,0],p.getQuaternionFromEuler([0,0,0]), globalScaling=globalScaling)
        
    _,_,init_id = create_primitives(radius = 0.04, rgbaColor=[1,0,0,1.])
    _,_,target_id = create_primitives(radius = 0.04, rgbaColor=[0,1,0,1.])
    _,_,border_id = create_primitives(rgbaColor=[0,0,1,0.1], shapeType=p.GEOM_BOX, halfExtents=[1.,1.,1.])
    obstacle_ids = create_obstacles(obstacles, colors)
    
    p.resetBasePositionAndOrientation(init_id, x0[:3], (0,0,0,1))
    p.resetBasePositionAndOrientation(target_id, x_target[:3], (0,0,0,1))
    return obj_id, init_id, target_id, border_id, obstacle_ids

def create_primitives(shapeType=2, rgbaColor=[1, 1, 0, 1], pos = [0, 0, 0], radius = 1, length = 2, halfExtents = [0.5, 0.5, 0.5], baseMass=1, basePosition = [0,0,0]):
    visualShapeId = p.createVisualShape(shapeType=shapeType, rgbaColor=rgbaColor, visualFramePosition=pos, radius=radius, length=length, halfExtents = halfExtents)
    collisionShapeId = p.createCollisionShape(shapeType=shapeType, collisionFramePosition=pos, radius=radius, height=length, halfExtents = halfExtents)
    bodyId = p.createMultiBody(baseMass=baseMass,
                      baseInertialFramePosition=[0, 0, 0],
                      baseVisualShapeIndex=visualShapeId,
                      baseCollisionShapeIndex=collisionShapeId,    
                      basePosition=basePosition,
                      useMaximalCoordinates=True)
    return visualShapeId, collisionShapeId, bodyId

import pybullet as p
import numpy as np
import pinocchio as pin
# import transforms3d
import time

#pb_joint_indices = np.array(list(np.arange(45,51)) + list(np.arange(52,58)) + [0, 1] + \
#                list(np.arange(11,18)) +[21] + list(np.arange(28,35)) + [38,3,4]).astype(int)
# pb_joint_indices = np.array([45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57,  0,  1, 11, 12, 13,
#        14, 15, 16, 17, 28, 29, 30, 31, 32, 33, 34])
tf_joint_indices = np.array([ 0,  1, 28, 29, 30, 31, 32, 33, 34, 11, 12, 13, 14, 15, 16, 17, 52, 53, 54, 55, 56, 57, 45, 46, 47, 48, 49, 50])#r 'right, left'
pb_joint_names_complete = ['torso_1_joint', 'torso_2_joint', 'imu_joint', 
                           'head_1_joint', 'head_2_joint', 'rgbd_joint', 
                           'rgbd_optical_joint', 'rgbd_depth_joint', 
                           'rgbd_depth_optical_joint', 'rgbd_rgb_joint', 
                           'rgbd_rgb_optical_joint', 'arm_left_1_joint', 
                           'arm_left_2_joint', 'arm_left_3_joint',
                           'arm_left_4_joint', 'arm_left_5_joint', 
                           'arm_left_6_joint', 'arm_left_7_joint', 
                           'wrist_left_ft_joint', 'wrist_left_tool_joint', 
                           'gripper_left_base_link_joint', 'gripper_left_joint', 
                           'gripper_left_inner_double_joint', 
                           'gripper_left_fingertip_1_joint', 
                           'gripper_left_fingertip_2_joint', 
                           'gripper_left_motor_single_joint',
                           'gripper_left_inner_single_joint', 
                           'gripper_left_fingertip_3_joint', 'arm_right_1_joint',
                           'arm_right_2_joint', 'arm_right_3_joint', 
                           'arm_right_4_joint', 'arm_right_5_joint', 
                           'arm_right_6_joint', 'arm_right_7_joint', 
                           'wrist_right_ft_joint', 'wrist_right_tool_joint',
                           'gripper_right_base_link_joint', 'gripper_right_joint', 
                           'gripper_right_inner_double_joint', 
                           'gripper_right_fingertip_1_joint', 
                           'gripper_right_fingertip_2_joint',
                           'gripper_right_motor_single_joint', 
                           'gripper_right_inner_single_joint', 
                           'gripper_right_fingertip_3_joint', 
                           'leg_left_1_joint', 'leg_left_2_joint', 
                           'leg_left_3_joint', 'leg_left_4_joint',
                           'leg_left_5_joint', 'leg_left_6_joint', 
                           'leg_left_sole_fix_joint', 'leg_right_1_joint', 
                           'leg_right_2_joint', 'leg_right_3_joint', 
                           'leg_right_4_joint', 'leg_right_5_joint', 
                           'leg_right_6_joint', 'leg_right_sole_fix_joint']

q0Complete =  np.array([ 0.     , 0.    ,  1.019272,  0.      ,  0.      ,  0.,
        1., 0.00000e+00 , 0.00000e+00, -4.11354e-01 , 8.59395e-01 ,-4.48041e-01,
  -1.70800e-03 , 0.00000e+00 , 0.00000e+00 ,-4.11354e-01 , 8.59395e-01,
  -4.48041e-01 ,-1.70800e-03 , 0.00000e+00 , 6.76100e-03,  2.58470e-01,
   1.73046e-01 ,-2.00000e-04 ,-5.25366e-01 , 0.00000e+00 , 0.00000e+00,
   1.00000e-01 , 1.00000e-01 ,-1.73046e-01 , 2.00000e-04,
  -5.25366e-01 , 0.00000e+00 , 0.00000e+00 , 1.00000e-01 ])


def get_pb_config(q):
    """
    Convert tf's format of 'joint angles + base position' to
    'base_pos + base_ori + joint_angles' according to pybullet order
    """
    joint_angles = q[:28]
    #qnew = np.concatenate([q[28:31], euler2quat(q[-3:]),
    #qnew = np.concatenate([np.array([0,0,q[28]]), euler2quat(q[-3:]),
    qnew = np.concatenate([q[28:31], euler2quat(np.array([0,0,0])),
                           joint_angles[-6:], joint_angles[-12:-6], 
                          joint_angles[:2], joint_angles[9:16], 
                          joint_angles[2:9]])
    return qnew

def get_tf_config(q):
    """
    Convert 'base_pos + base_ori + joint_angles' according to pybullet order
    to tf's format of 'joint angles + base position'
    
    """
    joint_angles = q[7:]
    qnew = np.concatenate([joint_angles[12:14], joint_angles[-7:], 
                          joint_angles[-14:-7], joint_angles[6:12], 
                          joint_angles[:6], q[:3]])
    return qnew


def normalize(x):
    return x/np.linalg.norm(x)
        
def set_q(robot_id, joint_indices, q, set_base = False):
    if set_base:
        localInertiaPos = np.array(p.getDynamicsInfo(robot_id,-1)[3])
        q_root = q[0:7]
        ori = q_root[3:]
        Rbase = np.array(p.getMatrixFromQuaternion(ori)).reshape(3,3)
        shift_base = Rbase.dot(localInertiaPos)
        pos = q_root[:3]+shift_base
        p.resetBasePositionAndOrientation(robot_id,pos,ori)
        q_joint = q[7:]
    else:
        q_joint = q
    
    #set joint angles
    for i in range(len(q_joint)):
        p.resetJointState(robot_id, joint_indices[i], q_joint[i])


def vis_traj(qs, vis_func, dt=0.1):
    for q in qs:
        vis_func(q)
        time.sleep(dt)


def get_joint_limits(robot_id, indices):
    lower_limits = []
    upper_limits = []
    for i in indices:
        info = p.getJointInfo(robot_id, i)
        lower_limits += [info[8]]
        upper_limits += [info[9]]
    limits = np.vstack([lower_limits, upper_limits])
    return limits

def computeJacobian(rmodel,rdata,ee_frame_id,q):
    pin.forwardKinematics(rmodel,rdata,q)
    pin.updateFramePlacements(rmodel,rdata)
    pin.computeJointJacobians(rmodel, rdata, q)
    J = pin.getFrameJacobian(rmodel, rdata,ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J[:,:7]

def computePose(rmodel, rdata, ee_frame_id, q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateFramePlacements(rmodel, rdata)
    pos, ori = rdata.oMf[ee_frame_id].translation, rdata.oMf[ee_frame_id].rotation
    return pos,ori
    
def check_joint_limits(q, joint_limits):
    """
    Return True if within the limit
    """
    upper_check = False in ((q-joint_limits[0]) > 0)
    lower_check = False in ((joint_limits[1]-q) > 0)
    if upper_check or lower_check:
        return False
    else:
        return True
    
def calc_dist_limit(q, joint_limits):
    lower_error = joint_limits[0]-q
    lower_check = (lower_error > 0)
    lower_error = lower_error*lower_check
    upper_error = q-joint_limits[1]
    upper_check = (upper_error > 0)
    upper_error = upper_error*upper_check
    error = lower_error-upper_error
    return error
    
def mat2euler(rot, axes = 'rzyx'):
    return np.array(transforms3d.euler.mat2euler(rot, axes = axes))

def euler2quat(rpy, axes='sxyz'):
    #euler sxyz: used by Manu's codes
    return rectify_quat(transforms3d.euler.euler2quat(*rpy, axes=axes))

def rectify_quat(quat):
    #transform from transforms3d format (w,xyz) to pybullet and pinocchio (xyz, w)
    quat_new = np.concatenate([quat[1:], quat[0:1]])
    return quat_new

def mat2w(rot):
    rot_aa = pin.AngleAxis(rot)
    return rot_aa.angle*rot_aa.axis

def w2quat(q):
    angle = np.linalg.norm(q)
    if abs(angle) < 1e-7:
        ax = np.array([1,0,0])
    else:
        ax, angle = normalize(q), np.linalg.norm(q)
    w = p.getQuaternionFromAxisAngle(ax, angle)
    return np.array(w)

def quat2w(q):
    ax, angle = p.getAxisAngleFromQuaternion(q)
    return np.array(ax)*angle

def w2mat(w):
    angle = np.linalg.norm(w)
    if abs(angle) < 1e-7:
        ax = np.array([1,0,0])
    else:
        ax, angle = w/angle, angle
    R = pin.AngleAxis.toRotationMatrix(pin.AngleAxis(angle, ax))
    return R

def get_link_base(robot_id, frame_id):
    '''
    Obtain the coordinate of the link frame, according to the convention of pinocchio (at the link origin, 
    instead of at the COM as in pybullet)
    '''
    p1 = np.array(p.getLinkState(robot_id,frame_id)[0])
    ori1 = np.array(p.getLinkState(robot_id,frame_id)[1])
    R1 = np.array(p.getMatrixFromQuaternion(ori1)).reshape(3,3)
    p2 = np.array(p.getLinkState(robot_id,frame_id)[2])
    return  p1 - R1.dot(p2), ori1
