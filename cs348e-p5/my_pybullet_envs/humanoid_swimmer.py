#  Copyright 2020 Stanford University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# DoF index, DoF (joint) Name, joint type (0 means hinge joint), joint lower and upper limits, child link of this joint
# (0, b'abdomen_z', 0) -1.0471975512 1.0471975512 b'link1_2'
# (1, b'abdomen_y', 0) -1.0471975512 1.0471975512 b'lwaist'
# (2, b'abdomen_x', 0) -1.0471975512 1.0471975512 b'pelvis'
# (3, b'right_hip_x', 0) -1.57 1.57 b'link1_7'
# (4, b'right_hip_z', 0) -1.0471975512 1.0471975512 b'link1_8'
# (5, b'right_hip_y', 0) -2.57 2.57 b'right_thigh'
# (6, b'right_knee', 0) 0.0 2.8 b'right_shin'
# (7, b'right_ankle_y', 0) -1.0471975512 1.5771975512 b'link1_13'
# (8, b'right_ankle_x', 0) -1.0471975512 1.0471975512 b'right_foot'
# (9, b'left_hip_x', 0) -1.57 1.57 b'link1_16'
# (10, b'left_hip_z', 0) -1.0471975512 1.0471975512 b'link1_17'
# (11, b'left_hip_y', 0) -2.57 2.57 b'left_thigh'
# (12, b'left_knee', 0) 0.0 2.8 b'left_shin'
# (13, b'left_ankle_y', 0) -1.0471975512 1.5771975512 b'link1_22'
# (14, b'left_ankle_x', 0) -1.0471975512 1.0471975512 b'left_foot'
# (15, b'right_shoulder1', 0) -2.57 3.14 b'link1_25'
# (16, b'right_shoulder2', 0) -3.14 2.57 b'right_upper_arm'
# (17, b'right_elbow', 0) 0.0 2.57 b'right_lower_arm'
# (18, b'right_wrist', 0) -1.0471975512 1.0471975512 b'right_hand'
# (19, b'left_shoulder1', 0) -2.57 3.14 b'link1_30'
# (20, b'left_shoulder2', 0) -3.14 2.57 b'left_upper_arm'
# (21, b'left_elbow', 0) 0.0 2.57 b'left_lower_arm'
# (22, b'left_wrist', 0) -1.0471975512 1.0471975512 b'left_hand'
# (23, b'jointfix_head', 4) 0.0 -1.0 b'head'


import pybullet_utils.bullet_client as bc
import time
import gym, gym.utils.seeding
import numpy as np
import math
from . import rigidBodySento as rb

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class HumanoidSwimmer:
    def __init__(self,
                 init_noise=True,
                 time_step=1. / 480,
                 np_random=None,
                 using_torque_ctrl=False
                 ):

        self.init_noise = init_noise
        self._ts = time_step
        self.np_random = np_random
        self.using_torque_ctrl = using_torque_ctrl

        self.base_init_pos = np.array([0., 0, 1.3])  # starting position in pool
        self.base_init_euler = np.array([0., 0, 0])  # starting orientation

        self.max_forces = [100, 200, 200] + ([100, 100, 250, 250] + [250, 100]) * 2 \
                          + [150, 150, 120, 80] * 2  # joint torque limits

        self.n_dofs = 23
        assert len(self.max_forces) == self.n_dofs

        self.human_density = 1000

        self._p = None  # bullet session to connect to
        self.human_id = -2  # bullet id for the loaded humanoid, to be overwritten
        self.tar_q = None  # if using position control, the current target joint angles vector
        self.delta_tar_q_scale = 0.03  # scale the raw action from neural network to make tar_q change slower
        self.torque = None  # if using torque control, the current torque vector to apply

        self.ll = None  # stores joint lower limits
        self.ul = None  # stores joint upper limits

    def reset(
            self,
            bullet_client
    ):
        self._p = bullet_client
        self.human_id = self._p.loadURDF(os.path.join(currentdir,
                                                      "assets/humanoid_box.urdf"),
                                         list(self.base_init_pos),
                                         self._p.getQuaternionFromEuler(list(self.base_init_euler)),
                                         flags=self._p.URDF_USE_SELF_COLLISION
                                         # | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
                                         ,
                                         useFixedBase=0)

        # self.print_all_joints_info()

        if self.init_noise:
            init_q = self.perturb([0.0] * self.n_dofs, 0.01)
            init_dq = self.perturb([0.0] * self.n_dofs, 0.1)
        else:
            init_q = self.perturb([0.0] * self.n_dofs, 0.0)
            init_dq = self.perturb([0.0] * self.n_dofs, 0.0)

        for ind in range(self.n_dofs):
            self._p.resetJointState(self.human_id, ind, init_q[ind], init_dq[ind])

        if self.using_torque_ctrl:
            self._p.setJointMotorControlArray(
                bodyIndex=self.human_id,
                jointIndices=range(self.n_dofs),
                controlMode=self._p.VELOCITY_CONTROL,
                forces=[0.0] * self.n_dofs)

        if self.using_torque_ctrl:
            self.torque = [0.0] * self.n_dofs
        else:
            self.tar_q = init_q

        self.ll = np.array([self._p.getJointInfo(self.human_id, i)[8] for i in range(self.n_dofs)])
        self.ul = np.array([self._p.getJointInfo(self.human_id, i)[9] for i in range(self.n_dofs)])

        self.calculate_and_set_mass_inertia()

    def get_bb_dimension_from_visual_shape(self, visual_shape_data):
        # depends on geometry type: for GEOM_BOX: extents, (different from create visual where halfExtent)

        shape = visual_shape_data[2]
        shape_dim_info = visual_shape_data[3]
        if shape == self._p.GEOM_BOX:
            return list(shape_dim_info)
        else:
            assert False and "not implemented"

    def calculate_and_set_mass_inertia(self):
        for dof in range(-1, self._p.getNumJoints(self.human_id)):
            self._p.changeDynamics(self.human_id, dof, mass=0.)
            self._p.changeDynamics(self.human_id, dof, localInertiaDiagonal=(0., 0, 0))

        visual_shapes = self._p.getVisualShapeData(self.human_id)

        for visual_shape in visual_shapes:
            _, link_id = visual_shape[0], visual_shape[1]

            dim = self.get_bb_dimension_from_visual_shape(visual_shape)
            dyn = self._p.getDynamicsInfo(self.human_id, link_id)
            mass = dyn[0]
            lid = list(dyn[2])
            dm = dim[0] * dim[1] * dim[2] * self.human_density
            mass += dm
            lid[0] += dm / 12.0 * (dim[1] * dim[1] + dim[2] * dim[2])
            lid[1] += dm / 12.0 * (dim[0] * dim[0] + dim[2] * dim[2])
            lid[2] += dm / 12.0 * (dim[0] * dim[0] + dim[1] * dim[1])
            self._p.changeDynamics(self.human_id, link_id, mass=mass)
            self._p.changeDynamics(self.human_id, link_id, localInertiaDiagonal=tuple(lid))

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        return np.copy(np.array(arr) + self.np_random.uniform(low=-r, high=r, size=len(arr)))

    def print_all_joints_info(self):
        for i in range(self._p.getNumJoints(self.human_id)):
            print(self._p.getJointInfo(self.human_id, i)[0:3],
                  self._p.getJointInfo(self.human_id, i)[8], self._p.getJointInfo(self.human_id, i)[9],
                  self._p.getJointInfo(self.human_id, i)[12])

    def apply_action(self, a):
        # TODO: student code starts here
        
        self.tar_q += a * self.delta_tar_q_scale
        self._p.setJointMotorControlArray(
                bodyIndex=self.human_id,
                jointIndices=range(self.n_dofs),
                controlMode=self._p.POSITION_CONTROL,
                targetPositions=self.tar_q,
                forces=self.max_forces)
        """
        self.tar_q += a * self.delta_tar_q_scale
        self._p.setJointMotorControlArray(self.human_id, range(self.n_dofs), 
            self._p.POSITION_CONTROL, forces = self.max_forces)
        """
        # student code ends

    def get_robot_observation(self):
        # TODO: student code starts here
        
        
        observation = []
        for j in range(self._p.getNumJoints(self.human_id)):
            jnt_angle, jnt_vel, *_ = self._p.getJointState(self.human_id, j)
            observation.append(jnt_angle)
            observation.append(jnt_vel)
        
        base_com, base_quat = self._p.getBasePositionAndOrientation(self.human_id)
        base_vel, _ = self._p.getBaseVelocity(self.human_id)
        
        observation.extend(base_com)
        observation.extend(base_quat)
        observation.extend(base_vel)
        
        rf_com, _ = rb.get_link_com_xyz_orn(self.human_id, 8) # right foot
        lf_com, _ = rb.get_link_com_xyz_orn(self.human_id, 14) # left foot
        
        observation.extend(rf_com)
        observation.extend(lf_com)
        observation.extend(list(self.tar_q))
        
        return observation
        """
        observation = []
        joints = np.array(self._p.getJointStates(self.human_id, range(self.n_dofs)), dtype=object)
        joint_positions = joints[:, 0]
        joint_velocities = joints[:, 1]
        high, low = rb.get_highest_and_lowest_points_world_xyz(self.human_id, -1)

        observation.extend(list(joint_positions))
        observation.extend(list(joint_velocities))
        
        observation.extend(list(low))
        observation.extend(list(self.tar_q))
        return observation
        """
        # student code ends
