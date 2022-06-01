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

# python -m a2c_ppo_acktr.train_policy --env-name "HumanoidSwimmerEnv-v1" --num-steps 1920  --num-processes 12
# --lr 3e-4 --entropy-coef 0 --ppo-epoch 10 --num-mini-batch 32 --num-env-steps 10000000 --use-linear-lr-decay
# --clip-param 0.2 --save-dir trained_models_swim_1 --seed 20072 --using-torque-ctrl 1

from .humanoid_swimmer import HumanoidSwimmer

from pybullet_utils import bullet_client
import pybullet
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import math

from . import rigidBodySento as rb

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class HumanoidSwimmerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 control_skip=10,
                 using_torque_ctrl=True
                 ):

        self.render = render
        self.init_noise = init_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 480.

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = HumanoidSwimmer(init_noise=self.init_noise, time_step=self._ts, np_random=self.np_random,
                                     using_torque_ctrl=using_torque_ctrl)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None
        obs = self.reset()  # and update init obs

        action_dim = self.robot.n_dofs
        self.act = [0.0] * self.robot.n_dofs
        self.action_space = gym.spaces.Box(low=np.array([-1.] * action_dim), high=np.array([+1.] * action_dim))
        obs_dim = len(obs)
        obs_dummy = np.array([1.12234567] * obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._ts)
        self._p.setGravity(0, 0, -rb.GRAVITY)

        self.floor_id = self._p.loadURDF(
            os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1
        )
        self._p.changeDynamics(self.floor_id, -1,
                               lateralFriction=1.0, restitution=0.5)
        _ = rb.create_primitive_shape(-1, self._p.GEOM_BOX, [4, 4, rb.WATER_SURFACE / 2.0], [0, 0, 0.9, 0.3],
                                      False, [0, 0, rb.WATER_SURFACE / 2.0])
        self.robot.reset(self._p)

        self.timer = 0

        # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

        self._p.stepSimulation()

        obs = self.get_extended_observation()

        return np.array(obs)
    
    def step(self, a):
        # TODO: student code starts here
        # do two things here: apply action "a" and simulate with fluid force for control_skip times
        # then calculate reward based on post-sim state for return
        
        
        fluid_forces = []
        for i in range(self.control_skip):
            self.robot.apply_action(a)
            if i % 5 == 0:   # recompute after 5 simulation steps
                fluid_forces = rb.apply_fluid_force(self.robot.human_id)
            else:
                fluid_forces = rb.apply_fluid_force(self.robot.human_id, fluid_forces)
            self._p.stepSimulation()
        
        observation = self.get_extended_observation()
        
        reward = 0.
        
        #if head is by the water surface within margin weight with proportion to distance
        margin = 1.
        
        head_id = 23
        head_q = observation[head_id * 2] 
        if head_q <= margin: 
            diff = rb.WATER_SURFACE - head_q
            weight = 5.0
            reward += weight * (1.0 - diff)
        
        velocity_threshold = 5.0
        
        r_arm = 16
        r_arm_dq = observation[r_arm * 2 + 1]  
        l_arm = 20
        l_arm_dq = observation[l_arm * 2 + 1]
        if np.linalg.norm(r_arm_dq) >= velocity_threshold or np.linalg.norm(l_arm_dq) >= velocity_threshold: 
            r_diff = r_arm_dq - velocity_threshold
            l_diff = l_arm_dq - velocity_threshold
            weight = -0.1
            reward += (weight * r_diff) + (weight * l_diff)
        
        #print("reward: ", reward)
        
        # end student code
        return self.get_extended_observation(), reward, False, {}
        
        
        """
        for s in range(self.control_skip):
            self.robot.apply_action(a)
            if (s % 5 == 0):
                rb.apply_fluid_force(self.robot.human_id)
            self._p.stepSimulation()

        observation = self.get_extended_observation()


        #if head is by the water surface within margin weight with proportion to distance
        margin = 1.0
        reward = 0.0
        #print(observation[23])
        if (observation[23] <= margin):
            diff = rb.WATER_SURFACE - observation[23]
            print("diff: " + str(diff))
            weight = 2.0
            print("weight:" + str(weight))
            # print("diff: " + str(diff) + " weight: " + str(weight))

            r = weight * (1.0-diff)
            print("reward: " + str(r))
            reward += r

        # trying to test velocities of arms
        velocity_threshold = 5.0
        #print("velocity: " + str(observation[23+16]))
        if (observation[23+16] >= velocity_threshold or observation[23+20] >= velocity_threshold):
            diff_1 = observation[23+16]  - velocity_threshold
            diff_2 = observation[23+20]  - velocity_threshold

            weight = -0.1
            reward += (weight * diff_1) + (weight * diff_2)

        # trying to make sure velocities are high enough to move
        ll_velocity = 0.01
        if (observation[23+16] <= ll_velocity or observation[23+20] >= ll_velocity):
            diff_1 = velocity_threshold - observation[23+16]
            diff_2 = velocity_threshold - observation[23+20]

            weight = -0.5
            reward += (weight * diff_1) + (weight * diff_2)
        # print("reward: " + str(reward))
        # end student code
        return self.get_extended_observation(), reward, False, {}
        """
    
    
    def get_extended_observation(self):
        # TODO: student code starts here
        return self.robot.get_robot_observation()
        # end student code

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s
