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

    def reward_function(action, observation):
        #l2normofdq is too high, then give a penalty
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
        return reward
    
    def step(self, a):
        # TODO: student code starts here
        # do two things here: apply action "a" and simulate with fluid force for control_skip times
        # then calculate reward based on post-sim state for return
        fluid_forces = []
        for i in range(self.control_skip):
            self.apply_action(a)
            if i % 5 == 0:   # recompute after 5 simulation steps
                fluid_forces = rb.apply_fluid_force(self.human_id)
            else:
                fluid_forces = rb.apply_fluid_force(self.human_id, fluid_forces)
            self._p.stepSimulation()
        
        observation = self.get_extended_observation()
        reward = reward_function(a, self.get_extended_observation())
        
        # end student code
        return self.get_extended_observation(), reward, False, {}

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
        
        observation.append(base_com)
        observation.append(base_quat)
        observation.append(base_vel)
        
        rf_com, _ = rb.get_link_com_xyz_orn(self.human_id, 8) # right foot
        lf_com, _ = rb.get_link_com_xyz_orn(self.human_id, 14) # left foot
        
        observation.append(rf_com)
        observation.append(lf_com)
        observation.append(self.tar_q)
        
        return observation
