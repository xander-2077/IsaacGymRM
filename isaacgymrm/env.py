from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
from gymnasium.spaces import Box
import torch
import sys, time

from isaacgymrm.utils.utils import *

# TODO: 
# 1. 给出部分需要reset的env
# 2. 为了实现课程学习，需要给出总的训练步数

class Soccer:
    def __init__(self, args):
        self.args = args
        self.num_env = self.args.num_env
        self.num_agent = self.args.num_agent  # 单边的agent数量

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.create_envs()
        self.gym.prepare_sim(self.sim)
        self.create_viewer()

        # Actor root state tensor
        # Shape: (num_env * num_actor, 13)
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

        # DoF state tensor
        # Shape: (num_dofs, 2)
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)

        self.actor_index_in_sim_flat = torch.tensor(list(self.actor_index_in_sim.values()), dtype=torch.int32, device=self.args.sim_device).flatten()

        # -----------------------------------------------------------------------
        # Robot: Pos(2), Vel(2), Ori(1), AngularVel(1), Gripper(1) 
        # Ball: Pos(2), Vel(2)
        # Goal: GoalPos(2), OpponentGoalPos(2)
        self.num_info_per_robot: int = 7   # Ignore gripper for now
        self.num_state = self.num_agent * 2 * self.num_info_per_robot + 8
        self.num_obs = (self.num_agent * 2 - 1) * self.num_info_per_robot + 8
        self.num_act_per_robot = 4
        self.num_act = self.num_agent * self.num_act_per_robot  # 控制四个轮子转速

        self.state_space = Box(low=-np.inf, high=np.inf, shape=(self.num_state,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.num_act,), dtype=np.float32,)

        # self.share_observation_space = [Box(low=-100, high=100, shape = ([self.num_obs]), dtype=np.float16) for _ in range(self.args.num_envs*self.n_agents)]

        self.clip_action = 1.0
        self.clip_obs = np.inf

        # Args about episode info
        self.total_train_env_step = 0
        self.max_episode_length = self.args.max_episode_length
        self.episode_step = 0
        self.tensor_clone_flag = False

        self.allocate_buffers()
        self.obs_dict = {}   # TODO: obs_dict['state'], obs_dict['obs']
        # -----------------------------------------------------------------------
        # self.reset()


    def create_sim(self):
        # Configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 60.0  # default: 1/60.0
        sim_params.substeps = 2  # default: 2
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.physx.num_position_iterations = 8  # default: 4
        sim_params.physx.num_velocity_iterations = 1  # default: 1
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02

        self.dt = sim_params.dt
        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )


    def create_envs(self):
        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        self.gym.add_ground(self.sim, plane_params)

        # Define environment space (for visualisation)
        lower = gymapi.Vec3(0, 0, 0)
        upper = gymapi.Vec3(12, 9, 0)
        num_per_row = int(np.sqrt(self.num_env))

        asset_root = self.args.asset_root

        # Create field
        field_asset_file = 'field.urdf'
        field_options = gymapi.AssetOptions()
        field_options.fix_base_link = True
        field_options.collapse_fixed_joints = True
        field_asset = self.gym.load_asset(
            self.sim, asset_root, field_asset_file, field_options)

        # Create robomaster asset
        rm_asset_file = 'RM_description/robot/robomaster.urdf'
        rm_options = gymapi.AssetOptions()
        rm_options.fix_base_link = False
        rm_options.collapse_fixed_joints = False
        rm_asset = self.gym.load_asset(self.sim, asset_root, rm_asset_file, rm_options)
        self.num_rm_dof = self.gym.get_asset_dof_count(rm_asset)

        rm_dof_props = self.gym.get_asset_dof_properties(rm_asset)
        rm_gripper_limits = []

        for i in range(rm_dof_props.shape[0]):
            if rm_dof_props[i]['hasLimits']:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_EFFORT
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 0.0
                rm_gripper_limits.append([rm_dof_props[i]['lower'], rm_dof_props[i]['upper'], rm_dof_props[i]['effort']])

            elif rm_dof_props[i]['velocity'] < 1e2:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_VEL
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 500.0  # need to tuned
                self.rm_wheel_vel_limit = rm_dof_props[i]['velocity']  # max velocity
            else:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_NONE
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 0.0

        self.rm_gripper_limits = torch.tensor(rm_gripper_limits, device=self.args.sim_device)   # lower, upper, max effort

        rm_pose = [gymapi.Transform() for i in range(4)]
        rm_pose[0].p = gymapi.Vec3(-2, -2, 0.01)
        rm_pose[1].p = gymapi.Vec3(-2, 2, 0.01)
        rm_pose[2].p = gymapi.Vec3(2, -2, 0.01)
        rm_pose[3].p = gymapi.Vec3(2, 2, 0.01)
        rm_pose[2].r = gymapi.Quat(0, 0, 1, 0)
        rm_pose[3].r = gymapi.Quat(0, 0, 1, 0)

        # Create ball asset
        ball_asset_file = "ball.urdf"
        ball_options = gymapi.AssetOptions()   # need to tuned
        ball_options.angular_damping = 0.77
        ball_options.linear_damping = 0.77
        ball_asset = self.gym.load_asset(
            self.sim, asset_root, ball_asset_file, ball_options
        )
        ball_init_pose = gymapi.Transform()
        ball_init_pose.p = gymapi.Vec3(0, 0, 0.1)

        # Generate environments
        self.envs = []
        self.rm_handles = {}    # {env_ptr: [rm_handle1, rm_handle2, rm_handle3, rm_handle4]}
        self.ball_handles = {}  # {env_ptr: ball_handle}
        self.actor_index_in_sim = {}    # {env_ptr: [rm1_index, rm2_index, rm3_index, rm4_index, ball_index]}
        self.wheel_dof_handles = {}    # {env_ptr: [[front_left_wheel_dof, front_right_wheel_dof, rear_left_wheel_dof, rear_right_wheel_dof], ...]}

        for i in range(self.num_env):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.create_actor(
                env_ptr, field_asset, gymapi.Transform(), "field", i, 0, 0
            )

            # Create robomaster actors
            self.rm_handles[env_ptr] = []
            self.actor_index_in_sim[env_ptr] = []
            self.wheel_dof_handles[env_ptr] = []

            for j in range(self.num_agent * 2):
                rm_handle = self.gym.create_actor(
                    env_ptr, rm_asset, rm_pose[j], "rm" + "_" + str(j), i, 2**(j+1), 0
                )
                self.gym.set_actor_dof_properties(env_ptr, rm_handle, rm_dof_props)
                self.rm_handles[env_ptr].append(rm_handle)

                self.actor_index_in_sim[env_ptr].append(self.gym.get_actor_index(env_ptr, rm_handle, gymapi.DOMAIN_SIM))

                front_left_wheel_dof = self.gym.find_actor_dof_handle(
                    env_ptr, rm_handle, "front_left_wheel_joint"
                )
                front_right_wheel_dof = self.gym.find_actor_dof_handle(
                    env_ptr, rm_handle, "front_right_wheel_joint"
                )
                rear_left_wheel_dof = self.gym.find_actor_dof_handle(
                    env_ptr, rm_handle, "rear_left_wheel_joint"
                )
                rear_right_wheel_dof = self.gym.find_actor_dof_handle(
                    env_ptr, rm_handle, "rear_right_wheel_joint"
                )

                self.wheel_dof_handles[env_ptr].append(
                    [
                        front_left_wheel_dof,
                        front_right_wheel_dof,
                        rear_left_wheel_dof,
                        rear_right_wheel_dof,
                    ]
                )

            ball_handle = self.gym.create_actor(
                env_ptr, ball_asset, ball_init_pose, "ball", i, 1, 0
            )
            self.ball_handles[env_ptr] = ball_handle

            self.actor_index_in_sim[env_ptr].append(self.gym.get_actor_index(env_ptr, ball_handle, gymapi.DOMAIN_SIM))

            self.envs.append(env_ptr)

        self.wheel_dof_handles_per_env = torch.tensor(
            self.wheel_dof_handles[self.envs[0]]
        ).reshape(self.num_agent * 2 * 4, )   # tensor([  0,  16,  33,  49,  66,  82,  99, 115, 132, 148, 165, 181, 198, 214, 231, 247])

        self.num_dof_per_env = self.gym.get_env_dof_count(self.envs[0])
            

    def create_viewer(self):
        self.enable_viewer_sync = True
        self.viewer = None

        if not self.args.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            cam_pos = gymapi.Vec3(10, 0.0, 5)
            cam_target = gymapi.Vec3(-1, 0, 0)
            self.gym.viewer_camera_look_at(
                self.viewer, self.envs[self.num_env // 2], cam_pos, cam_target
            )
        
    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)

                # # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # # this code will slow down the rendering to real time
                # now = time.time()
                # delta = now - self.last_frame_time
                # if self.render_fps < 0:
                #     # render at control frequency
                #     render_dt = self.dt * self.control_freq_inv  # render every control step
                # else:
                #     render_dt = 1.0 / self.render_fps

                # if delta < render_dt:
                #     time.sleep(render_dt - delta)

                # self.last_frame_time = time.time()
            else:
                self.gym.poll_viewer_events(self.viewer)

    def allocate_buffers(self):
        '''Allocate the observation, states, etc. buffers.'''
        self.state_buf = torch.zeros(
            (self.num_env, self.num_state), device=self.args.sim_device,
            dtype=torch.float)
        # TODO: 考虑一下obs维度的设置
        self.obs_buf = torch.zeros(
            (self.num_env * self.num_agent * 2, self.num_obs),
            device=self.args.sim_device, dtype=torch.float)
        # TODO: 考虑一下reward的设置
        self.reward_buf = torch.zeros(
            (self.num_env * self.num_agent * 2), device=self.args.sim_device, dtype=torch.float)
        self.reset_buf = torch.zeros(
            self.num_env, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_env, device=self.args.sim_device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_env, device=self.args.sim_device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_env, device=self.args.sim_device, dtype=torch.long)
        self.extras = {}


    def get_state(self):
        '''
        Global Observation
        Robot: ID(1), Pos(2), Vel(2), Ori(1), AngularVel(1) * 4
        Ball: BallPos(2), BallVel(2)
        Goal: GoalRedPos(2), GoalBluePos(2)
        '''
        num_state = self.num_state  # 36
        num_info_per_robot = self.num_info_per_robot  # 7

        # 每个env的观测值顺序为: Robot1, Robot2, Robot3, Robot4, Ball, GoalRed, GoalBlue
        state = torch.zeros((self.num_env, num_state), device=self.args.sim_device)

        # self.root_tensor: (num_env * num_actor, 13)
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_linvels = self.root_tensor[:, 7:10]
        self.root_orientations = self.root_tensor[:, 3:7]   # xyzw
        self.root_angvels = self.root_tensor[:, 10:13]

        for i, env_ptr in enumerate(self.envs):
            for j, actor_index in enumerate(self.actor_index_in_sim[env_ptr]):
                if j < len(self.actor_index_in_sim[env_ptr]) - 1:
                    # Robot: ID(1), Pos(2), Vel(2), Ori(1), AngularVel(1)
                    state[i, j * num_info_per_robot] = j
                    state[i, j * num_info_per_robot + 1 : j * num_info_per_robot + 3] = self.root_positions[actor_index][:-1]
                    state[i, j * num_info_per_robot + 3 : j * num_info_per_robot + 5] = self.root_linvels[actor_index][:-1]
                    state[i, j * num_info_per_robot + 5] = quaternion_to_yaw(self.root_orientations[actor_index], self.args.sim_device)
                    state[i, j * num_info_per_robot + 6] = self.root_angvels[actor_index][-1]
                else:
                    # Ball: BallPos(2), BallVel(2)
                    state[i, j * num_info_per_robot : j * num_info_per_robot + 2] = self.root_positions[actor_index][:-1]
                    state[i, j * num_info_per_robot + 2 : j * num_info_per_robot + 4] = self.root_linvels[actor_index][:-1]

            # Goal: GoalRed(2), GoalBlue(2)
            state[i, j * num_info_per_robot + 4 : j * num_info_per_robot + 6] = torch.tensor([-4.5, 0.0])
            state[i, j * num_info_per_robot + 6 : j * num_info_per_robot + 8] = torch.tensor([4.5, 0.0])

        state_dict = {}
        state_dict['r0'] = state[..., :num_info_per_robot]
        state_dict['r1'] = state[..., num_info_per_robot:2*num_info_per_robot]
        state_dict['b0'] = state[..., 2*num_info_per_robot:3*num_info_per_robot]
        state_dict['b1'] = state[..., 3*num_info_per_robot:4*num_info_per_robot]
        state_dict['ball_pos'] = state[..., 4*num_info_per_robot:4*num_info_per_robot+2]
        state_dict['ball_vel'] = state[..., 4*num_info_per_robot+2:4*num_info_per_robot+4]
        state_dict['goal_r'] = state[..., 4*num_info_per_robot+4:4*num_info_per_robot+6]
        state_dict['goal_b'] = state[..., 4*num_info_per_robot+6:4*num_info_per_robot+8]

        return state, state_dict


    def get_obs(self, state, rm_id):
        '''
        暂时不用
        机器人局部观测，以机器人为中心的相对坐标
        Robot: Pos(2), Vel(2), Ori(1), AngularVel(1), ID(1)
        OpponentRobot: Pos(2), Vel(2), Ori(1), AngularVel(1), ID(1) * 3
        Ball: BallPos(2), BallVel(2)
        Goal: GoalPos(2), OpponentGoalPos(2)
        '''
        state = self.get_state()
        num_envs = state.shape[0]
        num_info_per_robot = self.num_info_per_robot
        local_obs_dim = num_info_per_robot * 5 + 4 + 4  # 1 robot + 3 opponents + ball + 2 goals
        local_obs = torch.zeros((num_envs, local_obs_dim), device=self.args.sim_device)

        for i in range(num_envs):
            # 获取当前机器人的全局状态
            base_idx = rm_id * num_info_per_robot
            robot_pos = state[i, base_idx + 1:base_idx + 3]
            robot_ori = state[i, base_idx + 5]
            robot_cos = torch.cos(-robot_ori)
            robot_sin = torch.sin(-robot_ori)
            rotation_matrix = torch.tensor([[robot_cos, -robot_sin], [robot_sin, robot_cos]])

            # 自身状态（不变）
            local_obs[i, :num_info_per_robot] = state[i, base_idx:base_idx + num_info_per_robot]

            # 其他机器人的状态
            local_idx = num_info_per_robot
            for j in range(4):
                if j != rm_id:
                    opp_base_idx = j * num_info_per_robot
                    opp_pos = state[i, opp_base_idx + 1:opp_base_idx + 3]
                    opp_vel = state[i, opp_base_idx + 3:opp_base_idx + 5]
                    # 转换为相对位置和速度
                    relative_pos = opp_pos - robot_pos
                    relative_pos = torch.matmul(rotation_matrix, relative_pos.unsqueeze(-1)).squeeze(-1)
                    relative_vel = torch.matmul(rotation_matrix, opp_vel.unsqueeze(-1)).squeeze(-1)
                    # 更新局部观测
                    local_obs[i, local_idx:local_idx + 2] = relative_pos
                    local_obs[i, local_idx + 2:local_idx + 4] = relative_vel
                    local_obs[i, local_idx + 4:local_idx + 6] = state[i, opp_base_idx + 5:opp_base_idx + 7]
                    local_obs[i, local_idx + 6] = state[i, opp_base_idx]
                    local_idx += num_info_per_robot

            # 球的状态
            ball_idx = 4 * num_info_per_robot
            ball_pos = state[i, ball_idx + 7:ball_idx + 9]
            ball_vel = state[i, ball_idx + 9:ball_idx + 11]
            relative_ball_pos = ball_pos - robot_pos
            relative_ball_pos = torch.matmul(rotation_matrix, relative_ball_pos.unsqueeze(-1)).squeeze(-1)
            relative_ball_vel = torch.matmul(rotation_matrix, ball_vel.unsqueeze(-1)).squeeze(-1)
            local_obs[i, local_idx:local_idx + 2] = relative_ball_pos
            local_obs[i, local_idx + 2:local_idx + 4] = relative_ball_vel

            # 目标和对方目标的位置（相对位置）
            goal_pos = state[i, ball_idx + 11:ball_idx + 13] - robot_pos
            opponent_goal_pos = state[i, ball_idx + 13:ball_idx + 15] - robot_pos
            goal_pos = torch.matmul(rotation_matrix, goal_pos.unsqueeze(-1)).squeeze(-1)
            opponent_goal_pos = torch.matmul(rotation_matrix, opponent_goal_pos.unsqueeze(-1)).squeeze(-1)
            local_obs[i, local_idx + 4:local_idx + 6] = goal_pos
            local_obs[i, local_idx + 6:local_idx + 8] = opponent_goal_pos

        return local_obs


    def get_reward(self, state_dict, obs_r0, obs_r1, obs_b0, obs_b1):
        reward_r = 0
        reward_b = 0
        terminated = False

        # Scoring or Conceding
        if state_dict['ball_pos'][0] < state_dict['goal_r'][0]:
            reward_r = -1 * self.args.reward_conceding
            reward_b = 1 * self.args.reward_scoring
            return reward_r, reward_b, True
        elif state_dict['ball_pos'][0] > state_dict['goal_b'][0]:
            reward_r = 1 * self.args.reward_scoring
            reward_b = -1 * self.args.reward_conceding
            return reward_r, reward_b, True

        # Velocity to ball
        for robot in ['r0', 'r1', 'b0', 'b1']:
            dir_vec = state_dict['ball_pos'] - state_dict[robot][1:3]
            norm_dir_vec = dir_vec / dir_vec.norm()
            vel_towards_ball = torch.dot(state_dict[robot][3:5], norm_dir_vec).item()
            if robot in ['r0', 'r1']:
                reward_r += vel_towards_ball * self.args.reward_vel_to_ball
            else:
                reward_b += vel_towards_ball * self.args.reward_vel_to_ball

        # Velocity forward
        reward_r += (state_dict['r0'][3] + state_dict['r1'][3]) * self.args.reward_vel
        reward_b += - (state_dict['b0'][3] + state_dict['b1'][3]) * self.args.reward_vel

        # TODO: add more rewards


        return reward_r, reward_b, terminated


    def pre_physics_step(self, actions):
        '''
        Apply the actions to the environment (eg by setting torques, position targets).

        控制的变量actions_target_tensor维度是2, 很奇怪为什么gym.set_dof_velocity_target_tensor支持这样的参数输入
        '''
        actions = mecanum_tranform(actions, self.num_env, self.args.sim_device)

        actions_target_tensor = torch.zeros((self.num_env, self.num_dof_per_env), device=self.args.sim_device)

        actions_target_tensor[:, self.wheel_dof_handles_per_env] = actions
        
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_target_tensor))
        

    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        # TODO:
        self.progress_buf += 1
        env_idx = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_idx) > 0:
            self.reset_idx(env_idx)


        self.gym.refresh_actor_root_state_tensor(self.sim)  # self.root_tensor
        self.gym.refresh_dof_state_tensor(self.sim)   # self.dof_states


        state, state_dict = self.get_state()

        # self.get_reward(state_dict, **kwargs)

        return None


    def step(self, actions: torch.Tensor):
        '''
        actions(Tuple): (action_r, action_b)
        action_r/action_b(Tensor): (num_env, num_agent * 4) 
        tensor([  0,  16,  33,  49,  66,  82,  99, 115, 132, 148, 165, 181, 198, 214, 231, 247])
        # TODO: apply randomizations to actions and observations
        '''
        action_tensor = torch.clamp(actions, -self.clip_action, self.clip_action)
        
        self.pre_physics_step(action_tensor)

        for _ in range(self.args.control_freq_inv):  # TODO: control_freq_inv
            self.render()
            self.gym.simulate(self.sim)
        
        self.post_physics_step()

        # if self.episode_step == 2:
        if not torch.all(self.root_tensor.eq(0.)) and not torch.all(self.dof_states.eq(0.)) and not self.tensor_clone_flag:
            self.saved_root_tensor = self.root_tensor.clone()
            self.saved_dof_states = self.dof_states.clone()
            self.tensor_clone_flag = True

        self.episode_step += 1

        

        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)   # TODO: truncated


        # self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)
        
        # if self.episode_step >= self.max_episode_length:
        #     return state, reward, False, True, {}
        self.total_train_env_step += 1  # TODO: 这里可能有问题
        
        # return state, reward, terminated, False, {}
        return self.obs_dict, self.reward_buf, self.reset_buf, self.extras


    def reset_idx(self, env_idx):
        """Reset environment with indices in env_idx. """

        actor_indices = self.actor_index_in_sim[self.envs[env_idx]].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.saved_root_tensor), actor_indices, len(actor_indices)
        )

        # Clear up desired buffer states
        self.reset_buf[env_idx] = 0
        self.progress_buf[env_idx] = 0
        
        state, _ = self.get_state()

        return state, None


    def reset(self):
        '''
        TODO: Add annotation
        Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.

        '''
        # FIXME: It seems that the following code will cause the error: "RuntimeError: CUDA error: an illegal memory access was encountered"
        # self.gym.set_dof_state_tensor_indexed(
        #     self.sim, gymtorch.unwrap_tensor(self.saved_dof_states), gymtorch.unwrap_tensor(self.actor_index_in_sim_flat), len(self.actor_index_in_sim_flat)
        # )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.saved_root_tensor), gymtorch.unwrap_tensor(self.actor_index_in_sim_flat), len(self.actor_index_in_sim_flat)
        )

        for _ in range(self.args.control_freq_inv):  # TODO: control_freq_inv
            self.render()
            self.gym.simulate(self.sim)
            
        self.gym.refresh_actor_root_state_tensor(self.sim)  # self.root_tensor
        self.gym.refresh_dof_state_tensor(self.sim)   # self.dof_states

        state, _ = self.get_state()

        self.render()

        self.episode_step = 0

        return state, None


    def apply_randomization(self):
        pass


    def exit(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_reward():
    pass
