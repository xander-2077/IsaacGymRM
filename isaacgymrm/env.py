from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
from gymnasium.spaces import Box
import torch

from utils.utils import *


class Soccer:
    def __init__(self, args):
        self.args = args
        self.num_agent = self.args.num_agent  # 单边的agent数量

        self.gym = gymapi.acquire_gym()

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

        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )

        self.create_envs()
        if not self.args.headless:
            self.create_viewer()

        self.gym.prepare_sim(self.sim)

        # Actor root state tensor
        # Shape: (num_env * num_actor, 13)
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)

        # DoF state tensor
        # Shape: (num_dofs, 2)
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)

        self.actor_index_in_sim_flat = torch.tensor(list(self.actor_index_in_sim.values()), dtype=torch.int32, device=self.args.sim_device).flatten()

        # ---------------------------------------------------------------------------
        # Some args after creating envs

        # Robot: Pos(2), Vel(2), Ori(1), AngularVel(1), Gripper(1) 
        # Ball: Pos(2), Vel(2)
        # Goal: GoalPos(2), OpponentGoalPos(2)
        self.num_info_per_robot = 7   # Ignore gripper for now
        self.num_obs = self.num_agent * 2 * self.num_info_per_robot + 8
        self.num_obs_per_robot = (self.num_agent * 2 - 1) * self.num_info_per_robot + 8
        self.num_act_per_robot = 4
        self.num_act = self.num_agent * self.num_act_per_robot  # 控制四个轮子转速

        # Observation space
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )

        # Action space
        self.action_space = Box(
            -self.rm_wheel_vel_limit,
            self.rm_wheel_vel_limit,
            (self.num_act,),
            dtype=np.float32,
        )

        # Args about episode info
        self.max_episode_length = self.args.episode_length
        self.episode_step = 0

        # Some buffers
        self.state_buf = torch.zeros(
            (self.args.num_env * self.num_agent * 2, self.num_obs),
            device=self.args.sim_device,
        )
        # TODO: Modify
        self.obs_buf = torch.zeros(
            (self.args.num_env * self.num_agent * 2, self.num_obs),
            device=self.args.sim_device,
        )
        self.reward_buf = torch.zeros(
            (self.args.num_env * self.num_agent * 2), device=self.args.sim_device
        )
        self.reset_buf = torch.zeros(
            self.args.num_env, device=self.args.sim_device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self.args.num_env, device=self.args.sim_device, dtype=torch.long
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
        num_per_row = int(np.sqrt(self.args.num_env))

        asset_root = self.args.asset_root

        # Create field
        field_asset_file = 'field.urdf'
        field_options = gymapi.AssetOptions()
        field_options.fix_base_link = True
        field_options.collapse_fixed_joints = True
        field_asset = self.gym.load_asset(
            self.sim, asset_root, field_asset_file, field_options
        )

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

        for i in range(self.args.num_env):
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
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(
            self.viewer, self.envs[self.args.num_env // 2], cam_pos, cam_target
        )
    

    def get_obs_global(self):
        '''
        Global Observation
        Robot: ID(1), Pos(2), Vel(2), Ori(1), AngularVel(1) * 4
        Ball: BallPos(2), BallVel(2)
        Goal: GoalRedPos(2), GoalBluePos(2)
        '''
        num_global_obs = self.num_obs  # 36
        num_info_per_robot = self.num_info_per_robot  # 7

        # 每个env的观测值顺序为: Robot1, Robot2, Robot3, Robot4, Ball, GoalRed, GoalBlue
        obs = torch.zeros((self.args.num_env, num_global_obs), device=self.args.sim_device)

        # self.root_tensor: (num_env * num_actor, 13)
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_linvels = self.root_tensor[:, 7:10]
        self.root_orientations = self.root_tensor[:, 3:7]   # xyzw
        self.root_angvels = self.root_tensor[:, 10:13]

        for i, env_ptr in enumerate(self.envs):
            for j, actor_index in enumerate(self.actor_index_in_sim[env_ptr]):
                if j < len(self.actor_index_in_sim[env_ptr]) - 1:
                    # Robot: ID(1), Pos(2), Vel(2), Ori(1), AngularVel(1)
                    obs[i, j * num_info_per_robot] = j
                    obs[i, j * num_info_per_robot + 1 : j * num_info_per_robot + 3] = self.root_positions[actor_index][:-1]
                    obs[i, j * num_info_per_robot + 3 : j * num_info_per_robot + 5] = self.root_linvels[actor_index][:-1]
                    obs[i, j * num_info_per_robot + 5] = quaternion_to_yaw(self.root_orientations[actor_index], self.args.sim_device)
                    obs[i, j * num_info_per_robot + 6] = self.root_angvels[actor_index][-1]
                else:
                    # Ball: BallPos(2), BallVel(2)
                    obs[i, j * num_info_per_robot : j * num_info_per_robot + 2] = self.root_positions[actor_index][:-1]
                    obs[i, j * num_info_per_robot + 2 : j * num_info_per_robot + 4] = self.root_linvels[actor_index][:-1]

            # Goal: GoalRed(2), GoalBlue(2)
            obs[i, j * num_info_per_robot + 4 : j * num_info_per_robot + 6] = torch.tensor([-4.5, 0.0])
            obs[i, j * num_info_per_robot + 6 : j * num_info_per_robot + 8] = torch.tensor([4.5, 0.0])

        obs_dict = {}
        obs_dict['r0'] = obs[..., :num_info_per_robot]
        obs_dict['r1'] = obs[..., num_info_per_robot:2*num_info_per_robot]
        obs_dict['b0'] = obs[..., 2*num_info_per_robot:3*num_info_per_robot]
        obs_dict['b1'] = obs[..., 3*num_info_per_robot:4*num_info_per_robot]
        obs_dict['ball_pos'] = obs[..., 4*num_info_per_robot:4*num_info_per_robot+2]
        obs_dict['ball_vel'] = obs[..., 4*num_info_per_robot+2:4*num_info_per_robot+4]
        obs_dict['goal_r'] = obs[..., 4*num_info_per_robot+4:4*num_info_per_robot+6]
        obs_dict['goal_b'] = obs[..., 4*num_info_per_robot+6:4*num_info_per_robot+8]

        return obs, obs_dict


    def get_obs_local(self, rm_id):
        '''
        暂时不用
        机器人局部观测，以机器人为中心的相对坐标
        Robot: Pos(2), Vel(2), Ori(1), AngularVel(1), ID(1)
        OpponentRobot: Pos(2), Vel(2), Ori(1), AngularVel(1), ID(1) * 3
        Ball: BallPos(2), BallVel(2)
        Goal: GoalPos(2), OpponentGoalPos(2)
        '''
        obs_global = self.get_obs_global()
        num_envs = obs_global.shape[0]
        num_info_per_robot = self.num_info_per_robot
        local_obs_dim = num_info_per_robot * 5 + 4 + 4  # 1 robot + 3 opponents + ball + 2 goals
        local_obs = torch.zeros((num_envs, local_obs_dim), device=self.args.sim_device)

        for i in range(num_envs):
            # 获取当前机器人的全局状态
            base_idx = rm_id * num_info_per_robot
            robot_pos = obs_global[i, base_idx + 1:base_idx + 3]
            robot_ori = obs_global[i, base_idx + 5]
            robot_cos = torch.cos(-robot_ori)
            robot_sin = torch.sin(-robot_ori)
            rotation_matrix = torch.tensor([[robot_cos, -robot_sin], [robot_sin, robot_cos]])

            # 自身状态（不变）
            local_obs[i, :num_info_per_robot] = obs_global[i, base_idx:base_idx + num_info_per_robot]

            # 其他机器人的状态
            local_idx = num_info_per_robot
            for j in range(4):
                if j != rm_id:
                    opp_base_idx = j * num_info_per_robot
                    opp_pos = obs_global[i, opp_base_idx + 1:opp_base_idx + 3]
                    opp_vel = obs_global[i, opp_base_idx + 3:opp_base_idx + 5]
                    # 转换为相对位置和速度
                    relative_pos = opp_pos - robot_pos
                    relative_pos = torch.matmul(rotation_matrix, relative_pos.unsqueeze(-1)).squeeze(-1)
                    relative_vel = torch.matmul(rotation_matrix, opp_vel.unsqueeze(-1)).squeeze(-1)
                    # 更新局部观测
                    local_obs[i, local_idx:local_idx + 2] = relative_pos
                    local_obs[i, local_idx + 2:local_idx + 4] = relative_vel
                    local_obs[i, local_idx + 4:local_idx + 6] = obs_global[i, opp_base_idx + 5:opp_base_idx + 7]
                    local_obs[i, local_idx + 6] = obs_global[i, opp_base_idx]
                    local_idx += num_info_per_robot

            # 球的状态
            ball_idx = 4 * num_info_per_robot
            ball_pos = obs_global[i, ball_idx + 7:ball_idx + 9]
            ball_vel = obs_global[i, ball_idx + 9:ball_idx + 11]
            relative_ball_pos = ball_pos - robot_pos
            relative_ball_pos = torch.matmul(rotation_matrix, relative_ball_pos.unsqueeze(-1)).squeeze(-1)
            relative_ball_vel = torch.matmul(rotation_matrix, ball_vel.unsqueeze(-1)).squeeze(-1)
            local_obs[i, local_idx:local_idx + 2] = relative_ball_pos
            local_obs[i, local_idx + 2:local_idx + 4] = relative_ball_vel

            # 目标和对方目标的位置（相对位置）
            goal_pos = obs_global[i, ball_idx + 11:ball_idx + 13] - robot_pos
            opponent_goal_pos = obs_global[i, ball_idx + 13:ball_idx + 15] - robot_pos
            goal_pos = torch.matmul(rotation_matrix, goal_pos.unsqueeze(-1)).squeeze(-1)
            opponent_goal_pos = torch.matmul(rotation_matrix, opponent_goal_pos.unsqueeze(-1)).squeeze(-1)
            local_obs[i, local_idx + 4:local_idx + 6] = goal_pos
            local_obs[i, local_idx + 6:local_idx + 8] = opponent_goal_pos

        return local_obs


    def get_obs_local(self, obs_global, rm_id):
        '''
        机器人局部观测，以机器人为中心的相对坐标
        Teammeta: ID(1), Pos(2), Vel(2), Ori(1), AngularVel(1)
        OpponentRobot: ID(1), Pos(2), Vel(2), Ori(1), AngularVel(1) * 2
        Ball: BallPos(2), BallVel(2)
        Goal: GoalPos(2), OpponentGoalPos(2)
        '''
        self.num_obs_per_robot



    def get_reward(self, obs_global_dict, obs_r0, obs_r1, obs_b0, obs_b1):
        '''
        TODO: annotation
        '''
        reward_r = 0
        reward_b = 0
        terminated = False

        # Scoring or Conceding
        if obs_global_dict['ball_pos'][0] < obs_global_dict['goal_r'][0]:
            reward_r = -1 * self.args.reward_conceding
            reward_b = 1 * self.args.reward_scoring
            return reward_r, reward_b, True
        elif obs_global_dict['ball_pos'][0] > obs_global_dict['goal_b'][0]:
            reward_r = 1 * self.args.reward_scoring
            reward_b = -1 * self.args.reward_conceding
            return reward_r, reward_b, True

        # Velocity to ball
        for robot in ['r0', 'r1', 'b0', 'b1']:
            dir_vec = obs_global_dict['ball_pos'] - obs_global_dict[robot][1:3]
            norm_dir_vec = dir_vec / dir_vec.norm()
            vel_towards_ball = torch.dot(obs_global_dict[robot][3:5], norm_dir_vec).item()
            if robot in ['r0', 'r1']:
                reward_r += vel_towards_ball * self.args.reward_vel_to_ball
            else:
                reward_b += vel_towards_ball * self.args.reward_vel_to_ball

        # Velocity forward
        reward_r += (obs_global_dict['r0'][3] + obs_global_dict['r1'][3]) * self.args.reward_vel
        reward_b += - (obs_global_dict['b0'][3] + obs_global_dict['b1'][3]) * self.args.reward_vel

        # TODO:



        


        return reward_r, reward_b, terminated





    def apply_actions(self, actions):
        '''
        控制的变量actions_target_tensor维度是2, 很奇怪为什么gym.set_dof_velocity_target_tensor支持这样的参数输入
        '''
        actions_target_tensor = torch.zeros((self.args.num_env, self.num_dof_per_env), device=self.args.sim_device)

        actions_target_tensor[:, self.wheel_dof_handles_per_env] = actions
        
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_target_tensor))
        


    def step(self, actions):
        '''
        actions(Tuple): (action_r, action_b)
        action_r/action_b(Tensor): (num_env, num_agent * 4) 
        tensor([  0,  16,  33,  49,  66,  82,  99, 115, 132, 148, 165, 181, 198, 214, 231, 247])
        '''
        actions = mecanum_tranform(actions, self.args.num_env, self.args.sim_device)
        self.apply_actions(actions)

        # Simulate one step
        self.simulate()
        self.gym.refresh_actor_root_state_tensor(self.sim)  # self.root_tensor
        self.gym.refresh_dof_state_tensor(self.sim)   # self.dof_states

        if self.episode_step == 2:
            self.saved_root_tensor = self.root_tensor.clone()
            self.saved_dof_states = self.dof_states.clone()

        if not self.args.headless:
            self.render()

        obs_global, obs_global_dict = self.get_obs_global()

        # # Compute reward and check if episode is done
        # reward, terminated = self.get_reward(obs_global)

        self.episode_step += 1
        # if self.episode_step >= self.max_episode_length:
        #     return obs_global, reward, False, True, {}
        
        # return obs_global, reward, terminated, False, {}


    def reset(self):
        '''
        TODO: Add annotation

        '''
        # FIXME: It seems that the following code will cause the error: "RuntimeError: CUDA error: an illegal memory access was encountered"
        # self.gym.set_dof_state_tensor_indexed(
        #     self.sim, gymtorch.unwrap_tensor(self.saved_dof_states), gymtorch.unwrap_tensor(self.actor_index_in_sim_flat), len(self.actor_index_in_sim_flat)
        # )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.saved_root_tensor), gymtorch.unwrap_tensor(self.actor_index_in_sim_flat), len(self.actor_index_in_sim_flat)
        )

        self.simulate()
        self.gym.refresh_actor_root_state_tensor(self.sim)  # self.root_tensor
        self.gym.refresh_dof_state_tensor(self.sim)   # self.dof_states

        obs_global = self.get_obs_global()

        if not self.args.headless:
            self.render()

        return obs_global


    def simulate(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)


    def render(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)


    def exit(self):
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


@torch.jit.script
def compute_reward():
    pass
