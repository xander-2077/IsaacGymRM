from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
from gymnasium.spaces import Box
import torch


class Soccer:
    def __init__(self, args):
        self.args = args

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
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.002

        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )

        # Some args before creating envs
        self.num_agent = self.args.num_agent  # 单边的agent数量

        self.create_envs()
        # import pdb; pdb.set_trace()
        self.create_viewer()

        # num_dof ?
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.gym.prepare_sim(self.sim)

        # Some args after creating envs

        # self.actors_per_env = self.num_agent * 2
        self.num_obs = (
            self.num_agent * 2 * 7 + 4
        )  # Robot: Pos(2), Vel(2), Ori(1), AngularVel(1), Gripper(1) Ball: BallPos(2), BallVel(2)
        self.num_act = self.num_agent * 4  # 控制四个轮子转速

        # Observation space
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )

        # Action space
        self.action_space = Box(
            -self.rm_wheel_vel_limit,
            self.rm_wheel_vel_limit,
            (self.num_act * self.num_agent,),
            dtype=np.float32,
        )

        self.max_episode_length = self.args.episode_length
        self.episode_length = 0

        self.state_buf = torch.zeros(
            (self.args.num_envs * self.num_agent * 2, self.num_obs),
            device=self.args.sim_device,
        )
        self.obs_buf = torch.zeros(
            (self.args.num_envs * self.num_agent * 2, self.num_obs),
            device=self.args.sim_device,
        )
        self.reward_buf = torch.zeros(
            (self.args.num_envs * self.num_agent * 2), device=self.args.sim_device
        )
        self.reset_buf = torch.zeros(
            self.args.num_envs, device=self.args.sim_device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self.args.num_envs, device=self.args.sim_device, dtype=torch.long
        )


        # For Test Only!
        dof_velocity_tensor = torch.zeros((self.args.num_envs, self.env_dof_count), device=self.args.sim_device)
        dof_velocity_tensor[:, self.wheel_dof_handles_per_env[:4]] = 10.0
        # import pdb; pdb.set_trace()
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_velocity_tensor))


    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        lower = gymapi.Vec3(0, 0, 0)
        upper = gymapi.Vec3(12, 9, 0)
        num_per_row = int(np.sqrt(self.args.num_envs))

        # create field
        asset_root = 'assets'
        field_asset_file = 'field.urdf'
        field_options = gymapi.AssetOptions()
        field_options.fix_base_link = True
        field_options.collapse_fixed_joints = True
        field_asset = self.gym.load_asset(
            self.sim, asset_root, field_asset_file, field_options
        )

        # create robomaster asset
        rm_asset_file = 'RM_description/robot/robomaster.urdf'
        rm_options = gymapi.AssetOptions()
        rm_options.fix_base_link = False
        rm_options.collapse_fixed_joints = True
        rm_asset = self.gym.load_asset(self.sim, asset_root, rm_asset_file, rm_options)
        each_rm_num_dof = self.gym.get_asset_dof_count(rm_asset)

        rm_dof_props = self.gym.get_asset_dof_properties(rm_asset)
        self.rm_gripper_limits = np.zeros((2, 3))  # lower, upper, max effort
        self.rm_wheel_vel_limit = rm_dof_props[2]['velocity']  # max velocity

        for i in range(rm_dof_props.shape[0]):
            if rm_dof_props[i]['hasLimits']:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_EFFORT
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 0.0
                # self.rm_gripper_limits[i][0] = rm_dof_props[i]['lower']
                # self.rm_gripper_limits[i][1] = rm_dof_props[i]['upper']
                # self.rm_gripper_limits[i][2] = rm_dof_props[i]['effort']
            elif rm_dof_props[i]['hasLimits'] and rm_dof_props[i]['friction'] == 0.0:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_VEL
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 500.0
            else:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_NONE

            rm_pose = [gymapi.Transform() for i in range(4)]
            rm_pose[0].p = gymapi.Vec3(-2, -2, 0.1)
            rm_pose[1].p = gymapi.Vec3(-2, 2, 0.1)
            rm_pose[2].p = gymapi.Vec3(2, -2, 0.1)
            rm_pose[3].p = gymapi.Vec3(2, 2, 0.1)
            rm_pose[2].r = gymapi.Quat(0, 0, 1, 0)
            rm_pose[3].r = gymapi.Quat(0, 0, 1, 0)

        # create ball asset
        ball_asset_file = "ball.urdf"
        ball_options = gymapi.AssetOptions()
        ball_options.angular_damping = 0.77
        ball_options.linear_damping = 0.77
        ball_asset = self.gym.load_asset(
            self.sim, asset_root, ball_asset_file, ball_options
        )
        ball_init_pose = gymapi.Transform()
        ball_init_pose.p = gymapi.Vec3(0, 0, 0.2)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0.01)

        # define soccer dof properties
        dof_props = self.gym.get_asset_dof_properties(rm_asset)
        dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
        dof_props['stiffness'][:] = 10000.0
        dof_props['damping'][:] = 500.0

        # generate environments
        self.envs = []
        self.rm_handles = {}
        self.wheel_dof_handles = {}

        self.ball_handles = {}

        for i in range(self.args.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.create_actor(
                env_ptr, field_asset, gymapi.Transform(), "field", i, 0, 0
            )

            self.rm_handles[env_ptr] = []
            self.wheel_dof_handles[env_ptr] = []

            for j in range(self.num_agent * 2):
                rm_handle = self.gym.create_actor(
                    env_ptr, rm_asset, rm_pose[j], "rm" + "_" + str(j), i, 2**(j+1), 0
                )
                self.gym.set_actor_dof_properties(env_ptr, rm_handle, dof_props)
                self.rm_handles[env_ptr].append(rm_handle)

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

            self.envs.append(env_ptr)

            self.wheel_dof_handles_per_env = torch.tensor(
                self.wheel_dof_handles[env_ptr]
            ).reshape(
                self.num_agent * 2 * 4,
            )
            # # tensor([  0,  16,  33,  49,  66,  82,  99, 115, 132, 148, 165, 181, 198, 214, 231, 247])


            self.env_dof_count = self.gym.get_env_dof_count(self.envs[0])

            # Actor root state tensor
            # Shape: (num_env * num_actor, 13)
            self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)
            

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(
            self.viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target
        )

    def local_pos(self, global_pos, robot_xy, rotation_matrix):
        rotated_translation = torch.matmul(
            rotation_matrix, (global_pos - robot_xy).unsqueeze(-1)
        )
        return rotated_translation

    def get_obs(self):
        # Robot: Pos(2), Vel(2), Ori(1), AngularVel(1), Gripper(1) Ball: BallPos(2), BallVel(2)

        obs = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device)

        self.root_positions = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_linvels = self.root_tensor[:, 7:10]
        self.root_angvels = self.root_tensor[:, 10:13]

        import pdb; pdb.set_trace()


        _dof_states = gym.acquire_dof_state_tensor(sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        gym.refresh_dof_state_tensor(sim)

        _rb_states = gym.acquire_rigid_body_state_tensor(sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)
        gym.refresh_rigid_body_state_tensor(sim)

        return None


    def get_reward(self):
        pass

    def reset(self):
        root_tensor = gymtorch.wrap_tensor(_root_tensor)
        saved_root_tensor = root_tensor.clone()
        if step % 100 == 0:
            gym.set_actor_root_state_tensor(
                sim, gymtorch.unwrap_tensor(saved_root_tensor)
            )

        actor_indices = torch.tensor([0, 17, 42], dtype=torch.int32, device="cuda:0")
        gym.set_actor_root_state_tensor_indexed(
            sim, _root_states, gymtorch.unwrap_tensor(actor_indices), 3
        )

        return None

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

    # def mecanum_tranform(self, vel):
    #     action = torch.zeros((len(vel), 4), device=self.device, dtype=torch.float32)
    #     action[:, 0] = vel[:, 0] - vel[:,1] - vel[:, 2]
    #     action[:, 1] = vel[:, 0] + vel[:,1] + vel[:, 2]
    #     action[:, 2] = vel[:, 0] + vel[:,1] - vel[:, 2]
    #     action[:, 3] = vel[:, 0] - vel[:,1] + vel[:, 2]
    #     return action

    def step(self, actions):
        # # Convert actions to tensor
        # actions = torch.tensor(actions, dtype=torch.float32, device=self.args.sim_device)

        # u_wheel, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        # self._vel_control[:, self.control_idx[2:]] = 3*self.mecanum_tranform(u_wheel)
        # u_fingers = torch.ones_like(self._vel_control[:, :2])
        # u_fingers[:, 0] = torch.where(u_gripper >= 0.0, -1, 1)
        # u_fingers[:, 1] = torch.where(u_gripper >= 0.0, 1, -1)
        # self._vel_control[:, self.control_idx[:2]] = u_fingers.clone()
        # self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self._vel_control))

        # # Apply actions to the simulation
        # dof_velocity_target_tensor = torch.zeros((self.args.num_env, self.env_dof_count), device=self.args.sim_device)

        # flattened_action = actions.view(self.args.num_env, -1)

        # dof_velocity_target_tensor[:, self.wheel_handles_per_env] = flattened_action

        # self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.wrap_tensor(dof_velocity_target_tensor))

        # Simulate one step
        self.simulate()
        self.gym.refresh_actor_root_state_tensor(self.sim)  # self.root_tensor

        self.get_obs()

        if not self.args.headless:
            self.render()

        # Compute observations
        # obs = self.get_obs()

        # Compute reward and check if episode is done
        # reward, done = self.compute_reward_and_done()
        reward = 0

        # # Increment episode length counter
        # self.episode_length += 1
        # if self.episode_length >= self.max_episode_length:
        #     done = True

        # # Optionally reset if done
        # if done:
        #     obs = self.reset()

        # return obs, reward, done, {}
        return None


# define reward function using JIT
@torch.jit.script
def compute_reward():
    pass