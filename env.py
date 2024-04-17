from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
from gymnasium.spaces import Box, Discrete, Sequence
import torch


class Soccer:
    def __init__(self, args):
        self.args = args

        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 30.
        sim_params.substeps = 1
        sim_params.use_gpu_pipeline = True
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)


        self.envs = self.create_envs()

        self.num_agent = self.args.n_agent
        self.num_obs =  self.num_agent * 2 * 7 + 4   # Robot: Pos(2), Vel(2), Ori(1), AngularVel(1), Gripper(1) Ball: BallPos(2), BallVel(2)
        self.num_act = self.num_agent * 4  # 控制四个轮子转速

        # Observation space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)

        # Action space
        self.action_space = Box(- self.rm_wheel_vel_limit, self.rm_wheel_vel_limit, (self.num_act * self.num_agent,), dtype=np.float32)




        
        self.max_episode_length = self.args.episode_length


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

        self.actors_per_env = 6
        self.all_soccer_indices = self.actors_per_env * torch.arange(self.args.num_envs, dtype=torch.int32, device=self.args.sim_device)
        self.all_actor_indices = torch.arange(self.args.num_envs * int(self.actors_per_env), dtype=torch.int32, device=self.args.sim_device).view(self.args.num_envs, self.actors_per_env)

        # create field
        asset_root = 'assets'
        field_asset_file = 'field.urdf'
        field_options = gymapi.AssetOptions()
        field_options.fix_base_link = True
        field_options.collapse_fixed_joints = True
        field_asset = self.gym.load_asset(self.sim, asset_root, field_asset_file, field_options)

        # create robomaster asset
        rm_asset_file = '/RM_description/robot/robomaster.urdf'
        rm_options = gymapi.AssetOptions()
        rm_options.fix_base_link = False
        rm_options.collapse_fixed_joints = True
        rm_asset = self.gym.load_asset(self.sim, asset_root, rm_asset_file, rm_options)
        each_rm_num_dof = self.gym.get_asset_dof_count(rm_asset)

        rm_dof_props = self.gym.get_asset_dof_properties(rm_asset)
        self.rm_gripper_limits = np.zeros((2, 3))    # lower, upper, max effort
        self.rm_wheel_vel_limit = rm_dof_props[-1]['velocity']   # max velocity

        for i in range(rm_dof_props.shape[0]):
            if rm_dof_props[i]['hasLimits']:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_EFFORT
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 0.0
                self.rm_gripper_limits[i][0] = rm_dof_props[i]['lower']
                self.rm_gripper_limits[i][1] = rm_dof_props[i]['upper']
                self.rm_gripper_limits[i][2] = rm_dof_props[i]['effort']
            else:
                rm_dof_props[i]['driveMode'] = gymapi.DOF_MODE_VEL
                rm_dof_props[i]['stiffness'] = 0.0
                rm_dof_props[i]['damping'] = 500.0
            

        # create ball asset
        ball_asset_file = "ball.urdf"
        ball_options = gymapi.AssetOptions()
        ball_options.angular_damping = 0.77
        ball_options.linear_damping = 0.77
        ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, ball_options)
        ball_init_pose = gymapi.Transform()
        ball_init_pose.p = gymapi.Vec3(0, 0, 0.08)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0.01)

        # define soccer dof properties
        dof_props = self.gym.get_asset_dof_properties(rm_asset)
        dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
        dof_props['stiffness'][:] = 10000.0
        dof_props['damping'][:] = 500.0

        # generate environments
        envs = []
        self.rm_handles = []
        self.ball_handles = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            self.gym.create_actor(env_ptr, field_asset, gymapi.Transform(), "field", i, 0, 0)

            self.rm_handles.append([])
            for j in range(self.n_agents):
                rm_handle = self.gym.create_actor(env_ptr, rm_asset, pose, "rm"+"_"+str(j), i, 0, 0)
                self.gym.set_actor_dof_properties(env_ptr, rm_handle, dof_props)
                self.rm_handles[i].append(rm_handle)
            
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_init_pose, "ball", i, 1, 0)
            self.ball_handles.append(ball_handle)

            envs.append(env_ptr)

        return envs, each_rm_num_dof

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer

    def local_pos(self, global_pos, robot_xy, rotation_matrix):
        rotated_translation = torch.matmul(rotation_matrix, (global_pos - robot_xy).unsqueeze(-1))
        return rotated_translation
        
    def get_obs(self, env_ids=None):
        pass

    def get_reward(self):
        pass

    def reset(self):
        


        
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

    def step(self, actions):
        # Convert actions to tensor if needed
        actions = torch.tensor(actions, dtype=torch.float32, device=self.args.sim_device)

        # Apply actions to the simulation
        # Placeholder for the action application logic

        # Simulate one step
        self.simulate()
        if not self.args.headless:
            self.render()

        # Compute observations
        obs = self.get_obs()

        # Compute reward and check if episode is done
        reward, done = self.compute_reward_and_done()

        # Increment episode length counter
        self.episode_length += 1
        if self.episode_length >= self.max_episode_length:
            done = True

        # Optionally reset if done
        if done:
            obs = self.reset()

        return obs, reward, done, {}





# define reward function using JIT
@torch.jit.script
def compute_reward(obs_buf, ball_pos, ball_vel, reset_buf, progress_buf, max_episode_length, n_agents):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    goal_reward = 1000.0
    velocity_reward = 10.0
    out_of_field_reward = -10.0
    collision_reward = -0.1
    
    obs_mask = (torch.arange(obs_buf.shape[0]) % (n_agents*2) < n_agents)
    
    # goal reward
    extended_ball_pos = torch.repeat_interleave(ball_pos[:,:], n_agents*2, dim=0)
    extended_ball_pos[~obs_mask, 0] *= -1
    rew_goal = torch.zeros(extended_ball_pos.shape[0], device=obs_buf.device)
    rew_goal = torch.where((extended_ball_pos[:,0] > 4.5) & (torch.abs(extended_ball_pos[:,1]) < 1.3), torch.ones_like(rew_goal)*goal_reward, rew_goal)
    rew_goal = torch.where((extended_ball_pos[:,0] < -4.5) & (torch.abs(extended_ball_pos[:,1]) < 1.3), torch.ones_like(rew_goal)*(-goal_reward), rew_goal)
    
    # ball velocity reward
    extended_ball_vel = torch.repeat_interleave(ball_vel[:,:], n_agents*2, dim=0)
    extended_ball_vel[~obs_mask, 0] *= -1
    backward_ball = extended_ball_vel[:, 0] < 0
    extended_ball_vel[backward_ball, :] = 0.0
    goal_pos = torch.tensor([4.5, 0.0], device=obs_buf.device).repeat(extended_ball_pos.shape[0], 1)
    vectors = (goal_pos - extended_ball_pos)
    norm = torch.norm(vectors)
    unit_vectors = vectors / norm
    unit_vectors_3d = unit_vectors.unsqueeze(1)
    extended_ball_vel_3d = extended_ball_vel.unsqueeze(2)
    dot_products = torch.bmm(unit_vectors_3d, extended_ball_vel_3d).squeeze()
    robot_pos = obs_buf[:, 2:4]
    global_ball = torch.repeat_interleave(ball_pos[:,:], n_agents*2, dim=0)
    local_ball = global_ball - robot_pos
    ball_distances = torch.sum(local_ball**2, dim=1)
    without_1_0m = ball_distances > 1.0**2
    dot_products[without_1_0m] = 0.0
    rew_ball_vel = dot_products * velocity_reward

    # out of field reward
    rew_out_of_field = torch.zeros(obs_buf.shape[0], device=obs_buf.device)
    #robot_pos = obs_buf[:, 2:4]
    out_of_field = (torch.abs(robot_pos[:, 0]) >= 4.9) | (torch.abs(robot_pos[:, 1]) >= 3.4)
    rew_out_of_field[out_of_field] += out_of_field_reward

    # collision reward
    rew_collision = torch.zeros(obs_buf.shape[0], device=obs_buf.device)
    num_others = n_agents * 2 - 1

    robot_positions = obs_buf[:, 5:5+num_others*2].view(-1, num_others, 2)
    collisions = torch.sum(robot_positions**2, dim=2) < (0.2**2)
    collision = torch.any(collisions, dim=1)
    rew_collision[collision] += collision_reward

    reward = rew_goal + rew_ball_vel + rew_out_of_field + rew_collision

    # reset
    reset = torch.where((torch.abs(ball_pos[:,0]) > 4.5) | (torch.abs(ball_pos[:,1]) > 3), torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset)
    return reward, reset, rew_goal, rew_ball_vel, rew_out_of_field, rew_collision
