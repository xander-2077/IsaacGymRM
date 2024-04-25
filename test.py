import isaacgym
from isaacgym import gymapi
import numpy as np

def main():
    num_envs = 2

    # 初始化Isaac Gym
    gym = isaacgym.gymapi.acquire_gym()

    # configure sim (gravity is pointing down)
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 1
    # sim_params.use_gpu_pipeline = True

    # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.rest_offset = 0.001
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(0, 0, isaacgym.gymapi.SIM_PHYSX, sim_params)

    viewer = gym.create_viewer(sim, isaacgym.gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, isaacgym.gymapi.Vec3(2, 2, 2), isaacgym.gymapi.Vec3(0, 0, 0))


    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.distance = 0
    gym.add_ground(sim, plane_params)

    # define environment space (for visualisation)
    lower = gymapi.Vec3(0, 0, 0)
    upper = gymapi.Vec3(12, 9, 5)
    num_per_row = int(np.sqrt(num_envs))

    # create field
    asset_root = 'assets'
    field_asset_file = 'field.urdf'
    field_options = gymapi.AssetOptions()
    field_options.fix_base_link = True
    field_options.collapse_fixed_joints = True
    field_asset = gym.load_asset(sim, asset_root, field_asset_file, field_options)

    # # create robomaster asset
    rm_asset_file = 'RM_description/robot/robomaster.urdf'
    rm_options = gymapi.AssetOptions()
    rm_options.fix_base_link = False
    rm_options.disable_gravity = False
    rm_options.collapse_fixed_joints = False
    rm_asset = gym.load_asset(sim, asset_root, rm_asset_file, rm_options)
    each_rm_num_dof = gym.get_asset_dof_count(rm_asset)

    # create ball asset
    ball_asset_file = "ball.urdf"
    ball_options = gymapi.AssetOptions()
    # ball_options.angular_damping = 0.77
    # ball_options.linear_damping = 0.77
    ball_asset = gym.load_asset(sim, asset_root, ball_asset_file, ball_options)
    ball_init_pose = gymapi.Transform()
    ball_init_pose.p = gymapi.Vec3(0, 0, 0.1)

    # define cartpole pose
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.01)

    # define soccer dof properties
    dof_props = gym.get_asset_dof_properties(rm_asset)
    print(dof_props)
    dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
    dof_props['stiffness'][:] = 10000.0
    dof_props['damping'][:] = 500.0

    # generate environments
    envs = []
    rm_handles = []
    ball_handles = []

    rm_pose = [gymapi.Transform() for i in range(4)]
    rm_pose[0].p = gymapi.Vec3(-2, -2, 0.1)
    rm_pose[1].p = gymapi.Vec3(-2, 2, 0.1)
    rm_pose[2].p = gymapi.Vec3(2, -2, 0.1)
    rm_pose[3].p = gymapi.Vec3(2, 2, 0.1)
    rm_pose[2].r = gymapi.Quat(0, 0, 1, 0)
    rm_pose[3].r = gymapi.Quat(0, 0, 1, 0)

    # create env
    for i in range(num_envs):
        env = gym.create_env(sim, lower, upper, num_per_row)

        # rm_handles.append([])
        for j in range(4):
            rm_handle = gym.create_actor(env, rm_asset, rm_pose[j], "rm"+"_"+str(j), i, 1, 0)
            gym.set_actor_dof_properties(env, rm_handle, dof_props)
            wheel_joint_dof = gym.find_actor_dof_handle(env, rm_handle, "front_left_wheel_joint")
            print(wheel_joint_dof)
            # gym.set_dof_target_velocity(env, wheel_joint_dof, 5.0)

            # rm_handles[i].append(rm_handle)
            # dof_dict = gym.get_actor_dof_dict(env, rm_handle)
            # gym.set_actor_dof_position_target(env, rm_handle, dof_dict['rm_joint1'], 0.0)

        gym.create_actor(env, field_asset, gymapi.Transform(), "field", i, 1, 0)

        ball_handle = gym.create_actor(env, ball_asset, ball_init_pose, "ball", i, 1, 0)
        # ball_handles.append(ball_handle)

        # envs.append(env)

    # body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
    # body_states['pose']
    # body_states['pose']['p']
    # body_states['vel']
    # dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
    # dof_positions = dof_states['pos']

    # 主循环
    while not gym.query_viewer_has_closed(viewer):
    


        # 步进仿真
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 渲染
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # 同步渲染
        gym.sync_frame_time(sim)

    # 清理资源
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()