import torch

def quaternion_to_yaw(q, device):
    quaternions = torch.as_tensor(q, dtype=torch.float32, device=device)
    
    x = quaternions[..., 0]
    y = quaternions[..., 1]
    z = quaternions[..., 2]
    w = quaternions[..., 3]

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    
    return torch.atan2(siny_cosp, cosy_cosp).to(device)

def local_pos(global_pos, robot_xy, rotation_matrix):
  
    rotated_translation = torch.matmul(
        rotation_matrix, (global_pos - robot_xy).unsqueeze(-1)
    )
    return rotated_translation

def mecanum_tranform(vel, num_env, num_agent, device):
    '''
    Velocity limit:
    1. -3.5 <= v_x <= 3.5 m/s
    2. -3.5 <= v_y <= 3.5 m/s
    3. -5π/3 <= w <= 5π/3 rad/s

    RoboMaster EP Parameters:
    1. wheel_radius: r = 0.03 m
    2. 
    '''
    mecanum_vel = torch.zeros((*vel.shape[:-1], 4), dtype=torch.float32, device=device)
    mecanum_vel[..., 0] = vel[..., 0] - vel[..., 1] - vel[..., 2]
    mecanum_vel[..., 1] = vel[..., 0] + vel[..., 1] + vel[..., 2]
    mecanum_vel[..., 2] = vel[..., 0] + vel[..., 1] - vel[..., 2]
    mecanum_vel[..., 3] = vel[..., 0] - vel[..., 1] + vel[..., 2]
    return mecanum_vel