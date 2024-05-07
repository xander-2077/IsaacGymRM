import torch

def quaternion_to_yaw(q):
    quaternions = torch.as_tensor(q, dtype=torch.float32)
    
    x = quaternions[..., 0]
    y = quaternions[..., 1]
    z = quaternions[..., 2]
    w = quaternions[..., 3]

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    
    return torch.atan2(siny_cosp, cosy_cosp)

def local_pos(global_pos, robot_xy, rotation_matrix):
    '''
    
    '''
    rotated_translation = torch.matmul(
        rotation_matrix, (global_pos - robot_xy).unsqueeze(-1)
    )
    return rotated_translation

def mecanum_tranform(vel, device):
    action = torch.zeros((len(vel), 4), device=device, dtype=torch.float32)
    action[:, 0] = vel[:, 0] - vel[:,1] - vel[:, 2]
    action[:, 1] = vel[:, 0] + vel[:,1] + vel[:, 2]
    action[:, 2] = vel[:, 0] + vel[:,1] - vel[:, 2]
    action[:, 3] = vel[:, 0] - vel[:,1] + vel[:, 2]
    return action
