import torch

def quaternion_to_yaw(q):
        if q.dim() == 2:
            w = q[:, 0]
            x = q[:, 1]
            y = q[:, 2]
            z = q[:, 3]
            z_x = 2 * (x * z + w * y)
            z_y = 2 * (y * z - w * x)
            yaw = torch.atan2(z_y, z_x)
            return yaw
        elif q.dim() == 1: 
            w = q[0]
            x = q[1]
            y = q[2]
            z = q[3]
            z_x = 2 * (x * z + w * y)
            z_y = 2 * (y * z - w * x)
            yaw = torch.atan2(z_y, z_x)
            return yaw

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
