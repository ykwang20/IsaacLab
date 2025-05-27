import torch

def data_augmentation_func_g1(obs, actions, env, is_critic):
    if obs is not None:
        obs_batch = mirror_xz_plane(obs)
        obs_batch =torch.cat((obs,obs_batch),dim=0)
    else:
        obs_batch = None
    if actions is not None:
        mean_actions_batch = mirror_action_xz_plane(actions)
        action_batch=torch.cat((actions,mean_actions_batch),dim=0)
    else:
        action_batch = None
    return obs_batch, action_batch


def mirror_xz_plane(observation):
    """
    Perform x-z plane symmetry transformation on observation tensor.
    The observation is composed of 6 stacked frames of the same format.
    
    Args:
        observation: PyTorch tensor with shape [..., obs_dim] where obs_dim = 6 * single_frame_dim
        
    Returns:
        PyTorch tensor with mirrored observation
    """
    
    # Create a copy of the observation
    mirrored_obs = observation.clone()
    
    # Define joint mapping for left-right symmetry
    joint_map = {
        0: 1,   # left_hip_pitch_joint -> right_hip_pitch_joint
        1: 0,   # right_hip_pitch_joint -> left_hip_pitch_joint
        3: 4,   # left_hip_roll_joint -> right_hip_roll_joint
        4: 3,   # right_hip_roll_joint -> left_hip_roll_joint
        6: 7,   # left_hip_yaw_joint -> right_hip_yaw_joint
        7: 6,   # right_hip_yaw_joint -> left_hip_yaw_joint
        9: 10,  # left_knee_joint -> right_knee_joint
        10: 9,  # right_knee_joint -> left_knee_joint
        11: 12, # left_shoulder_pitch_joint -> right_shoulder_pitch_joint
        12: 11, # right_shoulder_pitch_joint -> left_shoulder_pitch_joint
        13: 14, # left_ankle_pitch_joint -> right_ankle_pitch_joint
        14: 13, # right_ankle_pitch_joint -> left_ankle_pitch_joint
        15: 16, # left_shoulder_roll_joint -> right_shoulder_roll_joint
        16: 15, # right_shoulder_roll_joint -> left_shoulder_roll_joint
        17: 18, # left_ankle_roll_joint -> right_ankle_roll_joint
        18: 17, # right_ankle_roll_joint -> left_ankle_roll_joint
        19: 20, # left_shoulder_yaw_joint -> right_shoulder_yaw_joint
        20: 19, # right_shoulder_yaw_joint -> left_shoulder_yaw_joint
        21: 22, # left_elbow_joint -> right_elbow_joint
        22: 21, # right_elbow_joint -> left_elbow_joint
        # Joints to keep as is (central joints)
        2: 2,   # waist_yaw_joint
        5: 5,   # waist_roll_joint
        8: 8,   # waist_pitch_joint
    }
    
    # Define which joints need sign flipping when mirrored
    flip_sign_joints = {
        # Roll joints
        3, 4, 5,          # hip roll and waist roll
        15, 16, 17, 18,   # shoulder roll and ankle roll
        
        # Yaw joints
        2, 6, 7,          # waist yaw and hip yaw
        19, 20,           # shoulder yaw
    }
    
    # Calculate dimensions of a single frame
    n_joints = 23
    # Single frame components: root_lin_vel(3) + root_ang_vel(3) + base_quat(4) + base_pos(3) + 
    # joint_pos(23) + joint_vel(23) + actions(23) + box_height(1) 
    single_frame_dim = 3 + 3 + 4 + 3 + n_joints + n_joints + n_joints + 1 

    # # Single frame components: root_lin_vel(3) + root_ang_vel(3) + proj_grav(3) + base_pos(3) + 
    # # joint_pos(23) + joint_vel(23) + actions(23) + box_height(1) 
    # single_frame_dim = 3 + 3 + 3 + 3 + n_joints + n_joints + n_joints + 1 
    
    # Process each of the 6 frames
    for frame in range(6):
        # Calculate starting index for this frame
        frame_start = frame * single_frame_dim
        
        # Current position in the frame
        pos = 0
        
        # root_lin_vel - flip y component (second element)
        mirrored_obs[..., frame_start + pos + 1] = -mirrored_obs[..., frame_start + pos + 1]  # Flip y
        pos += 3  # Move to next component
        
        # root_ang_vel - flip x and z components for xyz order
        mirrored_obs[..., frame_start + pos] = -mirrored_obs[..., frame_start + pos]  # Flip x angular velocity
        mirrored_obs[..., frame_start + pos + 2] = -mirrored_obs[..., frame_start + pos + 2]  # Flip z angular velocity
        pos += 3  # Move to next component
        
        # # projected gravity - flip y component (second element)
        # mirrored_obs[..., frame_start + pos + 1] = -mirrored_obs[..., frame_start + pos + 1]  # Flip y component
        # pos += 3  # Move to next component

        # base_quat - (w, x, y, z) order
        mirrored_obs[..., frame_start + pos + 2] = -mirrored_obs[..., frame_start + pos + 2]  # Flip y component of quaternion
        pos += 4  # Move to next component
        
        # base_pos - keep unchanged as requested
        pos += 3  # Move to next component
        
        # # target_commands - keep unchanged as requested
        # pos += 3  # Move to next component
        
        # joint_pos - remap joints and flip signs as needed
        joint_pos_temp = mirrored_obs[..., frame_start + pos:frame_start + pos + n_joints].clone()
        
        for i in range(n_joints):
            # Map joint position to its mirrored counterpart
            mirrored_obs[..., frame_start + pos + i] = joint_pos_temp[..., joint_map[i]]
            # Flip sign if needed
            if i in flip_sign_joints:
                mirrored_obs[..., frame_start + pos + i] = -mirrored_obs[..., frame_start + pos + i]
        
        pos += n_joints  # Move to next component
        
        # joint_vel - same mapping as joint_pos
        joint_vel_temp = mirrored_obs[..., frame_start + pos:frame_start + pos + n_joints].clone()
        
        for i in range(n_joints):
            # Map joint velocity to its mirrored counterpart
            mirrored_obs[..., frame_start + pos + i] = joint_vel_temp[..., joint_map[i]]
            # Flip sign if needed
            if i in flip_sign_joints:
                mirrored_obs[..., frame_start + pos + i] = -mirrored_obs[..., frame_start + pos + i]
        
        pos += n_joints  # Move to next component
        
        # actions - same mapping as joint_pos
        action_temp = mirrored_obs[..., frame_start + pos:frame_start + pos + n_joints].clone()
        
        for i in range(n_joints):
            # Map action to its mirrored counterpart
            mirrored_obs[..., frame_start + pos + i] = action_temp[..., joint_map[i]]
            # Flip sign if needed
            if i in flip_sign_joints:
                mirrored_obs[..., frame_start + pos + i] = -mirrored_obs[..., frame_start + pos + i]
        
        pos += n_joints  # Move to next component
        
        # box_height - scalar, no change needed
        pos += 1  # Move to next component
        
        # time - scalar, no change needed
    
    return mirrored_obs


def mirror_action_xz_plane(action):
    """
    Perform x-z plane symmetry transformation on action tensor.
    
    Args:
        action: PyTorch tensor with shape [..., n_joints]
        
    Returns:
        PyTorch tensor with mirrored action
    """
    
    # Create a copy of the action
    mirrored_action = action.clone()
    
    # Define joint mapping for left-right symmetry
    joint_map = {
        0: 1,   # left_hip_pitch_joint -> right_hip_pitch_joint
        1: 0,   # right_hip_pitch_joint -> left_hip_pitch_joint
        3: 4,   # left_hip_roll_joint -> right_hip_roll_joint
        4: 3,   # right_hip_roll_joint -> left_hip_roll_joint
        6: 7,   # left_hip_yaw_joint -> right_hip_yaw_joint
        7: 6,   # right_hip_yaw_joint -> left_hip_yaw_joint
        9: 10,  # left_knee_joint -> right_knee_joint
        10: 9,  # right_knee_joint -> left_knee_joint
        11: 12, # left_shoulder_pitch_joint -> right_shoulder_pitch_joint
        12: 11, # right_shoulder_pitch_joint -> left_shoulder_pitch_joint
        13: 14, # left_ankle_pitch_joint -> right_ankle_pitch_joint
        14: 13, # right_ankle_pitch_joint -> left_ankle_pitch_joint
        15: 16, # left_shoulder_roll_joint -> right_shoulder_roll_joint
        16: 15, # right_shoulder_roll_joint -> left_shoulder_roll_joint
        17: 18, # left_ankle_roll_joint -> right_ankle_roll_joint
        18: 17, # right_ankle_roll_joint -> left_ankle_roll_joint
        19: 20, # left_shoulder_yaw_joint -> right_shoulder_yaw_joint
        20: 19, # right_shoulder_yaw_joint -> left_shoulder_yaw_joint
        21: 22, # left_elbow_joint -> right_elbow_joint
        22: 21, # right_elbow_joint -> left_elbow_joint
        # Joints to keep as is (central joints)
        2: 2,   # waist_yaw_joint
        5: 5,   # waist_roll_joint
        8: 8,   # waist_pitch_joint
    }
    
    # Define which joints need sign flipping when mirrored
    flip_sign_joints = {
        # Roll joints
        3, 4, 5,          # hip roll and waist roll
        15, 16, 17, 18,   # shoulder roll and ankle roll
        
        # Yaw joints
        2, 6, 7,          # waist yaw and hip yaw
        19, 20,           # shoulder yaw
    }
    
    # Temporary copy to avoid overwriting values before using them
    action_temp = mirrored_action.clone()
    
    # Remap joints and flip signs as needed
    for i in range(action.shape[-1]):
        # Map action to its mirrored counterpart
        mirrored_action[..., i] = action_temp[..., joint_map[i]]
        # Flip sign if needed
        if i in flip_sign_joints:
            mirrored_action[..., i] = -mirrored_action[..., i]
    
    return mirrored_action