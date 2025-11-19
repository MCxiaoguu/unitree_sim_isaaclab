# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
# Adapted by Hanyang Gu for random button spawning

"""Custom MDP functions for button reset with collision avoidance and dynamic spawning."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

__all__ = [
    "reset_buttons_random",
    "reset_buttons_to_default",
    "reset_robot_only",
    "reset_robot_and_buttons",
]


def reset_buttons_random(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    min_buttons: int = 1,
    max_buttons: int = 4,
    base_pos: tuple = (-0.35, 0.40, 0.84),
    position_randomization: tuple = (0.15, 0.15, 0.0),
    min_distance: float = 0.12,
):
    """Reset button positions randomly with collision avoidance.
    
    This function uses vectorized operations following Isaac Lab best practices.
    """
    #NEVER USE "{object} in env.scene" to check existence, it is known to be buggy.
    num_envs = len(env_ids)
    
    # Generate random number of buttons for each environment
    num_buttons_per_env = torch.randint(min_buttons, max_buttons + 1, (num_envs,), device=env.device)
    print(f"env.scene: {env.scene}")
    # Get all button objects
    button_objects = []
    for i in range(max_buttons):
        print(f"looking for object_{i}")
        obj_name = f"object_{i}"
        try:
            button = env.scene[obj_name]
            button_objects.append(button)
            print(f"[Button Reset] Found {obj_name}")
        except (KeyError, AttributeError) as e:
            print(f"[Button Reset] ERROR: {obj_name} not found in scene: {e}")
            return
    
    print(f"converting parameters to tensors")
    # Convert parameters to tensors
    base_pos_tensor = torch.tensor(base_pos, device=env.device, dtype=torch.float32)
    pos_rand = torch.tensor(position_randomization, device=env.device, dtype=torch.float32)
    
    # Pre-allocate position tensors for all buttons and all environments
    # Shape: (max_buttons, num_envs, 3)
    all_button_positions = torch.zeros((max_buttons, num_envs, 3), device=env.device)
    
    # Generate positions for each environment (unfortunately can't fully vectorize collision check)
    print(f"[Button Reset] Processing {num_envs} environments")
    for env_idx in range(num_envs):
        n_buttons = num_buttons_per_env[env_idx].item()
        positions = []
        max_attempts = 100
        print(f"[Button Reset] Env {env_idx}: Placing {n_buttons} buttons")
        
        # Generate collision-free positions for active buttons
        for button_idx in range(n_buttons):
            placed = False
            
            for attempt in range(max_attempts):
                # Random offset: [-pos_rand, +pos_rand]
                rand_offset = (torch.rand(3, device=env.device) * 2 - 1) * pos_rand
                new_pos = base_pos_tensor + rand_offset
                
                # Clamp Z to ensure button stays on table (avoid falling through)
                new_pos[2] = torch.clamp(new_pos[2], min=0.82, max=0.86)
                
                # Check collision with existing buttons (2D distance in X-Y plane)
                collision = False
                for existing_pos in positions:
                    distance = torch.norm(new_pos[:2] - existing_pos[:2])
                    if distance < min_distance:
                        collision = True
                        break
                
                if not collision:
                    positions.append(new_pos)
                    placed = True
                    print(f"  Button {button_idx} placed at [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}] (attempt {attempt+1})")
                    break
            
            # Fallback position if couldn't find collision-free spot
            if not placed:
                fallback_pos = base_pos_tensor.clone()
                fallback_pos[0] += button_idx * 0.15  # Spread along X axis
                fallback_pos[2] = 0.84  # Ensure on table
                positions.append(fallback_pos)
                print(f"  Button {button_idx} FALLBACK at [{fallback_pos[0]:.3f}, {fallback_pos[1]:.3f}, {fallback_pos[2]:.3f}]")
        
        # Store positions for all buttons (active + inactive)
        for button_idx in range(max_buttons):
            if button_idx < n_buttons:
                all_button_positions[button_idx, env_idx] = positions[button_idx]
            else:
                # Inactive button - move far BELOW the ground to ensure it's truly hidden
                # Using large negative Z to put it well below any possible workspace
                inactive_pos = torch.tensor([0.0, 0.0, -100.0], device=env.device)
                all_button_positions[button_idx, env_idx] = inactive_pos
                print(f"  Button {button_idx} DISABLED (moved to underground)")
    
    # Now apply all positions at once for each button (vectorized across environments)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    zero_vel = torch.zeros(6, device=env.device)
    
    for button_idx, button in enumerate(button_objects):
        # Get positions for this button across all environments
        positions = all_button_positions[button_idx]  # Shape: (num_envs, 3)
        
        # Create orientations (identity quaternion for all envs)
        orientations = identity_quat.unsqueeze(0).repeat(num_envs, 1)
        
        # Combine position + orientation into full root state
        poses = torch.cat([positions, orientations], dim=-1)  # Shape: (num_envs, 7)
        
        # Create velocities (zero for all envs)
        velocities = zero_vel.unsqueeze(0).repeat(num_envs, 1)  # Shape: (num_envs, 6)
        
        # Combine into full root state [pos(3) + quat(4) + lin_vel(3) + ang_vel(3)]
        full_root_state = torch.cat([poses, velocities], dim=-1)  # Shape: (num_envs, 13)
        
        # Write complete root state to simulation in one call
        # This ensures physics engine properly updates the object
        button.write_root_state_to_sim(full_root_state, env_ids=env_ids)
    
    print(f"[Button Reset] Randomized buttons for {num_envs} environments")


def reset_robot_only(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
):
    """Reset only the robot to default pose, leaving buttons untouched.
    
    Uses Isaac Lab's vectorized approach for efficiency.
    """
    
    if "robot" not in env.scene:
        print("[Robot Reset] ERROR: 'robot' not found in scene")
        return
    
    robot = env.scene["robot"]
    
    # Get default states for all environments at once (vectorized)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    
    # Write to simulation (vectorized)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    # Reset root state if robot has one (vectorized)
    if hasattr(robot.data, 'default_root_state'):
        try:
            # Get default root state for all envs
            root_states = robot.data.default_root_state[env_ids].clone()
            
            # Split into pose and velocity
            poses = root_states[:, :7]  # position (3) + orientation (4)
            velocities = root_states[:, 7:13]  # linear (3) + angular (3)
            
            # Write to simulation (vectorized)
            robot.write_root_pose_to_sim(poses, env_ids=env_ids)
            robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        except Exception as e:
            print(f"[Robot Reset] Could not reset root state: {e}")
    
    print(f"[Robot Reset] Reset robot for {len(env_ids)} environments")


def reset_robot_and_buttons(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    min_buttons: int = 1,
    max_buttons: int = 4,
    base_pos: tuple = (-0.35, 0.40, 0.84),
    position_randomization: tuple = (0.15, 0.15, 0.0),
    min_distance: float = 0.12,
):
    """Reset both robot and buttons (full reset).
    
    This is a convenience function that calls both reset functions.
    """
    # First reset robot
    reset_robot_only(env, env_ids)
    
    # Then randomize buttons
    reset_buttons_random(
        env, env_ids,
        min_buttons=min_buttons,
        max_buttons=max_buttons,
        base_pos=base_pos,
        position_randomization=position_randomization,
        min_distance=min_distance
    )
    
    print(f"[Full Reset] Reset robot and randomized buttons")

def reset_buttons_to_default(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    num_buttons: int = 4,
):
    """Reset all buttons to their default positions.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to reset
        num_buttons: Number of button objects in the scene (default: 4)
    """
    for i in range(num_buttons):
        obj_name = f"object_{i}"
        if obj_name in env.scene:
            button = env.scene[obj_name]
            default_state = button.data.default_root_state[env_ids].clone()
            button.write_root_state_to_sim(default_state, env_ids=env_ids)
    
    print(f"[Button Reset] All buttons reset to default positions")