# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
# Adapted by Hanyang Gu for random button spawning

"""Custom MDP functions for button reset with collision avoidance."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "reset_buttons_random",
    "reset_buttons_to_default",
]


def reset_buttons_random(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    min_buttons: int = 1,
    max_buttons: int = 4,
    base_pos: tuple = (-0.35, 0.40, 0.84),
    position_randomization: tuple = (0.15, 0.15, 0.0),  # randomization range for x, y, z
    min_distance: float = 0.12,  # minimum distance between buttons (2x radius + safety margin)
):
    """Reset button positions randomly with collision avoidance.
    
    This function randomly spawns 1-4 buttons on each environment reset, with each button
    placed at a random position that doesn't overlap with other buttons. Inactive buttons
    (when fewer than 4 are spawned) are moved far away from the scene.
    
    Example:
        If 2 buttons are selected:
        - object_0 and object_1 get random positions within the spawn area
        - object_2 and object_3 are moved to [100, 100, -10] (effectively disabled)
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to reset (torch.Tensor of environment indices)
        min_buttons: Minimum number of buttons to spawn (default: 1)
        max_buttons: Maximum number of buttons to spawn (default: 4)
        base_pos: Base position for button spawning area center (x, y, z)
                  Default: (-0.35, 0.40, 0.84) - on the table surface
        position_randomization: Randomization range for each axis (x, y, z)
                                Buttons spawn within base_pos ± randomization
                                Default: (0.15, 0.15, 0.0) - 30cm x 30cm area, no Z variation
        min_distance: Minimum distance between button centers to avoid overlap (meters)
                      Should be at least 2x button radius. Default: 0.12m (button radius=0.045m)
    """
    # Generate random number of buttons for each environment
    num_envs = len(env_ids)
    num_buttons = torch.randint(min_buttons, max_buttons + 1, (num_envs,), device=env.device)
    
    # Get the button objects
    button_objects = []
    for i in range(max_buttons):
        obj_name = f"object_{i}"
        if hasattr(env.scene, obj_name):
            button_objects.append(getattr(env.scene, obj_name))
        else:
            print(f"Warning: {obj_name} not found in scene")
            return
    
    # Process each environment individually
    for idx, env_id in enumerate(env_ids):
        n_buttons = num_buttons[idx].item()
        
        # === STEP 1: Generate collision-free positions for active buttons ===
        positions = []  # Will store positions of successfully placed buttons
        max_attempts = 100  # Retry limit per button to find valid position
        
        for button_idx in range(n_buttons):
            placed = False
            
            # Try to find a valid position that doesn't overlap with existing buttons
            for attempt in range(max_attempts):
                # Generate random offset from base position
                # Range: base_pos ± position_randomization
                rand_offset = torch.tensor([
                    torch.rand(1).item() * 2 * position_randomization[0] - position_randomization[0],  # x offset
                    torch.rand(1).item() * 2 * position_randomization[1] - position_randomization[1],  # y offset
                    torch.rand(1).item() * 2 * position_randomization[2] - position_randomization[2]   # z offset
                ], device=env.device)
                
                new_pos = torch.tensor(base_pos, device=env.device) + rand_offset
                
                # Check if new position collides with any already-placed buttons
                collision = False
                for existing_pos in positions:
                    # Calculate 2D distance (only x-y plane, ignore z)
                    distance = torch.norm(new_pos[:2] - existing_pos[:2])
                    if distance < min_distance:
                        collision = True
                        break
                
                # If no collision, accept this position
                if not collision:
                    positions.append(new_pos)
                    placed = True
                    break
            
            # Fallback: if we couldn't find valid position after max_attempts
            # Place buttons in a line along x-axis with safe spacing
            if not placed:
                fallback_pos = torch.tensor(base_pos, device=env.device)
                fallback_pos[0] += button_idx * 0.15  # 15cm spacing
                positions.append(fallback_pos)
                print(f"Warning: Button {button_idx} in env {env_id} placed using fallback position")
        
        # === STEP 2: Set positions for all 4 button objects ===
        for button_idx in range(max_buttons):
            button = button_objects[button_idx]
            
            if button_idx < n_buttons:
                # Active button - use the collision-free position we generated
                pos = positions[button_idx]
            else:
                # Inactive button - move far away from the scene
                # Position [100, 100, -10] is underground and far from workspace
                # This effectively "disables" the button without removing it from simulation
                pos = torch.tensor([100.0, 100.0, -10.0], device=env.device)
            
            # Construct the root state (position, rotation, velocities)
            root_state = button.data.default_root_state[env_id].clone()
            root_state[:3] = pos  # position (x, y, z)
            root_state[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)  # rotation (identity quaternion)
            root_state[7:] = 0.0  # zero linear and angular velocities
            
            # Apply the state to the simulation
            button.write_root_state_to_sim(root_state.unsqueeze(0), env_ids=torch.tensor([env_id], device=env.device))
    
    # Print info for first environment (for debugging)
    if 0 in env_ids:
        print(f"Reset buttons: spawned {num_buttons[0].item()} buttons in environment 0")


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
        if hasattr(env.scene, obj_name):
            button = getattr(env.scene, obj_name)
            # Reset to default state
            default_state = button.data.default_root_state[env_ids].clone()
            button.write_root_state_to_sim(default_state, env_ids=env_ids)
