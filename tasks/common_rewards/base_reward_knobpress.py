# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
import os
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
# global variable to cache the DDS instance
_rewards_dds = None
_dds_initialized = False
import sys
import os
def _get_rewards_dds_instance():
    """get the DDS instance, delay initialization"""
    global _rewards_dds, _dds_initialized
    
    if not _dds_initialized or _rewards_dds is None:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            
            _rewards_dds = dds_manager.get_object("rewards")
            print("[Observations Rewards] DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _rewards_dds:
                        dds_manager.unregister_object("rewards")
                        print("[rewards_dds] DDS communication closed correctly")
                except Exception as e:
                    print(f"[rewards_dds] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[Observations Rewards] Failed to get DDS instances: {e}")
            _rewards_dds = None
        
        _dds_initialized = True
    
    return _rewards_dds

def compute_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    initial_height: float = 0.84,           # initial knob height (from scene config)
    press_threshold: float = 0.005,          # minimum press depth to count as pressed (5mm)
    target_press_depth: float = 0.015,       # optimal press depth for maximum reward (15mm)
    max_press_depth: float = 0.03,           # maximum allowable press depth (30mm)
    position_tolerance: float = 0.05,        # XY position tolerance from initial position
    initial_x: float = -0.35,                # initial X position from scene config
    initial_y: float = 0.40,                 # initial Y position from scene config
) -> torch.Tensor:
    """Compute reward for knob pressing task based on height-based detection.
    
    Reward structure:
    - Negative reward if knob moves too far in XY plane (not a valid press)
    - Zero reward if not pressed enough
    - Positive reward proportional to press depth up to target depth
    - Maximum reward at target press depth
    - Reduced reward if pressed beyond target (too hard)
    """
    # Handle reward interval caching
    interval = getattr(env, "_reward_interval", 1) or 1
    counter = getattr(env, "_reward_counter", 0)
    last = getattr(env, "_reward_last", None)
    if interval > 1 and last is not None and counter % interval != 0:
        env._reward_counter = counter + 1
        return last

    # 1. Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]
    
    # 2. Get current object position
    knob_x = object.data.root_pos_w[:, 0]      # x position
    knob_y = object.data.root_pos_w[:, 1]      # y position
    knob_height = object.data.root_pos_w[:, 2] # z position (height)
    
    # 3. Calculate press depth (how much the knob moved down)
    press_depth = initial_height - knob_height
    
    # 4. Calculate XY displacement from initial position
    x_displacement = torch.abs(knob_x - initial_x)
    y_displacement = torch.abs(knob_y - initial_y)
    xy_displaced = (x_displacement > position_tolerance) | (y_displacement > position_tolerance)
    
    # 5. Create reward tensor
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    
    # 6. Apply reward logic
    # Case 1: Knob moved too far in XY (fell off or pushed aside) - negative reward
    reward[xy_displaced] = -1.0
    
    # Case 2: Not pressed enough - zero reward
    not_pressed = (press_depth < press_threshold) & ~xy_displaced
    reward[not_pressed] = 0.0
    
    # Case 3: Pressed within valid range - positive reward
    pressed = (press_depth >= press_threshold) & (press_depth <= max_press_depth) & ~xy_displaced
    
    # Sub-case 3a: Pressed up to target depth - linearly increasing reward
    good_press = pressed & (press_depth <= target_press_depth)
    reward[good_press] = (press_depth[good_press] - press_threshold) / (target_press_depth - press_threshold)
    
    # Sub-case 3b: At or near target depth - maximum reward
    optimal_press = pressed & (press_depth >= target_press_depth) & (press_depth <= target_press_depth + 0.005)
    reward[optimal_press] = 1.0
    
    # Sub-case 3c: Pressed beyond target (too hard) - reduced reward
    over_press = pressed & (press_depth > target_press_depth + 0.005)
    reward[over_press] = 1.0 - 0.5 * ((press_depth[over_press] - target_press_depth) / (max_press_depth - target_press_depth))
    
    # Case 4: Pressed too hard (beyond max depth) - negative reward
    too_hard = (press_depth > max_press_depth) & ~xy_displaced
    reward[too_hard] = -0.5

    # Cache reward for interval optimization
    env._reward_last = reward
    env._reward_counter = counter + 1
    
    # Send reward data via DDS if available
    rewards_dds = _get_rewards_dds_instance()
    if rewards_dds:
        rewards_dds.write_rewards_data(reward)
    
    return reward
