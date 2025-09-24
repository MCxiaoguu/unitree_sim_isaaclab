# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import gymnasium as gym
import os

from . import pickplace_redblock_h12_26dof_inspire_joint_env_cfg


gym.register(
    id="Isaac-PickPlace-RedBlock-H12-26dof-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_redblock_h12_26dof_inspire_joint_env_cfg.PickPlaceH1226dofInspireHandBaseFixEnvCfg,
    },
    disable_env_checker=True,
)

