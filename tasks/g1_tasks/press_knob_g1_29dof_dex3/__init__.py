
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  

import gymnasium as gym
import os

from . import press_knob_g1_29dof_dex3_joint_env_cfg


gym.register(
    id="Isaac-Press-Knob-G129-Dex3-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": press_knob_g1_29dof_dex3_joint_env_cfg.PressKnobG129DEX3JointEnvCfg,
    },
    disable_env_checker=True,
)

