#!/usr/bin/env python

from gym.envs.registration import register

register(id="ModularLocomotion-v0",
         entry_point="gym_rem.envs:ModularEnv",
         max_episode_steps=240 * 20)

