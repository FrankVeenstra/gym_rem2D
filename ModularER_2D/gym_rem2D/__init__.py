#!/usr/bin/env python

from gym.envs.registration import register

register(id="Modular2DLocomotion-v0",
         entry_point="gym_rem2D.envs:Modular2D",
         max_episode_steps=240 * 20)

