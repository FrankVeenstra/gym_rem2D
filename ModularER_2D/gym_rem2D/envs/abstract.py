#!/usr/bin/env python

"""
Abstract modular environment.

Use this abstraction to implement new environments
"""

from collections import deque
#from gym_rem.morph import Module
import copy
import gym
import logging
import numpy as np
import os.path
import time
# This is the adjusted bipedal walker Box2D environment 
from .Modular2DEnv import Modular2D as m2d
from gym_rem2D.morph import CircularModule2D 
from gym_rem2D.morph import SimpleModule 

# Path to loadable assets
ASSET_PATH = os.path.join(os.path.dirname(__file__), "../../assets")


class ModularEnv2D(gym.Env):
	"""Abstract modular 2D environment"""

	metadata = {'render.modes': ['human'], 'video.frames_per_second': 240}

	def __init__(self):
		self.env = m2d();

		# Create logger for easy logging output
		#self.log = logging.getLogger(self.__class__.__name__)
		# Create pybullet interfaces
		#self.client = pyb.connect(pyb.DIRECT)
		#self._last_render = time.time()
		# Setup information needed for simulation
		#self.dt = 1. / 240.
		#pyb.setAdditionalSearchPath(ASSET_PATH)
		#self._modules = {}
		#self._joints = []
		# Stored for user interactions
		#self.morphology = None
		#self._max_size = None
		# Used for user interaction:
		#self._real_time = False
		# Run setup
		#self.log.info("Creating modular environment")

	def setup(self):
		"""Helper method to initialize default environment"""
		# This is abstracted out from '__init__' because we need to do this
		# first time 'render' is called
		#self.log.debug("Setting up simulation environment")
		#pyb.resetSimulation()
		#pyb.setGravity(0, 0, -9.81)
		# Extract time step for sleep during rendering
		#self.dt = pyb.getPhysicsEngineParameters()['fixedTimeStep']
		# Load ground plane for robots to walk on
		#self.log.debug("Loading ground plane")
		#self.plane_id = pyb.loadURDF('plane/plane.urdf')
		#assert self.plane_id >= 0, "Could not load 'plane.urdf'"
		#self.log.debug("Gym environment setup complete")

	def close(self):
		self.log.debug("Closing environment")
		pyb.disconnect(self.client)

	def reset(self, tree=None, module_list=None):
		# the reset function requires a tree blueprint and the module objects that it can choose from. 
		# This should be given to the reset function from the evolutionary algorithm ('ERmain.py')

		# Reset the environment by running all setup code again
		# self.setup()
		
		if tree is None:
			raise TypeError("The tree blueprint cannot be 'None'!")
		if module_list is None:
			raise TypeError("The module_list cannot be 'None'!")

		# reset the Box2D environment
		self.env.reset(tree, module_list);
		
		return self.observation()

	def step(self, action):
		# The internal state of the morphology is updated when the step function is called in the environment. 
		ob, rew = self.env.step(action)		
		return self.observation(), self.reward(), False, {}

	def observation(self):
		"""Get observation from environment

		No observation is being made yetm this is open to expore in future implementations
		"""
		return None

	def reward(self):
		"""Estimate current reward"""
		# Not set yet

	def render(self, mode="human"):
		self.env.render(mode);

	def handle_interaction(self):
		"""Handle user interaction with simulation

		This function is called in the 'render' call and is only called if the
		GUI is visible"""
		keys = pyb.getKeyboardEvents()
		# If 'n' is pressed then we want to step the simulation
		next_key = ord('n')
		if next_key in keys and keys[next_key] & pyb.KEY_WAS_TRIGGERED:
			pyb.stepSimulation()
		# If space is pressed we start/stop real-time simulation
		space = ord(' ')
		if space in keys and keys[space] & pyb.KEY_WAS_TRIGGERED:
			self._real_time = not self._real_time
			pyb.setRealTimeSimulation(self._real_time)
		# if 'r' is pressed we restart simulation
		r = ord('r')
		if r in keys and keys[r] & pyb.KEY_WAS_TRIGGERED:
			self.reset(self.morphology, self._max_size)
