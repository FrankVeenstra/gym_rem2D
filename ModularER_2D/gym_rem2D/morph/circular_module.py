
#!/usr/bin/env python

"""
Standard BOX 2D module with single joint
"""

import gym_rem2D.morph.module_utility as mu
from gym_rem.utils import Rot

from enum import Enum

import numpy as np
from Controller import m_controller

import random
import math
from gym_rem2D.morph import abstract_module
from gym_rem2D.morph import simple_module as sm
import Box2D as B2D
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

class Connection(Enum):
	"""Available connections for standard 2D module"""
	left = (1.,0.,0.)
	right = (-1.,0.,0.)
	top = (0.,1.0,0.)

class Circular2D(abstract_module.Module):
	"""Standard 2D module"""
	def __init__(self, theta=0, size=(0.1,0.1, 0.0)):
		self.theta = theta % 2 # double check
		self.size = np.array(size)
		assert self.size.shape == (3,), "Size must be a 3 element vector! : this is a 2D module but takes in a three dimensional size vector for now. Third entry is ignored"
		
		self.connection_axis = np.array([0., 0., 1.])
		self.orientation = Rot.from_axis(self.connection_axis,
										 -self.theta * (np.pi / 2.))		
		# NOTE: The fudge factor is to avoid colliding with the plane once
		# spawned
		self.position = np.array([0., self.size[2] / 2. + 0.002, 0.]) # uses only x and y
		self._children = {}
		self.controller = m_controller.Controller()
		# relative scales
		self.radius = 0.25
		self.angle = math.pi/2
		self.type = "CIRCLE"
		self.MIN_RADIUS = 0.25
		self.MAX_RADIUS = 0.5
		self.MIN_ANGLE = math.pi/4
		self.MAX_ANGLE = math.pi*2
		self.torque = 50
		#self.joint = None # needs joint


	def limitWH(self):
		"""Limit morphology to bounds"""
		if self.radius > self.MAX_RADIUS:
			self.radius = self.MAX_RADIUS
		elif self.radius < self.MIN_RADIUS:
			self.radius = self.MIN_RADIUS
		if self.angle >self.MAX_ANGLE:
			self.angle = self.MAX_ANGLE
		elif self.angle < self.MIN_ANGLE:
			self.angle = self.MIN_ANGLE
	def mutate(self, MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA):
		"""
		To mutate the shape and controller stored in the modules. 
		"""
		#return
		if random.uniform(0,1) < MORPH_MUTATION_RATE:
			self.radius = random.gauss(self.radius, MUT_SIGMA)
		if random.uniform(0,1) < MORPH_MUTATION_RATE:
			self.angle = random.gauss(self.angle,MUT_SIGMA * math.pi)
		self.limitWH()
		if self.controller is not None:
			self.controller.mutate(MUTATION_RATE,MUT_SIGMA, self.angle)
	
	def setMorph(self,val1, val2, val3):
		# values are between -1 and 1
		self.radius = val1 + 1.5
		# val2 is not used since radius
		self.angle = self.MIN_ANGLE +(((val3 + 1.0)*0.5) * (self.MAX_ANGLE-self.MIN_ANGLE))
		# limit values
		self.limitWH()

	def __setitem__(self, key, module):
		if not isinstance(key, Connection):
			raise TypeError("Key: '{}' is not a Connection type".format(key))
		if key in self._children:
			raise ModuleAttached()
		if key not in self.available:
			raise ConnectionObstructed()
		# Add module as a child
		self._children[key] = module
		# Calculate connection point
		direction = self.orientation.rotate(np.array(key.value))
		position = self.position + (direction * self.size) / 2.
		# Update parent pointer of module
		module.update(self, position, direction)

	def update(self, parent=None, pos=None, direction=None):
		# Update own orientation first in case we have been previously
		# connected
		self.orientation = Rot.from_axis(self.connection_axis,
										 -self.theta * (np.pi / 2.))
		# Update position in case parent is None
		self.position = np.array([0., 0., self.size[2] / 2. + 0.002])
		# Reset connection in case parent is None
		self.connection = None
		# Call super to update orientation
		super().update(parent, pos, direction)
		# If parent is not None we need to update position and connection point
		if self.parent is not None:
			# Update center position for self
			# NOTE: We add a little fudge factor to avoid overlap
			self.position = pos + (direction * self.size * 1.01) / 2.
			# Calculate connection points for joint
			conn = np.array([0., 0., -self.size[2] / 2.])
			parent_conn = parent.orientation.T.rotate(pos - parent.position)
			self.connection = (parent_conn, conn)
		# Update potential children
		self.update_children()

	def update_children(self):
		for conn in self._children:
			direction = self.orientation.rotate(np.array(conn.value))
			position = self.position + (direction * self.size) / 2.
			self._children[conn].update(self, position, direction)

	def spawn(self):
		orient = self.orientation.as_quat()
		cuid = B2D.b2CircleShape 
		cuid.m_p.Set(self.position)
		if (self.parent):
			self.joint = B2D.b2RevoluteJoint()
		return cuid
	def get_global_position_of_connection_site(self,con=None, parent_component = None):
		if con is None:
			con = Connection.left		# get intersection of rectangle from width and height
		local_position = [] # 2d array
		local_angle = (con.value[0] * (self.angle))  # positive for left, negative for right
		# position relative to y directional vector
		if parent_component:
			local_angle+=parent_component.angle
		x = math.cos(local_angle+ math.pi/2)*self.radius
		y = math.sin(local_angle+ math.pi/2)*self.radius
		local_position.append(x)
		local_position.append(y)
		if parent_component is None:
			return local_position,local_angle
		global_position = [local_position[0]+parent_component.position[0],
					 local_position[1]+parent_component.position[1]]
		
		return global_position, local_angle

	def create(self,world,TERRAIN_HEIGHT,module=None,node=None,connection_site=None, p_c=None, module_list=None, position = None):
		# get module height and width
		if p_c is not None and connection_site is None:
			raise("When you want to attach a new component to a parent component, you have to supply",
			"a connection_site object with it. This connection_site object defines where to anchor",
			"the joint in between to components")
		n_radius = self.radius
		angle = 0

		pos = [7,10,0];
		if position is not None:
			pos = position
		if (p_c is not None):
			local_pos_x =math.cos(connection_site.orientation.x+ math.pi/2) * n_radius 
			local_pos_y =math.sin(connection_site.orientation.x+ math.pi/2) * n_radius 
			pos[0] = (local_pos_x) + connection_site.position.x
			pos[1] = (local_pos_y) + connection_site.position.y

		# This module will create one component that will be temporarily stored in ncomponent
		new_component = None
		# This module will create one joint (if a parent component is present) that will be temporarily stored in njoint
		njoint = None
		
		components = []
		joints = []

		if connection_site:
			angle += connection_site.orientation.x

		if (pos[1] - n_radius < TERRAIN_HEIGHT): #TODO CHANGE TO TERRAIN_HEIGT OR DO CHECK ELSEWHERE
			if node is not None:
				node.component = None
			return components,joints
		else:
			fixture = fixtureDef(
					shape=B2D.b2CircleShape(radius =n_radius),
					density=1,
					friction=0.1,
					restitution=0.0,
					categoryBits=0x0020,
					maskBits=0x001
				)
			new_component = world.CreateDynamicBody(
				position=(pos[0],pos[1]),
				angle = angle,
				fixtures = fixture)

			color = [255,255,255]
			if node is not None and module_list is not None:
				color = world.cmap(node.type/len(module_list))
			elif node is not None and module_list is None:
				print("Note: cannot assign a color to the module since the 'module_list' is not passed as an argument")
			# move to component creator
			new_component.color1 = (color[0],color[1],color[2])
			new_component.color2 = (color[0],color[1],color[2])
			components.append(new_component)
			if node is not None:
				node.component = [new_component]

		if connection_site is not None:
			joint = mu.create_joint(world, p_c,new_component,connection_site, angle, self.torque)
			joints.append(joint)
			

		return components, joints

	