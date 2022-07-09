#!/usr/bin/env python

"""
Standard 2D module with single joint
"""

#from gym_rem.morph.exception import ModuleAttached, ConnectionObstructed
#from gym_rem.morph.module import Module

from gym_rem2D.morph import abstract_module
from enum import Enum

import numpy as np
import Box2D as B2D
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

from Controller import m_controller

import random
import math

class Connection(Enum):
	"""Available connections for standard 2D module"""
	left = (-1.,0.,0.)
	right = (1.,0.,0.)
	top = (0.,1.0,0.)

class Standard2D(abstract_module.Module):
	"""Standard 2D module"""
	def __init__(self, theta=0, size=(0.1,0.1, 0.0)):
		
		self.theta = theta % 2 # double check
		self.size = np.array(size)
		assert self.size.shape == (3,), "Size must be a 3 element vector! : this is a 2D module but takes in a three dimensional size vector for now. Third entry is ignored"

		self.position = np.array([0., self.size[2] / 2. + 0.002, 0.]) # uses only x and y
		self.connection_type = Connection
		self._children = {}
		self.controller = m_controller.Controller()

		# relative scales
		self.width = 0.2 
		self.height = 0.8
		self.angle = math.pi/2

		self.type = "SIMPLE"
		self.MAX_HEIGHT = 1.0
		self.MIN_HEIGHT = 0.5
		self.MAX_WIDTH = 1.0
		self.MIN_WIDTH = 0.5
		self.MAX_ANGLE = math.pi
		self.MIN_ANGLE = 0
		self.torque = 50
		#self.joint = None # needs joint
	
	def limitWH(self):
		"""Limit morphology to bounds"""
		if self.height > self.MAX_HEIGHT:
			self.height = self.MAX_HEIGHT
		elif self.height < self.MIN_HEIGHT:
			self.height = self.MIN_HEIGHT
		if self.width > self.MAX_WIDTH:
			self.width = self.MAX_WIDTH
		elif self.width < self.MIN_WIDTH:
			self.width = self.MIN_WIDTH
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
			self.width = random.gauss(self.width,MUT_SIGMA)
			#self.height = self.width
		if random.uniform(0,1) < MORPH_MUTATION_RATE:
			self.height = random.gauss(self.height,MUT_SIGMA)
		if random.uniform(0,1) < MORPH_MUTATION_RATE:
			self.angle = random.gauss(self.angle,MUT_SIGMA) * math.pi
		# function below ensures values are not out of bound
		self.limitWH()
		if self.controller:
			self.controller.mutate(MUTATION_RATE,MUT_SIGMA, self.angle)
	
	def setMorph(self,val1, val2, val3):
		# val1 and val2 are between -1 and 1
		self.width = (val1 * 0.5* (self.MAX_WIDTH-self.MIN_WIDTH)) + 0.5*(self.MAX_WIDTH-self.MIN_WIDTH)
		self.height = (val1 * 0.5* (self.MAX_HEIGHT-self.MIN_HEIGHT)) + 0.5*(self.MAX_HEIGHT-self.MIN_HEIGHT)
		self.angle = self.MIN_ANGLE +(((val3 + 1.0)*0.5) * (self.MAX_ANGLE-self.MIN_ANGLE))
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
		# Update own orientation first in case we have been previously connected
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

	# Spawn function is not used here
	def spawn(self):
		orient = self.orientation.as_quat()
		cuid = B2D.b2CircleShape 
		cuid.m_p.Set(self.position)
		if (self.parent):
			self.joint = B2D.b2RevoluteJoint()
		return cuid

	def get_global_position_of_connection_site(self,con=None, parent_component = None):
		# get intersection of rectangle from width and height
		if con is None:
			con = Connection.left
		local_position = [] # 2d array
		local_angle = (con.value[0] * (self.angle) + math.pi/2)# + math.pi/2 # positive for left, negative for right
		angle = local_angle 
		while (angle > 2*math.pi):
			angle-=2*PI
		#angle = 0.7
		z1AD = 1.
		z2AD = 1.
		if angle > 0.5*math.pi and angle < 1.5 * math.pi:
			z1AD = -1.
		if angle > math.pi and angle < 2* math.pi:
			z2AD = -1.

		z1 = []
		if 2*math.sin(angle) == 0:
			z1.append(10000)
			z1.append(10000)
		else:
			z1.append((self.height*math.cos(angle))/(2*math.sin(angle))*z2AD);
			z1.append(self.height/2*z2AD);
		z2 = []
		if 2*math.cos(angle)==0:
			z2.append(10000)
			z2.append(10000)
		else:
			z2.append(self.width/2 * z1AD);
			z2.append((self.width*math.sin(angle))/(2*math.cos(angle))* z1AD);

		dis1 = math.sqrt(math.pow(z1[0],2)+math.pow(z1[1],2))
		dis2 = math.sqrt(math.pow(z2[0],2)+math.pow(z2[1],2))
		
		pangle = 0
		ppos = []
		if parent_component is not None:
			pangle = parent_component.angle
			ppos = parent_component.position
		else:
			raise("Parent component is none...")
		
		distance = dis1
		if dis2 < dis1:
			distance = dis2

		global_position = [
			(math.cos(pangle+angle) * distance) + ppos[0],
			(math.sin(pangle+angle) * distance) + ppos[1]
			]
		global_angle = pangle + angle - math.pi/2
		return global_position, global_angle

		if (dis1 < dis2):
			local_position.append(math.cos(angle+pangle)*dis1)
			local_position.append(math.sin(angle+pangle)*dis1)
		else:
			local_position.append(math.cos(angle+pangle)*dis2)
			local_position.append(math.sin(angle+pangle)*dis2)
		return local_position,local_angle

		# position relative to y directional vector
		dis = math.sqrt(pow(self.width,2)+pow(self.height,2))
		x = math.cos(local_angle+ parent_angle)*dis*2
		y = math.sin(local_angle+ parent_angle)*dis*2
		#if math.sqrt(math.pow(x,2))/self.width > math.sqrt(math.pow(y,2))/self.height:
			# we know that the x axis would be crossed first
			#x = self.width
			#y = 0.0# math.tanh(local_angle+parent_angle)*x
		#else:
			#y = self.height
			#x = 0.0# math.tanh(local_angle+parent_angle)*y
		local_position.append(x)
		local_position.append(y)

		return local_position,local_angle

	def get_angle(self,add_angle=0.0,con=None):
		a_angle = add_angle 
		if con is not None:
			a_angle = self.angle * con.value[0]
		return a_angle

	def create(self,world,TERRAIN_HEIGHT,module=None,node=None,connection_site=None, p_c=None, module_list = None,position = None):
		if p_c is not None and connection_site is None:
			raise("When you want to attach a new component to a parent component, you have to supply",
			"a connection_site object with it. This connection_site object defines where to anchor",
			"the joint in between to components")

		n_height = 0.5
		n_width = 0.5
		angle = 0
		#if p_c is not None:
		#	parent_angle = p_c.angle
		if node is not None:
			if node.module_ is not None:
				n_height = node.module_.height
				n_width = node.module_.width
				#angle = node.module_.get_angle(con = node.parent_connection_coordinates)
			else:
				n_height = module_list[node.type].height
				n_width = module_list[node.type].width
				#angle = module_list[node.type].get_angle(con = node.parent_connection_coordinates)
		elif module is not None:
			n_height = module.height
			n_width = module.width
			angle = module.get_angle(0)
		
		pos = [7,7,0]
		if position is not None:
			pos = position
		if (p_c is not None):
			# The local position should be based on how much the module is rotated from the connection site. 
			# - math.pi to compensate for y directionality of the angle (TODO: should be removed)
			local_pos_x =math.cos(connection_site.orientation.x + angle + math.pi/2) * n_height/2
			local_pos_y =math.sin(connection_site.orientation.x + angle + math.pi/2) * n_height/2
			pos[0] = (local_pos_x) + connection_site.position.x
			pos[1] = (local_pos_y) + connection_site.position.y
		components = []
		joints = []
		# below is a quick hack formula. Should check based on lowest point in space. 
		if (pos[1] - math.sqrt(math.pow(n_width,2) + math.pow(n_height,2)) < TERRAIN_HEIGHT): # TODO CHANGE 5 TO TERRAIN_HEIGHT
			if node is not None:
				node.component = None
			return components,joints
		# if connection_site is not None:
		# We remove -math.pi/2 since we want the y vector to face in the correct direction
		#	angle = connection_site.orientation.x -math.pi/2
		# This module will create one component that will be temporarily stored in ncomponent
		new_component = None
		# This module will create one joint (if a parent component is present) that will be temporarily stored in njoint
		njoint = None

				
		# get parent node
		par = None	

		if connection_site:
			angle += connection_site.orientation.x
				
		fixture = fixtureDef(
				shape=polygonShape(box=(n_width/2, n_height/2)),
				density=1.0,
				friction=0.1,
				restitution=0.0,
				categoryBits=0x0020,
				maskBits=0x001
			)

		new_component = world.CreateDynamicBody(
			position=(pos[0],pos[1]),
			angle = angle ,
			fixtures = fixture)
		color = (125,125,125)
		if node is not None:
			color = world.cmap(node.type/len(module_list))
		#new_component.color1 = (int(color[0]*255),int(color[1]*255),int(color[2]*255))
		#new_component.color2 = (int(color[0]*255),int(color[1]*255),int(color[2]*255))
		new_component.color1 = (color[0],color[1],color[2])
		new_component.color2 = (color[0],color[1],color[2])
		#components.append(new_component)
		components.append(new_component)
		if node is not None:
			node.component = [new_component]
		if connection_site is not None:
			joint = self.create_joint(world, p_c,new_component,connection_site)
			joints.append(joint)

		return components,joints

	def create_joint(self,world, parent_component,new_component,connection_site,actuated =True):
		# First the local coordinates are calculated based on the absolute coordinates and angles of the parent, child and connection site
		
		disA = math.sqrt(math.pow(connection_site.position.x - parent_component.position.x,2)+math.pow(connection_site.position.y - parent_component.position.y,2))
		local_anchor_a = [connection_site.position.x- parent_component.position[0], connection_site.position.y - parent_component.position[1],0]
		local_anchor_a[0] = math.cos(connection_site.orientation.x-parent_component.angle+math.pi/2)*disA;
		local_anchor_a[1] = math.sin(connection_site.orientation.x-parent_component.angle+math.pi/2)*disA;
		
		disB = math.sqrt(math.pow(connection_site.position.x - new_component.position.x,2)+math.pow(connection_site.position.y - new_component.position.y,2))
		local_anchor_b = [new_component.position[0]-connection_site.position.x, new_component.position[1] - connection_site.position.y,0]
		local_anchor_b[0] = math.cos(new_component.angle-connection_site.orientation.x - math.pi/2)*disB;
		local_anchor_b[1] = math.sin(new_component.angle-connection_site.orientation.x - math.pi/2)*disB;
		
		if (actuated == True):
			rjd = revoluteJointDef(
				bodyA=parent_component,
				bodyB=new_component,
				localAnchorA=(local_anchor_a[0],local_anchor_a[1]),
				localAnchorB= (local_anchor_b[0],local_anchor_b[1]),# (connectionSite.a_b_pos.x,connectionSite.a_b_pos.y),# if (math.isclose(connectionSite.orientation.x,0.0)) else (connectionSite.a_b_pos.x,-connectionSite.a_b_pos.y),
				enableMotor=actuated,
				enableLimit=True,
				maxMotorTorque=self.torque,
				motorSpeed = 0.0,	
				lowerAngle = -math.pi/2,
				upperAngle = math.pi/2,
				referenceAngle = connection_site.orientation.x - parent_component.angle #new_component.angle #connectionSite.orientation.x
			)
			joint = world.CreateJoint(rjd)
			return joint;