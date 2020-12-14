import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import Tree
import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

import matplotlib.pyplot as plt
import copy

import os

ASSET_PATH = os.path.join(os.path.dirname(__file__), "../../assets")

# Adjusted from Oleg Klimov's Bipedal walker environment. Licensed on the same terms as the rest of OpenAI Gym.
#
# Adjusted by Frank Veenstra and Joergen Jorgensen. Licensed on the same terms as the rest of OpenAI Gym.
#
# Note: this file is still filled with hacks

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 200
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

MAX_PERTURBANCE_TERRAIN = 24
COLOR_CONTROL = True

INITIAL_RANDOM = 5

HULL_POLY =[
	(-30,+9), (+6,+9), (+34,+1),
	(+34,-8), (-30,-8)
	]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE

MODULE_R = 8/SCALE
MODULE_W = 8/SCALE
MODULE_H = 8/SCALE

DISPLAY_JOINTS = False
DISPLAY_VECTORS = False
WOD_SPEED = 0.04

VIEWPORT_W = 800
VIEWPORT_H = 600

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

COLLISIONALLOWED = True # TODO

VERBOSE = False


class ModularRobotBox2D:
	"""
	This instance simply stores all the components created for Box2D
	"""
	def __init__(self):
		self.components = []
		self.joints = []
	def add_components(self, components,joints=None):
		for c in components:
			self.components.append(c)
		if joints is not None:
			for j in joints:
				self.joints.append(j)
		return self


class Vector3:
	"""
	This instance just helped me out wrapping my mind around Box2D and its construction process 
	"""
	def __init__(self,x=None,y=None,z=None):
		self.x = x
		self.y = y
		self.z = z

class ConnectionSite:
	""" 
	The ConnectionSite instance helps to define the coordinates of joints and can be used to 
	easily set the anchor points when connecting objects
	"""
	def __init__(self, position, orientation):
		self.position = position
		self.orientation = orientation
		# Note, could add a parent component reference here


class WallOfDeath:
	def __init__(self, speed):
		# x position
		self.position = 0.0
		self.speed = speed
	def update(self):
		self.position += self.speed
		#self.speed = self.speed + 0.00001
		

class ContactDetector(contactListener):
	def __init__(self, env):
		contactListener.__init__(self)
		self.env = env
	def BeginContact(self, contact):
		if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
			self.env.game_over = True
		for leg in [self.env.legs[1], self.env.legs[3]]:
			if leg in [contact.fixtureA.body, contact.fixtureB.body]:
				leg.ground_contact = True
	def EndContact(self, contact):
		for leg in [self.env.legs[1], self.env.legs[3]]:
			if leg in [contact.fixtureA.body, contact.fixtureB.body]:
				leg.ground_contact = False

class Modular2D(gym.Env, EzPickle):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : FPS
	}

	hardcore = False
	def __init__(self, random_seed = None):
		self.robot_components = []
		self.drawlist = []
		EzPickle.__init__(self)
		self.seed(random_seed)
		self.viewer = None
		self.tree_morphology = None
		self.scroll = 0 # 12-VIEWPORT_W/SCALE/5
		self.scroll_y = 0
		self.robot = None
		self.world = Box2D.b2World()
		self.terrain = None
		self.hull = None
		self.wod = None
		self.prev_shaping = None

		self.fd_polygon = fixtureDef(
						shape = polygonShape(vertices=
						[(0, 0),
						 (1, 0),
						 (1, -1),
						 (0, -1)]),
						friction = FRICTION)

		self.fd_edge = fixtureDef(
					shape = edgeShape(vertices=
					[(0, 0),
					 (1, 1)]),
					friction = FRICTION,
					categoryBits=0x0001,
				)

		
		high = np.array([np.inf] * 24)
		self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _destroy(self):
		if not self.terrain: return
		self.world.contactListener = None
		for t in self.terrain:
			self.world.DestroyBody(t)
		self.terrain = []
		if self.robot is not None:
			for component in self.robot.components:
				self.world.DestroyBody(component)
		world = self.world 
		self.hull = None
		self.joints = []

	def _generate_terrain(self, hardcore):
		GRASS, STUMP, PIT,STAIRS, _STATES_ = range(5)
		state    = GRASS
		useRandomTerrain = False
		#self.seed(8)
		#state = self.np_random.randint(0,3)
		velocity = 0.0
		y        = TERRAIN_HEIGHT
		original_y = TERRAIN_HEIGHT
		counter  = TERRAIN_STARTPAD
		oneshot  = False
		self.terrain   = []
		self.terrain_x = []
		self.terrain_y = []
		for i in range(TERRAIN_LENGTH):
			x = i*TERRAIN_STEP
			self.terrain_x.append(x)
			if (useRandomTerrain):
				if i > 20:
					if self.np_random.uniform(0,1.0) < 0.1:
						state = self.np_random.randint(0,3)
					else:
						state = GRASS
			if state==GRASS and not oneshot:
				velocity = 0.5*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
				if i > TERRAIN_STARTPAD: 
					velocity += self.np_random.uniform(-MAX_PERTURBANCE_TERRAIN/TERRAIN_LENGTH*i, MAX_PERTURBANCE_TERRAIN/TERRAIN_LENGTH*i)/SCALE   #1
				y += velocity

			elif state==PIT and oneshot:
				counter = self.np_random.randint(3, 5)
				poly = [
					(x,              y),
					(x+TERRAIN_STEP, y),
					(x+TERRAIN_STEP, y-4*TERRAIN_STEP),
					(x,              y-4*TERRAIN_STEP),
					]
				self.fd_polygon.shape.vertices=poly
				t = self.world.CreateStaticBody(
					fixtures = self.fd_polygon)
				t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
				self.terrain.append(t)

				self.fd_polygon.shape.vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]
				t = self.world.CreateStaticBody(
					fixtures = self.fd_polygon)
				t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
				self.terrain.append(t)
				counter += 2
				original_y = y

			elif state==PIT and not oneshot:
				y = original_y
				if counter > 1:
					y -= 4*TERRAIN_STEP

			elif state==STUMP and oneshot:
				counter = self.np_random.randint(1, 3)
				poly = [
					(x,                      y),
					(x+counter*TERRAIN_STEP, y),
					(x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
					(x,                      y+counter*TERRAIN_STEP),
					]
				self.fd_polygon.shape.vertices=poly
				t = self.world.CreateStaticBody(
					fixtures = self.fd_polygon)
				t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
				self.terrain.append(t)

			elif state==STAIRS and oneshot:
				stair_height = +1 if self.np_random.rand() > 0.5 else -1
				stair_width = self.np_random.randint(4, 5)
				stair_steps = self.np_random.randint(3, 5)
				original_y = y
				for s in range(stair_steps):
					poly = [
						(x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
						(x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
						(x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
						(x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
						]
					self.fd_polygon.shape.vertices=poly
					t = self.world.CreateStaticBody(
						fixtures = self.fd_polygon)
					t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
					self.terrain.append(t)
				counter = stair_steps*stair_width

			elif state==STAIRS and not oneshot:
				s = stair_steps*stair_width - counter - stair_height
				n = s/stair_width
				y = original_y + (n*stair_height)*TERRAIN_STEP

			oneshot = False
			self.terrain_y.append(y)
			counter -= 1
			if counter==0:
				counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
				if state==GRASS and hardcore:
					state = self.np_random.randint(1, _STATES_)
					oneshot = True
				else:
					state = GRASS
					oneshot = True

		self.terrain_poly = []
		for i in range(TERRAIN_LENGTH-1):
			poly = [
				(self.terrain_x[i],   self.terrain_y[i]),
				(self.terrain_x[i+1], self.terrain_y[i+1])
				]
			self.fd_edge.shape.vertices=poly
			t = self.world.CreateStaticBody(
				fixtures = self.fd_edge)
			color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
			t.color1 = color
			t.color2 = color
			self.terrain.append(t)
			color = (0.4, 0.6, 0.3)
			poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
			self.terrain_poly.append( (poly, color) )
		self.terrain.reverse()

	def _generate_clouds(self):
		# Sorry for the clouds, couldn't resist
		self.cloud_poly   = []
		for i in range(TERRAIN_LENGTH//20):
			x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
			y = VIEWPORT_H/SCALE*3/4
			poly = [
				(x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
				 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
				for a in range(5) ]
			x1 = min( [p[0] for p in poly] )
			x2 = max( [p[0] for p in poly] )
			self.cloud_poly.append( (poly,x1,x2) )

	def get_component_index(self,node,handled_nodes):
		for i in range(len(handled_nodes)):
			if (handled_nodes[i].index == node.parent and handled_nodes[i].expressed):
				return handled_nodes[i], handled_nodes[i].component
		if VERBOSE:
			print("Cannot find a parent node; no worries, it could be that it is just not expressed")
			print("turn off verbose mode to see less messages")
		return None, -1


	def createJetModule(self, n, nodes, p_c):
		angle = p_c.angle
		c_angle = angle + (n.orientation.value[0] * n.module.angle)		
		# get module height and width
		n_width = n.module.width
		n_height = n.module.height
		# get parent node
		par = None
		for parent in nodes:
			if parent.index == n.parent:
				par = parent

		# get parent module height 
		p_h = n.module.height
		if (n.orientation != n.module.connection_type.top):
			p_h = moduleList[par.type].width

		pos = []
		pos.append(math.sin(c_angle) * (n_height + p_h) * 0.25 + components[compIndex].position[0])
		pos.append(math.cos(c_angle) * (n_height + p_h) * 0.25 + components[compIndex].position[1])
		fixture = fixtureDef(
				shape=polygonShape(box=(n_width * MODULE_W, n_height * MODULE_H)),
				density=1.0,
				friction=0.1,
				restitution=0.0,
				categoryBits=0x0020,
				maskBits=0x001
			)
		ncomponent = self.world.CreateDynamicBody(
			position=(pos[0],pos[1]),
			angle = c_angle,
			fixtures = fixture)
		color = cmap(n.type/5)
		ncomponent.color1 = (color[0],color[1],color[2])
		ncomponent.color2 = (color[0],color[1],color[2])
		components.append(ncomponent)
		n.component = ncomponent
		n.type = "JET"
		jointPosition = []
		jointPosition.append(pos[0]-components[compIndex].position[0])
		jointPosition.append(pos[1]-components[compIndex].position[1])
		# TODO joint has no local coordinate system
		rjd = revoluteJointDef(
			bodyA=components[compIndex],
			bodyB=ncomponent,
			localAnchorA=(jointPosition[0]/2, jointPosition[1]/2),
			localAnchorB=(-jointPosition[0]/2, -jointPosition[1]/2),
			enableMotor=True,
			enableLimit=True,
			maxMotorTorque=MOTORS_TORQUE,
			motorSpeed = 0,
			lowerAngle = -math.pi/2,
			upperAngle = math.pi/2,
			referenceAngle = 0
		)
		self.joints.append(self.world.CreateJoint(rjd))

	def create_joint(self,parent_component,new_component,connection_site,actuated =True):
		# First the local coordinates are calculated based on the absolute coordinates and angles of the parent, child and connection site
		
		disA = math.sqrt(math.pow(connection_site.position.x - parent_component.position.x,2)+math.pow(connection_site.position.y - parent_component.position.y,2))
		local_anchor_a = Vector3(connection_site.position.x- parent_component.position.x, connection_site.position.y - parent_component.position.y,0)
		local_anchor_a.x = math.cos(connection_site.orientation.x-parent_component.angle+math.pi/2)*disA;
		local_anchor_a.y = math.sin(connection_site.orientation.x-parent_component.angle+math.pi/2)*disA;
		
		disB = math.sqrt(math.pow(connection_site.position.x - new_component.position.x,2)+math.pow(connection_site.position.y - new_component.position.y,2))
		local_anchor_b = Vector3(new_component.position.x-connection_site.position.x, new_component.position.y - connection_site.position.y,0)
		local_anchor_b.x = math.cos(new_component.angle-connection_site.orientation.x - math.pi/2)*disB;
		local_anchor_b.y = math.sin(new_component.angle-connection_site.orientation.x - math.pi/2)*disB;
		
		if (actuated == True):
			rjd = revoluteJointDef(
				bodyA=parent_component,
				bodyB=new_component,
				localAnchorA=(local_anchor_a.x,local_anchor_a.y),
				localAnchorB= (local_anchor_b.x,local_anchor_b.y),# (connectionSite.a_b_pos.x,connectionSite.a_b_pos.y),# if (math.isclose(connectionSite.orientation.x,0.0)) else (connectionSite.a_b_pos.x,-connectionSite.a_b_pos.y),
				enableMotor=actuated,
				enableLimit=True,
				maxMotorTorque=MOTORS_TORQUE,
				motorSpeed = 0.0,	
				lowerAngle = -math.pi/2,
				upperAngle = math.pi/2,
				referenceAngle = 0.0 #new_component.angle #connectionSite.orientation.x
			)
			joint = self.world.CreateJoint(rjd)
			return joint;
			return rjd
		return

	def create_circle_module(self,module=None,node=None,connection_site=None, p_c=None, module_list=None):
		# get module height and width
		if p_c is not None and connection_site is None:
			raise("When attaching a new component to a parent component, you have to supply",
			"a connection_site object with it. This connection_site object defines where to anchor",
			"the joint in between to components")
		n_radius = 0.5
		angle = 0
		#if p_c is not None:
		#	parent_angle = p_c.angle
		if node is not None:
			if node.module_ is not None:
				n_radius = node.module_.radius
				#angle = node.module.get_angle(con = node.parent_connection_coordinates)
			else:
				n_radius = module_list[node.type].radius
				#angle = module_list[node.type].get_angle(con = node.parent_connection_coordinates)
		elif module is not None:
			n_radius = module.radius
			#angle = module.get_angle(0)

		pos = Vector3(2,7,0);
		if (p_c is not None):
			local_pos_x =math.cos(connection_site.orientation.x+ math.pi/2) * n_radius 
			local_pos_y =math.sin(connection_site.orientation.x+ math.pi/2) * n_radius 
			pos.x = (local_pos_x) + connection_site.position.x
			pos.y = (local_pos_y) + connection_site.position.y
		#if connection_site is not None:
			# We remove -math.pi/2 since we want the y vector to face in the correct direction
		#	angle = connection_site.orientation.x -math.pi/2
		# This module will create one component that will be temporarily stored in ncomponent
		new_component = None
		# This module will create one joint (if a parent component is present) that will be temporarily stored in njoint
		njoint = None
		
		components = []
		joints = []

		if (pos.y - n_radius < TERRAIN_HEIGHT):
			if node is not None:
				node.component = None
			return components,joints
		else:
			fixture = fixtureDef(
					shape=Box2D.b2CircleShape(radius =n_radius),
					density=1.0,
					friction=0.1,
					restitution=0.0,
					categoryBits=0x0020,
					maskBits=0x001
				)
			new_component = self.world.CreateDynamicBody(
				position=(pos.x,pos.y),
				angle = angle,
				fixtures = fixture)

			color = [255,255,255]
			if node is not None and module_list is not None:
				color = self.cmap(node.type/len(module_list))
			elif node is not None and module_list is None:
				print("Note: cannot assign a color to the module since the 'module_list' is not passed as an argument")
			# move to component creator
			new_component.color1 = (color[0],color[1],color[2])
			new_component.color2 = (color[0],color[1],color[2])
			components.append(new_component)
			if node is not None:
				node.component = [new_component]
			#if pos[1] < lowestY:
			#	lowestY = pos[1]
			if connection_site is not None:
				joint = self.create_joint(p_c,new_component,connection_site)
				joints.append(joint)
			
			#ncomponent.angle=connection_site.orientation.x*0.5

		return components, joints
		
	def create_simple_module(self,module=None,node=None,connection_site=None, p_c=None, module_list = None):
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
				angle = node.module_.get_angle(con = node.parent_connection_coordinates)
			else:
				n_height = module_list[node.type].height
				n_width = module_list[node.type].width
				angle = module_list[node.type].get_angle(con = node.parent_connection_coordinates)
		elif module is not None:
			n_height = module.height
			n_width = module.width
			angle = module.get_angle(0)
		
		pos = Vector3(7,7,0)
		if (p_c is not None):
			# The local position should be based on how much the module is rotated from the connection site. 
			# - math.pi to compensate for y directionality of the angle (TODO: should be removed)
			local_pos_x =math.cos(connection_site.orientation.x + angle + math.pi/2) * n_height/2
			local_pos_y =math.sin(connection_site.orientation.x + angle + math.pi/2) * n_height/2
			pos.x = (local_pos_x) + connection_site.position.x
			pos.y = (local_pos_y) + connection_site.position.y
		components = []
		joints = []
		# below is a quick hack formula. Should check based on lowest point in space. 
		if (pos.y - math.sqrt(math.pow(n_width,2) + math.pow(n_height,2)) < TERRAIN_HEIGHT):
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

		new_component = self.world.CreateDynamicBody(
			position=(pos.x,pos.y),
			angle = angle ,
			fixtures = fixture)
		color = (125,125,125)
		if node is not None:
			color = self.cmap(node.type/len(module_list))
		#new_component.color1 = (int(color[0]*255),int(color[1]*255),int(color[2]*255))
		#new_component.color2 = (int(color[0]*255),int(color[1]*255),int(color[2]*255))
		new_component.color1 = (color[0],color[1],color[2])
		new_component.color2 = (color[0],color[1],color[2])
		#components.append(new_component)
		components.append(new_component)
		if node is not None:
			node.component = [new_component]
		if connection_site is not None:
			joint = self.create_joint(p_c,new_component,connection_site)
			joints.append(joint)

		return components,joints

	def create_component(self, module = None, node=None, nodes=None, parent_component=None, module_list=None, connection_site = None):
		"""Should create a component of the robot knowing the parent component
		and the connection coordinates"""
		angle = 0.0
		position = []
		position.append(5)
		position.append(TERRAIN_HEIGHT +2)
		position.append(0)
		if connection_site is not None:
			angle = connection_site.orientation.x

		if parent_component is not None and connection_site is None:
			# A parent component is passed to the next module, a connection site should thereby also be passed
			raise Exception("connection_site needed to attach object to parent")

		if node is None and module is None:
			components,joints = self.create_circle_module(node=node,p_c=parent_component,module_list=module_list,connection_site = connection_site, position = position)
			return(components,joints)
		elif node is None and module is not None:
			components,joints = module.create(self.world,TERRAIN_HEIGHT,module = module, node=node, p_c=parent_component, module_list=module_list, connection_site = connection_site, position = position)
			return components,joints
			#type = module.type
			#if type == "CIRCLE":
			#	components,joints = create_circle_module(module = module)
		if node.module_ is None:
			reference_module = module_list[node.type]
			components,joints = module_list[node.type].create(self.world,TERRAIN_HEIGHT,node=node,p_c = parent_component,module_list = module_list, connection_site=connection_site, position=position)
			node.expressed = True
			return components,joints
			if	reference_module.type == "CIRCLE":
				components, joints = self.create_circle_module(node=node,p_c=parent_component,module_list=module_list,connection_site = connection_site)
				if len(components)>0:
					node.position = Vector3(components[0].position[0],components[0].position[1],0, position = position)
					node.angle = components[0].angle # TODO
					#if connection_site is not None:
						#node.angle += connection_site.orientation.x
					node.expressed = True
				return(components, joints)
			elif reference_module.type == "SIMPLE":
				components, joints = self.create_simple_module(node=node,p_c=parent_component,module_list=module_list,connection_site = connection_site)
				if len(components)>0:
					node.position = Vector3(components[0].position[0],components[0].position[1],0)
					node.angle = components[0].angle # TODO
					node.expressed = True
				return(components, joints)
			else:
				raise("Module type not found")
		else:
			components,joints = node.module_.create(self.world,TERRAIN_HEIGHT,node=node,p_c = parent_component,module_list = module_list, connection_site=connection_site, position = position)
			node.expressed = True
			return components,joints

			if	node.module_.type == "CIRCLE":
				components, joints = self.create_circle_module(node=node,p_c=parent_component,module_list=module_list,connection_site = connection_site)
				if len(components)>0:
					node.position = Vector3(components[0].position[0],components[0].position[1],0)
					node.angle = components[0].angle # TODO
					#if connection_site is not None:
						#node.angle += connection_site.orientation.x
					node.expressed = True
				return(components, joints)
			elif node.module_.type == "SIMPLE":
				components, joints = self.create_simple_module(node=node,p_c=parent_component,module_list=module_list,connection_site = connection_site)
				if len(components)>0:
					node.position = Vector3(components[0].position[0],components[0].position[1],0)
					node.angle = components[0].angle # TODO
					node.expressed = True
				return(components, joints)
			else:
				raise("Module type not found")
		if node.module_.type == "CIRCLE":
			return(self.create_circle_module(node,nodes, parent_component,module_list))
		elif node.module_.type == "SIMPLE":
			return(self.create_simple_module(node,nodes, parent_component,module_list))
		elif node.module_.type == "EMERGE":
			return(self.createEmergeModule(node,nodes, parent_component,module_list))

	def get_connection_site(self, parent,node,module_list):
		con = node.parent_connection_coordinates
		# relative to parent
		if parent.module_ is not None:
			local_position,local_angle = parent.module_.get_local_position_of_connection_site(con,parent.angle)
			global_position = Vector3(local_position[0]+parent.position.x,local_position[1]+parent.position.y,0)
			global_angle = Vector3(local_angle+parent.angle,0,0)
			return ConnectionSite(global_position,global_angle)
		else:
			module_reference = module_list[node.type]
			local_position,local_angle = module_reference.get_local_position_of_connection_site(con,parent.angle)
			global_position = Vector3(local_position[0]+parent.position.x,local_position[1]+parent.position.y,0)
			global_angle = Vector3(local_angle+parent.angle,0,0)
			return ConnectionSite(global_position,global_angle)

	def create_robot(self,nodes,module_list):
		# Instance to store robot components created from nodes
		self.robot = ModularRobotBox2D()
		handled_nodes = []
		for node in nodes:
			if node.parent == -1:
				# nodes with no parents are created first
				new_components, new_joints = self.create_component(node=node,nodes=nodes,module_list=module_list)
				self.robot.add_components(new_components, new_joints)
				handled_nodes.append(node)
		for node in nodes:
			# id the index of the node is not in the handled_nodes list, 
			if not node.expressed:
				parent, compIndex = self.get_component_index(node,handled_nodes)
				if parent is not None and parent.component is not None and parent.expressed:
					# if the parent is expressed its component should not be None, this is accepted here
					# The connection site is created based on the parent connection of the node
					cspos = []
					csor = 0
					if parent.module_ is not None:
						cspos, csor = parent.module_.get_global_position_of_connection_site( parent_component = parent.component[0], con=node.parent_connection_coordinates)
					position= Vector3(cspos[0],cspos[1],0)
					orientation= Vector3(csor,0,0)
					connection_site = ConnectionSite(position, orientation)
						#connection_site = self.get_connection_site(parent, node, module_list)
					#connection_site = parent.get_connection_site(node, module_list)
					"""
					module2 = copy.deepcopy(module)
					mod2,joints2 =  self.create_component(module = module2,parent_component = mod[0],connection_site = connection_site);
					pos2,orientation2 = module2.get_global_position_of_connection_site(parent_component=mod2[0])
					connection_site2 = ConnectionSite(Vector3(pos2[0],pos2[1],0),Vector3(orientation2,0,0))
					module3 = copy.deepcopy(module)
					mod3,joints3 =  self.create_component(module = module3,parent_component = mod2[0],connection_site = connection_site2)
					"""
					new_components, new_joints = self.create_component(node=node, nodes=nodes, parent_component=parent.component[0], module_list = module_list,connection_site = connection_site)
					self.robot.add_components(new_components, new_joints)
					handled_nodes.append(node)
					#for com in newComponents:
					#	components.append(com)
					#for j in newJoints:
					#	self.joints.append(j)
					#if len(newComponents)> 0:
					#	n.expressed= True
				else:
					# parent found but not expressed
					pass
		return
	
	def reset(self, tree=None, module_list=None):
		self.wod = WallOfDeath(WOD_SPEED)
		self.wod.position = 0.0
		self.tree_morphology = copy.deepcopy(tree)

		self._destroy()
		# Creating the world since not all objects were properly deleted. TODO need to debug this
		self.world = Box2D.b2World()
		self.world.cmap = plt.get_cmap('viridis')
		self.game_over = False
		self.scroll = 0.0
		self.scroll_y = 0.0
		# smooth scroll 
		self.prevscroll = 0.0
		self.prevscroll_y = 0.0
		self.scroll_y
		W = VIEWPORT_W/SCALE
		H = VIEWPORT_H/SCALE
		self._generate_terrain(self.hardcore)
		self._generate_clouds()

		init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
		init_y = TERRAIN_HEIGHT+2*LEG_H
		# will check if the robot position needs to be shifted upward
		lowestY = init_y
		# Fetch the nodes from the tree phenotype
		if (self.tree_morphology is not None):
			nodes = self.tree_morphology.getNodes() 
			# create a robot from these nodes
			self.create_robot(nodes,module_list)
			self.drawlist = self.terrain + self.robot.components + self.robot.joints
		else:
			self.drawlist = self.terrain
		return

	def PID(self,desiredAngle,joint):
		proportional = 1.9
		currentAngle = joint.angle
		angleDifference = desiredAngle-currentAngle
		speed = angleDifference * proportional
		return speed
	
	def step(self, action):
		observation = 0
		reward = 0
		done = 0
		info = 0

		if self.wod:
			self.wod.update()
		# move jet code 
		# temp
		c_values = []
		particles = []
		if self.tree_morphology is not None:
			for n in self.tree_morphology.nodes:
				if n.controller != None:
					c_values.append(n.controller.update(0))
					if (COLOR_CONTROL):
						cmap = plt.get_cmap('viridis')
						color = cmap(c_values[-1])
						if n.expressed and n.component is not None:
							n.component[0].color2 = (color[0],color[1],color[2])
			for i,j in enumerate(self.robot.joints):
				j.motorSpeed = self.PID(c_values[i+1],j)

		self.world.Step(1.0/FPS, 6*30, 2*30)
		if self.tree_morphology is not None:
			x_scroll = self.robot.components[0].position[0] - VIEWPORT_W/SCALE/5
			y_scroll = self.robot.components[0].position[1] - VIEWPORT_H/SCALE/4
			self.scroll = x_scroll+0.99*(x_scroll-self.prevscroll)
			self.scroll_y = y_scroll+0.99*(y_scroll-self.prevscroll_y)
			self.prevscroll = x_scroll
			self.prevscroll_y = y_scroll
			reward = self.robot.components[0].position[0]
			if self.game_over or self.robot.components[0].position[0] < 0:
				reward = -100
				done   = True
			if self.wod:
				if self.wod.position > self.robot.components[0].position[0]:
					reward = -100
					done   = True
		else:
			raise Exception("no tree_morphology")

		return observation, reward, done, info

	def render(self, mode='human'):
		from gym.envs.classic_control import rendering
		if self.viewer is None:
			self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
		self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, +self.scroll_y, VIEWPORT_H/SCALE + self.scroll_y)

		self.viewer.draw_polygon( [
			(self.scroll,                  self.scroll_y),
			(self.scroll+VIEWPORT_W/SCALE, self.scroll_y),
			(self.scroll+VIEWPORT_W/SCALE, self.scroll_y+VIEWPORT_H/SCALE),
			(self.scroll,                  self.scroll_y+VIEWPORT_H/SCALE),
			], color=(0.9, 0.9, 1.0) )
		for poly,x1,x2 in self.cloud_poly:
			if x2 < self.scroll/2: continue
			if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
			self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]+self.scroll_y) for p in poly], color=(1,1,1))
		for poly, color in self.terrain_poly:
			if poly[1][0] < self.scroll : continue
			if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
			self.viewer.draw_polygon(poly, color=color)
		for obj in self.drawlist:
			if (obj.type!=1):
				for f in obj.fixtures:
					trans = f.body.transform
					if type(f.shape) is circleShape:
						t = rendering.Transform(translation=trans*f.shape.pos)
						self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
						self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
						if DISPLAY_VECTORS:
							ll = 0.5 # linelength
							x1 = (math.cos(f.body.angle) * ll) +f.body.position[0]
							y1 = (math.sin(f.body.angle) * ll) +f.body.position[1]
							tpy = [x1,y1]
							x2 = (math.cos(f.body.angle + math.pi/2) * ll) +f.body.position[0]
							y2 = (math.sin(f.body.angle + math.pi/2) * ll) +f.body.position[1]
							tpx = [x2,y2]
							self.viewer.draw_line(t.translation, tpx,color=(0,0,255))
							self.viewer.draw_line(t.translation, tpy,color=(255,0,0))

					else:
						path = [trans*v for v in f.shape.vertices]
						self.viewer.draw_polygon(path, color=obj.color1)
						path.append(path[0])
						self.viewer.draw_polyline(path, color=obj.color2, filled=False, linewidth=2)
						if DISPLAY_VECTORS:
							t = rendering.Transform(translation=trans*f.body.position)
							ll = 0.5 # linelength
							x1 = (math.cos(f.body.angle) * ll) +f.body.position[0]
							y1 = (math.sin(f.body.angle) * ll) +f.body.position[1]
							tpy = [x1,y1]
							x2 = (math.cos(f.body.angle + math.pi/2) * ll) +f.body.position[0]
							y2 = (math.sin(f.body.angle + math.pi/2) * ll) +f.body.position[1]
							tpx = [x2,y2]
							self.viewer.draw_line(f.body.position, tpx,color=(0,0,255))
							self.viewer.draw_line(f.body.position, tpy,color=(255,0,0))
			else:
				if DISPLAY_JOINTS:
					t = rendering.Transform(translation=obj.anchorA)
					self.viewer.draw_circle(0.1, 30).add_attr(t)
					t = rendering.Transform(translation=obj.anchorB)
					self.viewer.draw_circle(0.1, 30).add_attr(t)
					t = rendering.Transform(translation=obj.anchorA)
					self.viewer.draw_circle(0.1, 30).add_attr(t)
					t = rendering.Transform(translation=obj.anchorB)
					self.viewer.draw_circle(0.1, 30, color = (0,0,250)).add_attr(t)
		
		if self.wod:
			wodposdown = []
			wodposup = []
			wodposdown.append(self.wod.position)
			wodposdown.append(-10)
			wodposup.append(self.wod.position)
			wodposup.append(40)
			self.viewer.draw_line(wodposup,wodposdown,color=(0,0,1))


		flagy1 = TERRAIN_HEIGHT
		flagy2 = flagy1 + 50/SCALE
		x = TERRAIN_STEP*3
		self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
		f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
		self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
		self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )
		return self.viewer.render(return_rgb_array = mode=='rgb_array')

		# ------------------------------
		self.lidar_render = (self.lidar_render+1) % 100
		i = self.lidar_render
		if i < 2*len(self.lidar):
			l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
			self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

		for obj in self.drawlist:
			for f in obj.fixtures:
				trans = f.body.transform
				if type(f.shape) is circleShape:
					t = rendering.Transform(translation=trans*f.shape.pos)
					self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
					self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
				else:
					path = [trans*v for v in f.shape.vertices]
					self.viewer.draw_polygon(path, color=obj.color1)
					path.append(path[0])
					self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

		flagy1 = TERRAIN_HEIGHT
		flagy2 = flagy1 + 50/SCALE
		x = TERRAIN_STEP*3
		self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
		f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
		self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
		self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None

