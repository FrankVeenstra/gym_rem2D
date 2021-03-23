import math

import Box2D
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)


def create_joint(world, parent_component,new_component,connection_site,angle,torque, actuated =True):
	# First the local coordinates are calculated based on the absolute coordinates and angles of the parent, child and connection sitedisA = math.sqrt(math.pow(connection_site.position.x - parent_component.position.x,2)+math.pow(connection_site.position.y - parent_component.position.y,2))
	disA = math.sqrt(math.pow(connection_site.position.x - parent_component.position.x,2)+math.pow(connection_site.position.y - parent_component.position.y,2))
	local_anchor_a = [connection_site.position.x- parent_component.position[0], connection_site.position.y - parent_component.position[1],0]
	local_anchor_a[0] = math.cos(connection_site.orientation.x-parent_component.angle+math.pi/2)*disA;
	local_anchor_a[1] = math.sin(connection_site.orientation.x-parent_component.angle+math.pi/2)*disA;
	
	disB = math.sqrt(math.pow(connection_site.position.x - new_component.position.x,2)+math.pow(connection_site.position.y - new_component.position.y,2))
	local_anchor_b = [new_component.position[0]-connection_site.position.x, new_component.position[1] - connection_site.position.y,0]
	local_anchor_b[0] = math.cos(new_component.angle-connection_site.orientation.x - math.pi/2)*disB;
	local_anchor_b[1] = math.sin(new_component.angle-connection_site.orientation.x - math.pi/2)*disB;
	
	rjd = revoluteJointDef(
		bodyA=parent_component,
		bodyB=new_component,
		localAnchorA=(local_anchor_a[0],local_anchor_a[1]),
		localAnchorB= (local_anchor_b[0],local_anchor_b[1]),# (connectionSite.a_b_pos.x,connectionSite.a_b_pos.y),# if (math.isclose(connectionSite.orientation.x,0.0)) else (connectionSite.a_b_pos.x,-connectionSite.a_b_pos.y),
		enableMotor=actuated,
		enableLimit=True,
		maxMotorTorque=torque,
		motorSpeed = 0.0,	
		lowerAngle = -math.pi/2,
		upperAngle = math.pi/2
		#referenceAngle = angle #connection_site.orientation.x - parent_component.angle #new_component.angle #connectionSite.orientation.x
	)
	joint = world.CreateJoint(rjd)
	return joint;