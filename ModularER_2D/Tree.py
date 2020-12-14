import matplotlib.pyplot as plt
'''
Tree blueprint
'''
class Tree:
	def __init__(self, moduleList, controller = None):
		self.nodes = []
		self.moduleList = moduleList
	def getNodes(self):
		return self.nodes

class Node:
	def __init__(self, index, parent, type, parent_connection_coordinates, controller=None,component=None, module_ = None):
		self.index = index
		self.type = type
		self.parent = parent
		self.parent_connection_coordinates = parent_connection_coordinates
		# A controller can be attached for decentralized control of the robot
		self.controller = controller
		self.expressed = False
		# Component can be used to attach an object for reference
		self.component = component
		self.module_ = module_

	def __bool__(self):
		return self.expressed


# Could be used later when using weighted edges
class Edge:
	def __init__(self,parent, target):
		self.parent = parent
		self.target = target

