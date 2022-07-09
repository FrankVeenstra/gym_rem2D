from copy import deepcopy
import matplotlib.pyplot as plt
import copy
'''
Tree blueprint
'''
class Tree:
	def __init__(self, moduleList):
		self.nodes = []
		self.moduleList = moduleList
	def getNodes(self):
		return self.nodes
	def prune_tree(self,tree,expressed_nodes=None):
		"""
		Function maintains original tree by making a deepcopy and pruning it. This returns a pruned version of the original tree.
		"""
		if not expressed_nodes:
			print("expressed nodes required")
			return
		pruned_tree = copy.deepcopy(tree)
		for idx in expressed_nodes:
			if not expressed_nodes.get(idx):
				node = next((x for x in pruned_tree if x.index == idx), None)
				if node is None:
					print("Tried to remove node, but failed.")
					return
				pruned_tree.remove(node)
		return pruned_tree
	def get_leaves(self,expressed_nodes=None):
		if not expressed_nodes:
			print("expressed nodes required")
			return
		leaves = 0
		nodes = self.getNodes()
		pruned_tree = self.prune_tree(nodes,expressed_nodes)
		for node in pruned_tree:
			if node.parent != -1 and node.parent != None:
				parent = next((x for x in pruned_tree if node.parent == x.index), None)
				if parent is None:
					print("parent does not exist")
					return
				parent.has_children = True
		for node in pruned_tree:
			if not node.has_children: # Need to make sure that the tree is simulated before this, otherwise nodes are not expressed.
				leaves += 1
		return leaves
	def get_nodes(self):
		return self.nodes

class Node:
	def __init__(self, index, parent, type, parent_connection_coordinates, controller=None,component=None, module_ = None):
		self.index = index
		self.type = type
		self.parent = parent
		self.has_children = False
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

