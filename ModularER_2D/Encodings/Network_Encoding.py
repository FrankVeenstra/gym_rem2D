"""
This encoding is an interface between neural networks and the robot blueprint
It is used to create a 'tree' structure that is interpreted as a robot. 
"""

import numpy as np
import copy
import random

from Encodings import Abstract_Encoding as enc
import Tree as tree_structure
from NeuralNetwork import NEAT_NN
from Encodings import Cellular_Encoding
from enum import Enum

MAX_MODULES = 20

class NETWORK_TYPE(Enum):
	CPPN = 0
	CE = 1

"""
Container Module : This is used to store arbitrary information of the module 
which the L-System uses as a placeholder to create the tree structure
"""
# This module class is a duplicate from the L-System encoding. TODO: change location of this class to 
class C_Module: 
	def __init__(self, index, module, moduleRef):
		# index
		self.index = index
		self.parent = -1
		self.moduleRef = moduleRef
		self.availableConnections = copy.deepcopy(module.available)
		self.children = []
		self.theta = -1
		self.parentConnectionSite = None
		self.handled = False
		self.module = copy.deepcopy(module)
		self.controller = None
		

class NN_enc(enc.Encoding):
	'''
	The neural networks that are being used to create the robot
	directed tree blueprints can both have a genotypic and 
	phenotypic part to them. For the cellular encoding, 
	the genotypic part is mutable, it is the short set of 
	rules that creates a neural network. The phenotype is 
	the network that is actually created from these rules. 
	''' 
	def __init__(self, modulelist, type,config=None):
		self.moduleList = copy.deepcopy(modulelist)			
		self.outputs = []
		# number of inputs # TODO: Not hardcode it like this.
		n_inputs = 3
		n_outputs = 10
		if (config is not None):
			self.maxTreeDepth = int(config['morphology']['max_depth'])
			self.maxModules = int(config['morphology']['max_size'])
		else:
			self.maxTreeDepth = 7
			self.maxModules = 20
		
		for i in range(n_outputs):
			self.outputs.append(0)
		self.inputs = []
		if (type is "CPPN"):
			# neural network genotype
			self.networkType = NETWORK_TYPE.CPPN
			if config is not None:
				self.nn_g = NEAT_NN.CPPN(n_inputs, n_outputs,t_config=config)
			else:
				self.nn_g = NEAT_NN.CPPN(n_inputs, n_outputs)
			# neural network phenotype
		elif(type is "CE"):
			self.networkType = NETWORK_TYPE.CE
			if config is not None:
				self.nn_g = Cellular_Encoding.CE(config = config)
			else:
				self.nn_g = Cellular_Encoding.CE()
			self.nn_g.mutate(0.5,0.5,0.5)
			self.nn_g.create()
		for mod in self.moduleList:
			mod.mutate(0.5,0.5,0.5)
	
	def update(self, index, par_symb, depth):
		# query for every possible connection site
		new_symbols = [] # TODO: change name
		newCon = []
		if (depth > self.maxTreeDepth or index > self.maxModules) :
			return index, new_symbols
		i = -1
		for con in par_symb.availableConnections:
			i+=1
			input = []
			# TODO 
			input.append(float(1)-(float(2)*(float(depth)/float(self.maxTreeDepth)))) # x coordinate is the tree depth normalized to a value between 0 and 1 
			input.append(float(1)-(float(2)*(float(par_symb.moduleRef+1)/float(len(self.moduleList))))) # module type 
			input.append(con.value[0]) # -1.0,0.0,1.0
			#input.append(float(1)-(float(2)*(float(i)/float(len(par_symb.availableConnections))))) #
			#input.append(float(1)-(float(index)/float(self.maxModules)*2))

			output = []
			#for j in range(10-len(input)):
			#	input.append(0)
			if (self.networkType == NETWORK_TYPE.CPPN):
				output = self.nn_p.activate(input)
			elif (self.networkType == NETWORK_TYPE.CE):
				output = self.nn_p.update(input,requested_number_of_outputs=9)
			else:
				raise Exception("Cannot update network, no network type found")
			# print(np.max(output))
			# outputs of the network are 0: module, 1: module type, 2,3: module size
			if output[0] > 0.5:
				newCon.append(con)
				if (output[1] < -1.):
					output[1] = -1.;
				elif (output[1] > 1.):
					output[1] = 1.;
				if (len(self.moduleList)< 1 or len(self.moduleList)> 2000):
					raise("Module list size was ", len(self.moduleList), " which should never occur")
				connectedMNr = int(((output[1]*0.5)+0.5)*float(len(self.moduleList)-1)) # output 0,1,2
				if (connectedMNr >= len(self.moduleList)):
					#raise("Trying to get a reference a module beyond the moduleList given to the network. Make sure the output value is between 0 and len(moduleList)-1")
					connectedMNr = len(self.moduleList)-1
				elif(connectedMNr < 0):
					#raise("Trying to get a reference a module beyond the moduleList given to the network. Make sure the output value is between 0 and len(moduleList)-1")
					connectedMNr = 0
				#print(output[i], connectedMNr)
				connectedModule = C_Module(index, self.moduleList[connectedMNr],connectedMNr)
				connectedModule.module.setMorph(output[2],output[3],output[4])
				#theta = (output[4]*3)-1 
				#connectedModule.theta = theta
				controller = copy.deepcopy(self.moduleList[connectedMNr].controller)
				controller.setControl(output[5],output[6],output[7],output[8], self.moduleList[connectedMNr].angle)
				connectedModule.controller = controller
				#connectedModule.module.controller.setControl(output[5],output[6],output[7],output[8])
				# make sure connection is not available anymore
				connectedModule.parent = par_symb.index
				connectedModule.parentConnectionSite = con
				par_symb.children.append(connectedModule)
				new_symbols.append(connectedModule)
				index+=1
				
		return index, new_symbols


	def mutate(self, MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA,TREE_DEPTH=None):
		if (self.networkType == NETWORK_TYPE.CPPN):
			self.nn_g.mutate()
			self.nn_p = self.nn_g.getPhenotype()
		elif (self.networkType == NETWORK_TYPE.CE):
			self.nn_g.mutate(MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA)
			self.nn_g.create()
		for mod in self.moduleList:
			mod.mutate(MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA)
	
		
		#print("Mutation rate is not implemented yet")
	def iterate(self,current_symbol,index, depth):
		# expand the tree structure based on the network outputs	
		# current module that could contain children
		if (current_symbol.handled == False):
			current_symbol.handled = True
			if (len(current_symbol.children) > 0):
				raise Exception("if symbol was not handled it shouldn't contain children")
			index, symbols = self.update(index,current_symbol, depth)
 		    #self.rules[currentSymbol.moduleRef].update(index) # change name of update 
			for i,s in enumerate(symbols):
				s.parent = current_symbol.index
		else:
			for c in current_symbol.children:
				index = self.iterate(c,index, depth+1)
		#print(string)
		return index

	def create(self, treedepth):
		"""
		creating a tree structure from the neural network is done in a similar manner as the L-System;
		intead of rewriting the tree structure a few times using the rules of the L-System
		the neural network will try to expand the tree structure every rewrite iteration
		"""
		# when using NEAT, a phenotype first needs ot be created out of a genotype. 
		# Since we will only use the phenotype for constructing the robot tree, 
		# we discard the phenotype after we're done
		self.maxTreeDepth = treedepth

		if (self.networkType == NETWORK_TYPE.CE):
			self.nn_g.create()
			self.nn_p = self.nn_g 
		elif (self.networkType == NETWORK_TYPE.CPPN):
			self.nn_p = self.nn_g.getPhenotype()

		# 1: first create the container module dependecy
		axiom = C_Module(0,self.moduleList[0],-1)		
		axiom.controller = copy.deepcopy(self.moduleList[0].controller)
		index = 0
		axiom.children = []
		axiom.index = index
		index+=1
		base = axiom
		for i in range(treedepth): # number of times iterated over the L-System
			index = self.iterate(base, index,0)

		# remove nn_p
		self.nn_p = None
		# 1: create the tree from the container modules
		# transform the string into a usable tree structure
		tree = tree_structure.Tree(self.moduleList)
		self.recursiveNodeGen(-1,base,tree,0)
		# print("number of nodes is : ",len(tree.nodes))
		return tree
		# return super().create()

	# NOTE: The function below is copied from the L-System. Should be defined in abstract class. 
	def recursiveNodeGen(self, parentIndex, m, tree,nodeCounter):
		if nodeCounter > MAX_MODULES:
			return nodeCounter
		controller = m.controller
		node = tree_structure.Node(m.index, parentIndex,m.moduleRef,m.parentConnectionSite,controller)
		node.module_ = m.module
		tree.nodes.append(node)

		for c in m.children:
			nodeCounter+=1
			#print(nodeCounter)
			nodeCounter = self.recursiveNodeGen(c.parent,c,tree, nodeCounter)
		return nodeCounter

