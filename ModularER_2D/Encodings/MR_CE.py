from Encodings import Cellular_Encoding as ce

import numpy as np
import copy
import random

from Encodings import encoding as enc
import Tree as tree_structure


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
		

class CE_enc(enc.Encoding):
	def __init__(self, modulelist):
		self.moduleList = modulelist			
		self.outputs = []
		n_inputs = 10 
		n_outputs = 10
		for i in range(n_outputs):
			self.outputs.append(0)
		self.inputs = []
		self.ce = ce.CE()
		self.ce.create()
	def display(self,ax):
		self.ce.display(ax)

	def update(self, index, par_symb):
		# TODO: set input based on parent symbol
		#for i in range(len(self.outputs)):
		#	self.outputs[i] = np.random.uniform(0.0,1.0)
		#n_newModules = int(self.outputs[0]*3)
		new_symbols = [] # TODO: change name
		input = []
		input.append(par_symb.theta/3)
		input.append(par_symb.moduleRef / 5)
		for i in range(8):
			input.append(1)
			#input.append(random.uniform(-1,1.0))
		#input.append(par_symb.parentConnectionSite)
		output = self.ce.update(input)
		
		#n_newModules = int(output[0]*3) # TODO change to max connections
		ac = par_symb.availableConnections
		newCon = []
		# TODO: What to do when there are not enough outputs
		l_out = len(output)
		if (l_out > 3 and output[3] > 0.1):
			con=ac[0]
			newCon.append(con)
		if (l_out > 4 and output[4] > 0.1):
			con = ac[1]
			newCon.append(con)
		if (l_out > 5 and output[5] > 0.1):
			con = ac[2]
			newCon.append(con)
		for i in range(len(newCon)):
			index+=1
			connectedMNr = int(((output[i]*0.5)+0.5)*float(len(self.moduleList))) # output 0,1,2
			if (connectedMNr >= len(self.moduleList)):
				connectedMNr = len(self.moduleList)-1
			elif(connectedMNr < 0):
				connectedMNr = 0
			#print(output[i], connectedMNr)
			connectedModule = C_Module(index, self.moduleList[connectedMNr],connectedMNr)
			# make sure connection is not available anymore
			connectedModule.parentConnectionSite = newCon[i]
			par_symb.children.append(connectedModule)
			new_symbols.append(connectedModule)
		
		return index, new_symbols

	def mutate(self, MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA):
		self.ce.mutate(MORPH_MUTATION_RATE,MUT_SIGMA)
		self.ce.create()
	def iterate(self,currentSymbol,index, depth):
		# expand the tree structure based on the network outputs			
		# current module that could contain children
		if (currentSymbol.handled == False):
			currentSymbol.handled = True
			if (len(currentSymbol.children) > 0):
				raise Exception("If symbol was not handled it shouldn't contain children")
			#print(index, depth)
			index, symbols = self.update(index,currentSymbol)
 		    #self.rules[currentSymbol.moduleRef].update(index) # change name of update 
			for i,s in enumerate(symbols):
				s.parent = currentSymbol.index
		else:
			for c in currentSymbol.children:
				index = self.iterate(c,index, depth)
		#print(string)
		return index

	def create(self, treedepth):
		"""
		creating a tree structure from the neural network is done in a similar manner as the L-System;
		intead of rewriting the tree structure a few times using the rules of the L-System
		the neural network will try to expand the tree structure every rewrite iteration
		"""
		# 1: first create the container module dependecy
		axiom = C_Module(0,self.moduleList[0],0)		
		index = 0
		axiom.children = []
		axiom.index = index
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

	# NOTE: The function below is copied from the L-System. Should be defined in abstract class. 
	def recursiveNodeGen(self, parentIndex, m, tree,nodeCounter):
		MAX_MODULES = 20
		if nodeCounter > MAX_MODULES:
			return nodeCounter
		controller = copy.deepcopy(self.moduleList[m.moduleRef].controller)

		tree.nodes.append(tree_structure.Node(m.index, parentIndex,m.moduleRef,m.parentConnectionSite,controller))

		for c in m.children:
			nodeCounter+=1
			#print(nodeCounter)
			nodeCounter = self.recursiveNodeGen(c.parent,c,tree, nodeCounter)
		return nodeCounter


