import random
#from gym_rem import morph


import copy
import numpy as np

# import matplotlib.pyplot as plt
# printing tree structures and activation levels -> should go to phenotype like file
# TODO global variable for collision environment
from Encodings import Abstract_Encoding as enc
import Tree as tree_structure
#from Modules import SimpleModule as sm

# two options, overwrite previous or append previous
context_sensitive = False # double check if this is the good name for it



"""
Container Module : This is used to store arbitrary information of the module 
which the L-System uses as a placeholder to create the tree structure
"""
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

class Rule:
	"""
	This Rule object is similar to rewriting rules in L-Systems
	Classically, in an L-System a rule determines how a symbol is rewritten
	e.g.: A := BCB
	Here, the L-System is perhaps a bit unconventional since the left hand side
	symbol is never rewritten. In this case, the L-System generates
	e.g.: A := A[BCB]
	BCB would represent three child modules and A the parent module

	"""
	# moduleref = index
	def __init__(self, moduleRef, moduleList):
		# transforms module object to container module
		self.module = C_Module(-1, moduleList[moduleRef], moduleRef)
		# max_children defines the maximum number of products it can be written to
		self.max_children = len(self.module.availableConnections)
		# placeholder for how many modules should be attached to the current module
		self.n_children = random.randint(0,self.max_children)
		# store reference to module list for easy mutation
		self.moduleList = moduleList 
		# reference to the module in module List 
		self.moduleRef = moduleRef
		
		# populate children with container objects 
		for i in range(self.n_children):
			# get a random connection from the container module
			con = random.choice(self.module.availableConnections)
			productModuleIndex = random.choice(range(len(moduleList)))
			connectedModule = C_Module(-1,moduleList[productModuleIndex],productModuleIndex)
			# set orientation # TODO: change orientation based on module specifications
			theta = random.randint(0,3) 
			connectedModule.theta = theta			
			# make sure connection is not available anymore
			self.module.availableConnections.remove(con)
			connectedModule.parentConnectionSite = con
			self.module.children.append(connectedModule)
		#for mod in moduleList:	
		#	mod.mutate(0.5,0.5,0.5)

	def mutate(self, MORPH_MUTATIONRATE,MUTATION_RATE,MUT_SIGMA):
		self.moduleList[self.moduleRef].mutate(MORPH_MUTATIONRATE,MUTATION_RATE,MUT_SIGMA)
		if (random.uniform(0.0,1.0) < MORPH_MUTATIONRATE):
			# add child to rule
			if (self.n_children < self.max_children-1):
				self.n_children+=1
				con = random.choice(self.module.availableConnections)
				productModuleIndex = random.choice(range(len(self.moduleList)))
				connectedModule = C_Module(-1,self.moduleList[productModuleIndex],productModuleIndex)
				# TODO change: theta is not used
				theta = random.randint(0,3)
				connectedModule.theta = theta
				self.module.availableConnections.remove(con)
				connectedModule.parentConnectionSite = con
				self.module.children.append(connectedModule)

		# delete child from rule
		if (random.uniform(0.0,1.0) < MORPH_MUTATIONRATE):
			if (self.n_children > 0):
				self.n_children -=1
				con = random.choice(self.module.children)
				self.module.availableConnections.append(con.parentConnectionSite)
				self.module.children.remove(con)

	def init(self):
		# 1 create list of child positions based on number of child modules
		# 2 assign random orientations to the list
		# 3 assign a random type 
		siteList = []
		while len(siteList) < n_children:
			newPos = random.randint(0,n_children-1)
			# checking if position is not taken
			present = False
			for s in siteList:
				if (newPos == s):
					present =True
			if present == False:
				siteList.append(newPos)

		for nm in siteList:
			tp = random.randint(0, 1)
			pr = 0 # not used
			o = random.randint(0,3)
			self.product.append(Module(tp,pr,nm,o));
	def update(self, index):
		output = []
		# loop through children stored in the rule
		for c in self.module.children:
			index+=1
			newC = copy.deepcopy(c)
			newC.children = []
			newC.index = index
			newC.handled = False
			output.append(newC)
		return index, output

class LSystem(enc.Encoding):
	"""
	"""
	def __init__(self, moduleList, config=None):
		# moduleTypes is a list that contains module objects. The first module is the axiom
		self.moduleList = moduleList
		# every module type gets its own rule (or scheme)
		self.rules = []
		if (config is not None):
			self.treeDepth = int(config['morphology']['max_depth'])
			self.maxModules = int(config['morphology']['max_size'])
		else:
			self.treeDepth = 8
			self.maxModules = 20

		for i,m in enumerate(moduleList):
			# passes reference of module object to the rules. (to determine number of childs etc.)
			self.rules.append(Rule(i,moduleList))
	def recursiveNodeGen(self, parentIndex, m, tree,nodeCounter):
		if nodeCounter > self.maxModules:
			return nodeCounter
		controller = copy.deepcopy(self.moduleList[m.moduleRef].controller)
		nnode = tree_structure.Node(m.index, parentIndex,m.moduleRef,m.parentConnectionSite,controller)
		nnode.module_ = copy.deepcopy(self.moduleList[m.moduleRef])
		tree.nodes.append(nnode)

		for c in m.children:
			nodeCounter+=1
			#print(nodeCounter)
			nodeCounter = self.recursiveNodeGen(c.parent,c,tree, nodeCounter)
		return nodeCounter
	def create(self, treedepth):
		# for now, axiom == rule one container
		axiom = copy.deepcopy(self.rules[0].module)
		index = 0
		axiom.children = []
		axiom.index = index
		base = axiom
		# this is a parameterized L-System and the term string refers to a classical L-System, 
		# it's not a string, but a list of container objects
		for i in range(self.treeDepth): # number of times iterated over the L-System
			index = self.iterate(base, index,0)  
		#print(axiom)

		# create tree
		# transform the string into a usable tree structure
		tree = tree_structure.Tree(self.moduleList)
		self.recursiveNodeGen(-1,base,tree,0)
		#print("number of nodes is : ",len(tree.nodes))
		return tree
		#return super().create()

	def mutate(self, MORPH_MUTATIONRATE,MUTATION_RATE,MUT_SIGMA):
		# mutate modules
		for m in self.moduleList:
			m.mutate(MORPH_MUTATIONRATE,MUTATION_RATE, MUT_SIGMA)
		# mutate rules
		for r in self.rules:
			r.mutate(MORPH_MUTATIONRATE,MUTATION_RATE,MUT_SIGMA)
	def iterate(self,currentSymbol,index, depth):
		# depth is for debugging
		depth+=1
		if index > self.maxModules:
			return index
		if (currentSymbol.handled == False):
			currentSymbol.handled = True
			if (len(currentSymbol.children) > 0):
				raise Exception("if symbol was not handled it shouldn't contain children")
			#print(index, depth)
			index, symbols = self.rules[currentSymbol.moduleRef].update(index) # change name of update 
			for s in symbols:
				s.parent = currentSymbol.index
				currentSymbol.children.append(s)
		else:
			for c in currentSymbol.children:
				index = self.iterate(c,index, depth)
		#print(string)
		return index

	def init(self, nr):
		for i in range(nr):
			m = 0
			if i == 0:
				m = self.axiom
			else:
				m = 1
			self.rules.append(Rule(i,m))
			self.rules[i].init()
	
	# not used no TODO
	def getProduct(self, symbol):
		return self.rules[symbol].update()
		

