from Encodings import Abstract_Encoding as enc
import Tree
from Controller import m_controller
import copy
import random

class DirectTree(Tree.Tree):
	def __init__(self, module_list):
		super().__init__(module_list)
		# The tree starts of with the first module in the module list
		parent = -1
		type = 0
		control = m_controller.Controller() 
		self.index = 0 # keeps track of unique ID of nodes
		self.tree_nodes = []
		self.tree_nodes.append(DirectNode(self.index,parent,type, None,control,copy.deepcopy(module_list[0])))
	
	def nodeGenerator(self,node,nodelist):
		nodelist.append(node)
		for ch in node.children:
			self.nodeGenerator(ch,nodelist)
		return nodelist
	def getNodes(self):
		nodelist = []
		self.nodeGenerator(self.tree_nodes[0],nodelist)
		self.nodes = nodelist
		return super().getNodes()

class DirectNode(Tree.Node):
	def __init__(self, index,parent,type,orientation,control, module_):
		super().__init__(index,parent,type,orientation,control,module_ = copy.deepcopy(module_))
		self.availableConnections = self.module_.available
		self.children = []
	def mutate(MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA):
		self.module_.mutate(MORPH_MUTATION_RATE)

	def addChild(self,module, index, parent,moduleRef, moduleController,parentConnectionSite):
		index+=1
		self.children.append(DirectNode(index,parent,moduleRef,parentConnectionSite,moduleController,module))
		self.availableConnections.remove(parentConnectionSite)
		return index

class DirectEncoding(enc.Encoding):
	def __init__(self,moduleList,config=None):
		# This list contains the modules to choose from. Needs to be deep-copied when used
		super().__init__()
		self.moduleList = moduleList
		# Direct tree contains a little bit more information to ease mutations
		self.tree = DirectTree(moduleList)

		self.n_modules = 1
		if config is not None:
			self.maxDepth = int(config['morphology']['max_depth'])
			self.maxModules = int(config['morphology']['max_size'])
		else:
			self.maxDepth = 8
			self.maxModules = 20
		
		for i in range(5):
			self.mutate(0.5,0.5,0.5)
	def create(self, treedepth):
		"""
		This function is used by other encodings to create a tree structure out of the 
		genes. Since the direct encoding alters the tree structure of the modular robot
		directly this structure is simply returned in the create function.
		"""
		# Note, treedepth can be used to prune the tree structure if it is too long
		# flush tree controllers
		for node in self.tree.nodes:
			node.controller.i_state = 0
		return self.tree
	def countModules(self):
		self.n_modules = self.countModulesRec(self.tree.tree_nodes[0],0)
	def countModulesRec(self, node, count):
		count+=1
		for mod in node.children:
			count = self.countModulesRec(mod,count)
		return count 

	# recursive function
	def mutateNode(self, node, morphMutationRate,mutationRate,sigma, depth):
		self.countModules()
		# iterate through all connected modules
		for mod in node.children:
			if (random.uniform(0,1) < float(morphMutationRate)/float(2)/float(self.n_modules)): # TODO add to configuration parser
				# cannot remove the base of the tree. This would delete the entire robot
				if (depth != 0):
					# free up the parent connection site
					site = mod.parent_connection_coordinates
					node.availableConnections.append(site)
					# Note, the connection site can now be selected again from the parent
					# Remove the module (and children)
					node.children.remove(mod)
					self.countModules()
			else:
				# Cannot expand tree beyond tree depth limit
				d = depth+1 # TODO
				self.mutateNode(mod, morphMutationRate,mutationRate,sigma,d)
		for con in node.availableConnections:
			self.countModules()
			if (self.n_modules < self.maxModules and depth < self.maxDepth and random.uniform(0,1)< morphMutationRate/float(self.n_modules)):
				# add module at connection site
				type = random.randint(0,len(self.moduleList)-1)
				newModule = self.moduleList[type]
				# type = newModule.
				# add a random controller
				moduleController =  m_controller.Controller()
												#module, index,parent,moduleRef, moduleController,parentConnectionSite
				self.tree.index = node.addChild(copy.deepcopy(newModule),self.tree.index,node.index,type, moduleController,con)
		node.module_.mutate(morphMutationRate,mutationRate,sigma)
		node.controller.mutate(mutationRate,sigma, node.module_.angle)
	def reassignIndices(self):
		index = 0
		self.index = self.reassignIndicesRec(self.tree.tree_nodes[0], index)

	def reassignIndicesRec(self, node, index):
		node.index = index
		index+=1
		for ch in node.children:
			index = self.reassignIndicesRec(ch,index)
			ch.parent = node.index
		return index

	def mutate(self, morphMutationRate, mutationRate,sigma):
		# The mutation function is a recursive function which calls all the nodes in the tree 
		self.mutateNode(self.tree.tree_nodes[0], morphMutationRate,mutationRate,sigma,0)
		self.countModules()
		self.reassignIndices()
	def crossover(self, other):
		# one point crossover
		point1 = random.randint(0,self.n_modules-1)
		point2 = random.randint(0,other.n_modules-1)
		# points refer to indices
		for i,node in enumerate(self.tree.tree_nodes):
			if node.index == point1:
				for target_node in other.tree.tree_nodes:
					if target_node.index == point2:
						self.tree.tree_nodes[nodes] = copy.deepcopy(target_node)
		self.reassignIndices()