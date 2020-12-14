import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from enum import Enum
from NeuralNetwork import activations as act
#import pygame

ACTIVATION_FUNCTIONS_SET = act.ActivationFunctionSet()
#TODO move variables below, hard coded for now
n_iterations = 10;
c_iteration = 0;
n_schemes = 10;
useMaxCells = True;
maxCells = 50;
width = 1000
height = 1000

# for displaying
dis = 100
displayNodes = True
displayConnections = True
displayConnectionWeights = True
displayActivity = True
leakiness = 0.8

class NETWORK_UPDATE(Enum):
	FLUSH = 0,
	LEAKY = 1

netUpdate = NETWORK_UPDATE.FLUSH
activationFunctions = []
activationFunctions.append('sigmoid')
activationFunctions.append('tanh')
activationFunctions.append('sin')
activationFunctions.append('gauss')
activationFunctions.append('relu')
activationFunctions.append('softplus')
activationFunctions.append('identity')
activationFunctions.append('clamped')
activationFunctions.append('inv')
activationFunctions.append('log')
activationFunctions.append('exp')
activationFunctions.append('abs')
activationFunctions.append('hat')
activationFunctions.append('square')
activationFunctions.append('cube')

class Scheme():
	def __init__(self,n_schemes):
		self.p_symbols = []; # product symbols
		n_products = random.randint(1,2)
		self.weights = []
		self.thresholds = [] 
		self.activationFunctions = []
		for i in range(n_products):
			newType = random.randint(0,n_schemes-1)
			self.p_symbols.append(newType)
			self.weights.append(random.uniform(-1,1))
			self.thresholds.append(random.uniform(0.5,1))
			ch = random.choice(activationFunctions)
			self.activationFunctions.append(ACTIVATION_FUNCTIONS_SET.get(ch))

		self.type = random.randint(0,1)

class Cell:
	def __init__(self,type, pos, threshold, index,activationFunction):
		self.type = type;
		# a vector2 positional coordinate is included for visualization purposes.
		self.pos = pos
		# input pointers should be of Cell class
		# note: the input pointers are used to ease copying cells. 
		self.input_indices=[]
		self.index = index
		self.activationFunction = activationFunction
		# output pointers should be links
		self.output_links = []
		self.activity = 0.0
		self.layer = 0
		self.threshold = threshold
	def update(self):
		# limit self activity
		if (self.activity > 1.0):
			self.activity = 1.0
		elif(self.activity < -1.0):
			self.activity = -1.0;
		accumulatedOutput = 0.0
		for out in self.output_links:
			output = 0.0
			#if (self.activity > self.threshold):
			#output = np.sin(self.activity*out.weight)
			output = self.activationFunction(self.activity*out.weight)
			#if (self.type > 10):
			#	output = np.sin(self.activity*out.weight)
			#if (self.type > 15):
			#	output = -(self.activity*out.weight)
			accumulatedOutput += output
		return accumulatedOutput

class Link:
	def __init__(self, weight, t_index):
		self.weight = weight
		self.c_index = t_index

class CE:
	def __init__(self, config = None):
		self.schemes = []
		for i in range(n_schemes):
			self.schemes.append(Scheme(n_schemes))
		self.init()
		self.index = 0
	def init(self):
		# all schemes (genotype)
		# all cells (phenotype)
		self.index = 0
		self.cells = []
		i_pos = []
		i_pos.append(width*0.5)
		i_pos.append(10)
		o_pos = []
		o_pos.append(width*0.5)
		o_pos.append(height-10)
		self.inputCell = Cell(-1,i_pos,-1,self.index, self.schemes[0].activationFunctions[0])
		self.index+=1
		self.outputCell = Cell(-1,o_pos,-1,self.index, self.schemes[0].activationFunctions[0])
		self.index+=1
		# starting cell between input and output
		pos = []
		pos.append((i_pos[0]+o_pos[0])*0.5)
		pos.append((i_pos[1]+o_pos[1])*0.5)

		self.cells.append(Cell(0,pos,self.schemes[0].thresholds[0],self.index, self.schemes[0].activationFunctions[0]))
		self.index+=1
		self.cells[0].output_links.append(Link(1.0,self.outputCell.index))
		self.cells[0].input_indices.append(self.inputCell.index)
		self.cells[0].layer = 1
		self.inputCell.output_links.append(Link(1.0,self.cells[0].index))
		self.outputCell.input_indices.append(self.cells[0].index)
	def create(self):
		#print("creating CE: ", n_iterations)
		self.init()
		for i in range(n_iterations):
			self.iterate(i+1)
		self.countCells()
		#print("created network from cellular encoding")
	def mutate(self,MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA):
		for scheme in self.schemes:
			# change the product completely
			if random.uniform(0,1) < MORPH_MUTATION_RATE:
				scheme.p_symbols = []
				n_symbols = random.randint(1,2)
				scheme.weights = []
				for i in range(n_symbols):
					scheme.p_symbols.append(random.randint(0,n_schemes-1))
					scheme.weights.append(random.uniform(0,1))
					scheme.thresholds.append(random.uniform(-1,1))
					scheme.activationFunctions.append(ACTIVATION_FUNCTIONS_SET.get(random.choice(activationFunctions)))
			# change activation functions
			for ac in scheme.activationFunctions:
				if random.uniform(0,1) < MUTATION_RATE:
					ac = ACTIVATION_FUNCTIONS_SET.get(random.choice(activationFunctions))
			# change a single product type
			for symbol in scheme.p_symbols:
				if random.uniform(0,1) < MORPH_MUTATION_RATE:
					symbol = random.randint(0,n_schemes-1)
			# change weights
			for i,w in enumerate(scheme.weights):
				if random.uniform(0,1) < MUTATION_RATE:
					w += random.gauss(w,MUT_SIGMA)
					if w > 1.0:
						w = 1.0
					elif w < -1.0:
						w = -1.0
					scheme.weights[i] = w
			for i,w in enumerate(scheme.thresholds):
				if random.uniform(0,1) < MUTATION_RATE:
					w += random.gauss(w,MUT_SIGMA)
					if w > 1.0:
						w = 1.0
					elif w < -1.0:
						w = -1.0
					scheme.thresholds[i] = w

	def reset(self):
		self.cells.clear()
		self.init()
		self.create()

	def countCells(self):
		count = 0
		# count links for debugging
		for cell in self.cells:
			for link in cell.output_links:
				count+=1
		#print("There are ", len(self.cells) , " cells and a total of ", count, " links")

	def update(self, inputs, requested_number_of_outputs = 1):
		"""
		The update function will update all the cells in the cellular encoding and return an 
		array of outputs. Every cell connected to the output cell will contribute to one value
		in this output array.
		"""
		# flush
		for cell in self.cells:
			if netUpdate == NETWORK_UPDATE.FLUSH:
				cell.activity = 0.0
			if netUpdate == NETWORK_UPDATE.LEAKY:
				cell.activity = cell.activity*leakiness

		cellsToUpdate = []
		for i, link in enumerate(self.inputCell.output_links):
			#inputNR = i % len(inputs)
			inputNR = int(float(i)/float(len(self.inputCell.output_links))*3.0) # % len(inputs)
			#print(inputNR, len(self.inputCell.output_links))
			self.inputCell.output_links[i].weight = inputs[inputNR]

		# alternative 
		# for i,input in enumerate(inputs):
		#	if i < len(self.inputCell.output_links):
		#		self.inputCell.output_links[i].activity = input

		self.inputCell.activity = 1.0
		output = self.inputCell.update()
		for link in self.inputCell.output_links:
			out_cell = None
			for cell in self.cells:
				if cell.index == link.c_index:
					out_cell = cell
			if self.outputCell.index == link.c_index:
				out_cell = self.outputCell
			if out_cell is None:
				Raise("Out cell is none")
			out_cell.activity += output

		output = []

		n_layers = -1
		for cell in self.cells:
			if cell.layer >= n_layers:
				n_layers = cell.layer
		#print(n_layers)
		while len(output) < requested_number_of_outputs:
			for i in range(n_layers + 1 ):
				for cell in self.cells:
					if i == cell.layer:
						out = cell.update()
						#if (i == n_layers-1):
							#print(out)
						for link in cell.output_links:
							if link.c_index == self.outputCell.index:
								#print(cell.activity,out)
								output.append(out)
							for cell in self.cells:
								if link.c_index == cell.index:
									#print(cell.activity,out)
									cell.activity += out
		return output

	def iterate(self,iterationNumber):
		n_cells = len(self.cells)
		for i in range(n_cells):
			cell = self.cells[i]
			if (useMaxCells):
				if len(self.cells) > maxCells:
					return

			if cell.type != -1:
				# not a leaf node
				s = self.schemes[cell.type]
				product = s.p_symbols
				if len(product) == 1:
					# modification
					cell.type = product[0]
					for link in cell.output_links:
						link.weight = s.weights[0]					
				else:
					if s.type == 0:
						# sequential division
						p_c1 = [cell.pos[0],cell.pos[1] - (dis/np.sqrt(iterationNumber))]
						p_c2 = [cell.pos[0],cell.pos[1] + (dis/np.sqrt(iterationNumber))]
						c1 = cell
						c1.type = product[0]
						c1.pos = p_c1
						c1.threshold = s.thresholds[0]
						c1.activationFunction = s.activationFunctions[0]
						c2 = Cell(product[1],p_c2,s.thresholds[1],self.index, s.activationFunctions[1])
						self.index+=1
						c2.layer = c1.layer
						c2.input_indices.append(c1.index)
						for out in c1.output_links:
							out.weight = s.weights[1]
							c2.output_links.append(out)
							c1.output_links.remove(out)
						c1.output_links.append(Link(s.weights[0],c2.index))
						c2.layer+=1
						self.cells.append(c2)
					elif s.type == 1:
						# parallel division
						p_c1 = [cell.pos[0] - (dis/np.sqrt(iterationNumber)), cell.pos[1]]
						p_c2 = [cell.pos[0] + (dis/np.sqrt(iterationNumber)), cell.pos[1]]
						c1 = cell
						c1.type = product[0]
						c1.pos = p_c1
						c1.threshold = s.thresholds[0]
						c1.activationFunction = s.activationFunctions[0];
						c2 = Cell(product[1],p_c2,s.thresholds[1],self.index,s.activationFunctions[1])
						self.index+=1
						c2.layer = c1.layer
						for out in c1.output_links:
							t_c = None
							for cell in self.cells:
								if cell.index == out.c_index:
									t_c = cell
							if (out.c_index == self.outputCell.index):
								t_c = self.outputCell
							if (t_c is None):
								raise("Target cell is none")
							c2.output_links.append(Link(s.weights[1],t_c.index))
							t_c.input_indices.append(c2.index)
							out.weight = s.weights[0]
						for inp in c1.input_indices:
							t_c = inp
							c2.input_indices.append(inp)
							inputCell = None
							for cell in self.cells:
								if cell.index == inp:
									inputCell = cell
							if self.inputCell.index == inp:
								inputCell = self.inputCell
							elif self.outputCell.index == inp:
								inputCell = self.inputCell
							inputCell.output_links.append(Link(inputCell.output_links[0].weight,c2.index))
						self.cells.append(c2)
			

	def display(self,ax):
		color = np.array(cm.viridis((self.inputCell.type+1)/(1+n_schemes)))
		if (displayNodes):
			ax.scatter(self.inputCell.pos[0],self.inputCell.pos[1],c=color)
			ax.scatter(self.outputCell.pos[0],self.outputCell.pos[1],c=color)
		if displayConnections:
			for link in self.inputCell.output_links:
				if displayConnectionWeights:
					color = np.array(cm.viridis((link.weight*0.5)+0.5))
				t_c = None
				for cell in self.cells:
					if cell.index == link.c_index:
						t_c = cell
						break
				x = []
				y = []
				x.append(self.inputCell.pos[0])
				y.append(self.inputCell.pos[1])
				x.append(t_c.pos[0])
				y.append(t_c.pos[1])
				ax.plot(x,y, c=color)
		for cell in self.cells:
			color = cm.viridis((cell.type+1)/(n_schemes+1))
			if displayNodes:
				if (displayActivity):
					color = cm.viridis((cell.activity*0.5)+0.5)
				ax.scatter(cell.pos[0],cell.pos[1], c=color)
			if displayConnections:
				for link in cell.output_links:
					if displayConnectionWeights:
						color = np.array(cm.viridis((link.weight*0.5)+0.5))
					t_c = None
					for c in self.cells:
						if c.index == link.c_index:
							t_c = c
							break
					if link.c_index == self.outputCell.index:
						t_c = self.outputCell
					x = []
					y = []
					x.append(cell.pos[0])
					y.append(cell.pos[1])
					x.append(t_c.pos[0])
					y.append(t_c.pos[1])
					ax.plot(x,y,c=color)
		plt.pause(0.001)
		plt.ion()



if __name__ == "__main__":
	fig, ax = plt.subplots()
	# test mutation
	random.seed(3)
	ce = CE()
	# test random creation
	for i in range(1000):
		ce = CE()
		ce.create()
		ax.clear()
		ce.display(ax)
	for i in range(1000):
		ce.mutate(0.1,0.8)
		ax.clear()
		ce.reset()
		# test updates
		#print(".")
		for j in range(1):
			inputs = []
			for i in range(10):
				  inputs.append(random.uniform(0,1))
			output = ce.update(inputs)
			ce.display(ax)
		
	plt.show()