"""
This class is a general implementation of a Neural Network 
"""

class Abstract_NN:
	def __init__(self, n_i, n_o):
		self.n_inputs = n_i
		self.n_outputs = n_o
		
	@abstractmethod
	def mutate(mutation_rate):
		pass

	@abstractmethod
	def update(inputs):
		pass

	@abstractmethod
	def crossover(self, other):
		pass


