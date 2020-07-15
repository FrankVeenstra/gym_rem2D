#!/usr/bin/env python

"""
Abstract encoding which all encodings must extend
"""
import numpy as np

class Encoding(object):
	"""Abstract encoding"""
	res = []
		
	@property
	def init(self):
		"""
		Initialize the encoding with random parameters

		This is different from '__iter__' in that this method only returns the
		children directly connected to this module and not their children.
		"""
		# tree depth
		self.treedepth = None
		if self.connection_type:
			for conn in self.connection_type:
				if conn in self._children:
					res.append(self._children[conn])
		return res

	@property
	def create(self):
		"""
		Create and return a 'tree' structure based on the rules of the encoding. 

		This tree structure is unaware of any possible collisions that will happen,
		it's genome is context insensitive. This means that the tree might not 
		be fully expressed when creating the robot out of it. 

	    """
		raise NotImplementedError("Not supported")
	@property
	def mutate(self):
		"""
		Mutate the hyperparameters of the encoding.

		The parameters that create the tree are mutated
		"""
		raise NotImplementedError("Not supported")