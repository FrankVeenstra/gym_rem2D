import math
import random

class Controller:
	def __init__(self):
		# For now i_state is a placeholder for the value of the internal state
		self.i_state = 0
		self.output = 0
		self.MAX_AMP = 1
		self.MAX_PHASE = 1
		self.MAX_OFFSET = math.pi
		self.MAX_FREQ = 0.1
		self.amplitude = random.uniform(0,self.MAX_AMP)
		self.phase = random.uniform(-self.MAX_PHASE,self.MAX_PHASE)
		self.frequency = random.uniform(-self.MAX_FREQ,self.MAX_FREQ)
		self.offset = random.uniform(-self.MAX_OFFSET,self.MAX_OFFSET)
	def update(self, input):
		self.phase+=input
		self.i_state += self.frequency
		self.output = (self.amplitude * (math.sin(self.i_state + self.phase)))+self.offset
		return self.output

	# hacky function to ensure mutation doesn't create out of bounds results. 
	def minMax(self, angle):
		if (self.amplitude > self.MAX_AMP):
			self.amplitude = self.MAX_AMP
		elif (self.amplitude < 0):
			self.amplitude = 0
		if (self.phase > self.MAX_PHASE):
			self.phase = self.MAX_PHASE
		elif (self.phase < -self.MAX_PHASE):
			self.phase = -self.MAX_PHASE
		if (self.frequency > self.MAX_FREQ):
			self.frequency = self.MAX_FREQ
		elif (self.frequency < -self.MAX_FREQ):
			self.frequency = -self.MAX_FREQ
		if (self.offset > angle/2):
			self.offset = angle/2
		elif (self.offset < -angle/2):
			self.offset = -angle/2

	# Hacky code to set controller settings manually 
	def setControl(self,a,b,c,d, angle):
		self.amplitude = ((a + 1.0) * 0.5) * self.MAX_AMP
		self.phase = b * self.MAX_PHASE
		self.offset = c * self.MAX_OFFSET
		self.frequency = d * self.MAX_FREQ
		self.minMax(angle)

	def mutate(self, mutationrate,sigma, angle):
		if random.uniform(0.0,1.0) < mutationrate:
			self.amplitude += random.gauss(self.amplitude,sigma)
		if random.uniform(0.0,1.0) < mutationrate:
			self.phase += random.gauss(self.phase,sigma)
		if random.uniform(0.0,1.0) < mutationrate:
			self.frequency += random.gauss(self.frequency,sigma*0.1)
		if random.uniform(0.0,1.0) < mutationrate:
			self.offset += random.gauss(self.offset,sigma)
		self.minMax(angle)