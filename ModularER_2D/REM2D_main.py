# This is the main file used for the paper : TODO
# The platform was designed by Frank Veenstra, Jørgen Nordmoen and Kyrre Glette
# We are aware that we might not have followed conventional python protocols
# but we are of course more than happy to illustrate what we've done. 
# You can send an email to frankvee(at)uio.no for any questions, suggestions, tips
# and anything you might think is of interest.  

# general 
import numpy as np
import random
import sys

from enum import Enum

import pickle
import multiprocessing
import datetime
import os

# EA
from deap import base,tools,algorithms

# gym
import gym
import gym_rem2D

# The two module types are imported to this file so that all can tweak some

# Encodings:
from Encodings import LSystem as ls
from Encodings import Network_Encoding as nn
from Encodings import Direct_Encoding as de
from Encodings import Cellular_Encoding as ce
import Tree as tree_morph # An encoding creates a tree, a tree creates a robot

# plotting
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import time

# configuration handler
import argparse
import configparser

# fancy printing
import tqdm
from termcolor import colored, cprint

# custom data analysis scripts
# removed, contained hacky scripts. 
import DataAnalysis as da

# singleton equivalent
env = None
def getEnv():
	global env
	if env is None:
		#env = M2D.Modular2D()
		# OpenAI code to register and call gym environment.
		parser = argparse.ArgumentParser()
		parser.add_argument('env_id', nargs='?', default='Modular2DLocomotion-v0', help='Select the environment to run')
		args = parser.parse_args()
		env = gym.make(args.env_id)
	return env

def get_module_list():
	from gym_rem2D.morph import SimpleModule
	from gym_rem2D.morph import CircularModule2D
	module_list = []
	for i in range(4):
		module_list.append(SimpleModule.Standard2D())
	for i in range(4):
		module_list.append(CircularModule2D.Circular2D())
	return module_list

class Encoding_Type(Enum):
	DIRECT = 0
	LSYSTEM = 1
	NEURAL_NETWORK = 2
	CELLULAR_ENCODING = 3

class Individual:
	def __init__(self):
		self.genome = None
		self.fitness = 0

	@staticmethod
	def random(moduleList=None,config=None, encoding = 'lsystem'):
		# creates a random individual based on the encoding type
		self = Individual()
		if moduleList is None:
			moduleList = get_module_list()
		if (config is not None):
			enc = config['encoding']['type']
			if enc == 'direct':
				self.ENCODING_TYPE = Encoding_Type.DIRECT
				self.genome = de.DirectEncoding(moduleList,config)
			elif enc == 'lsystem':
				self.ENCODING_TYPE = Encoding_Type.LSYSTEM
				self.genome = ls.LSystem(moduleList, config)
			elif enc == 'cppn':
				self.ENCODING_TYPE = Encoding_Type.NEURAL_NETWORK
				self.genome = nn.NN_enc(moduleList, "CPPN", config=config)
			elif enc == 'ce':
				self.ENCODING_TYPE = Encoding_Type.CELLULAR_ENCODING
				self.genome = nn.NN_enc(moduleList, "CE",config=config)
			self.tree_depth = int(config['morphology']['max_depth'])
			tree = self.genome.create(self.tree_depth)
			self.fitness = 0
			return self
		else:
			if encoding == 'direct':
				self.ENCODING_TYPE = Encoding_Type.DIRECT
				self.genome = de.DirectEncoding(moduleList)
			elif encoding == 'lsystem':
				self.ENCODING_TYPE = Encoding_Type.LSYSTEM
				self.genome = ls.LSystem(moduleList)
			elif encoding == 'cppn':
				self.ENCODING_TYPE = Encoding_Type.NEURAL_NETWORK
				self.genome = nn.NN_enc(moduleList, "CPPN")
			elif encoding == 'ce':
				self.ENCODING_TYPE = Encoding_Type.CELLULAR_ENCODING
				self.genome = nn.NN_enc(moduleList, "CE")
			self.tree_depth = 8
			self.genome.create(self.tree_depth)
			return self

	def mutate(MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA,self):
		self.genome.mutate(MORPH_MUTATION_RATE,MUTATION_RATE,MUT_SIGMA)
		# should add crossover mutations, have to be done uniquely for each encoding though
		# TODO: self.genome.crossover(CROSSOVER_RATE, other_genome);



class run2D():
	"""
	This is the main instance of the Modular 2D environment initializer.
	"""
	def __init__(self, config, dir):
		print(config.items())
		
		self.start_time = datetime.datetime.now()
		self.time = datetime.datetime.now()
		self.initialize_parameters_from_config_file(dir,config)
		self.fitnessData = da.FitnessData() # stores data of the progression

		# TODO take from configuration file
		self.EVALUATION_STEPS = 10000
		self.TOTAL_EVALUATIONS = 50000
		self.SAVEDATA = True

		# Initializing modules
		self.moduleList = get_module_list() # stores which module types to select from. This list is mutated using the L-System


		if self.load_best:
			print("Loading best")
			individual = pickle.load(open(self.SAVE_FILE_DIRECTORY + self.BEST_INDIVIDUAL_FILE,"rb"))
			for i in range(20):
				evaluate(individual,HEADLESS = False,INTERVAL =2 + i)

	def run(self, config, continue_progression=False):
		'''
		This function is a placeholder for starting the environment.
		The continue progression file specifies whether it should load an evolutionary run that might
		have crashed, or that you want to continue with perhaps different parameters. 
		'''
		if (continue_progression):
			try:
				self.fitnessData = pickle.load(open(self.SAVE_FILE_DIRECTORY,"rb"))
				population = pickle.load(open(self.SAVE_FILE_DIRECTORY + self.POPULATION_FILE,"rb"))
				self.plotter.plotFitnessProgress(self.fitnessData )
				print("Found existing population, continueing evolution")
				self.run_deap(config,population = population)
			except:
				raise("Could not find file to continue")
		self.run_deap(config)


	def initialize_parameters_from_config_file(self,dir, config):
		# TODO: Should I access the config directly? Or store variables here.
		self.config = config

		# Variables for storing data
		self.BEST_INDIVIDUAL_FILE =  "elite"
		self.POPULATION_FILE =  "pop"
		self.SAVE_FILE_DIRECTORY = os.path.join(dir, 's_')
		self.CHECKPOINT_FREQUENCY = int(config['experiment']['checkpoint_frequency'])

		# Keeping track of evolutionary progression
		self.EVALUATION_NR = 0
		self.POPULATION_SIZE = int(config['ea']['batch_size'])

		# Mutation rates
		self.MUTATION_RATE = float(config['ea']['mutation_prob'])
		self.MORPH_MUTATION_RATE = float(config['ea']['morphmutation_prob'])
		self.MUT_SIGMA = float(config['ea']['mutation_sigma'])
		self.TREE_DEPTH = int(config['morphology']['max_depth'])

		# 
		print("Mutation rates - ", " control: " , self.MUTATION_RATE, ", morphology: ", 
		self.MORPH_MUTATION_RATE, ", sigma: ", self.MUT_SIGMA)

		# Wall of death speed
		self.WOD_SPEED = float(config['evaluation']['wod_speed'])		

		# This parameter is used for showing the best individual every generation.
		# NOTE: this apparently doesn't work when headlessly simulating the rest
		self.show_best = False
		if (int(config['ea']['show_best']) == 1):
			self.show_best = True
		self.headless = False
		if (int(config['ea']['headless']) == 1):
			self.headless = True
		self.load_best = False
		if (int(config['ea']['load_best']) == 1):
			self.load_best = True
		# plots the virtual creates at every <interval> frames 
		self.interval = int(config['ea']['interval'])

		# Elements for visualization
		# plot fitness over time
		self.PLOT_FITNESS = False
		if (int(config['visualization']['v_progression']) == 1):
			self.PLOT_FITNESS = True
		# plot tree structure of current individual being evaluated (for debugging)
		self.PLOT_TREE = False
		if (int(config['visualization']['v_tree']) == 1):
			self.PLOT_TREE = True

	def run_deap(self, config, population = None):
		'''
		This function initializes and runs an EA from DEAP. You can find more information on how you can use DEAP
		at: https://deap.readthedocs.io/en/master/examples/es_fctmin.html 
		'''
		toolbox = base.Toolbox()
		toolbox.register("individual", Individual.random, self.moduleList, self.config)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", evaluate,HEADLESS = self.headless, TREE_DEPTH = self.TREE_DEPTH)
		toolbox.register("mutate", Individual.mutate, self.MORPH_MUTATION_RATE,self.MUTATION_RATE,self.MUT_SIGMA)
		toolbox.register("select",tools.selTournament, tournsize = 4)

		N_GENERATIONS = 1+ int(int(config['ea']['n_evaluations'])/self.POPULATION_SIZE)
		N_GENERATIONS -= len(self.fitnessData.avg)

		if config["ea"]["headless"] == "1":
			n_cores = int(self.config["ea"]["n_cores"])
			print("Starting deap in headless mode using " , n_cores , " cores")
			print("Evolution will run for ", N_GENERATIONS, " generations, with a population size of ", self.POPULATION_SIZE)
			pool = multiprocessing.Pool(n_cores)
			cs = int(np.ceil(float(self.POPULATION_SIZE)/float(n_cores)))
			toolbox.register("map", pool.map, chunksize=cs)

		# create population when none is given as an argument
		if population is None:
			population = toolbox.population(n=self.POPULATION_SIZE)
			fitnesses = toolbox.map(toolbox.evaluate, population) # list(toolbox.evaluate,population))
			for ind, fit in zip(population, fitnesses):
				ind.fitness = fit
		
		gen = 0 # keep track of generations simulated
		print("headless mode:", self.headless)
		
		if self.headless:
			writer = sys.stdout
			range_ = range(N_GENERATIONS)
		else:
			writer = range_ = tqdm.trange(N_GENERATIONS, file=sys.stdout)

		for i in range_:
			gen+=1
			offspring = toolbox.select(population, len(population))

			# deep copy of selected population
			offspring = list(map(toolbox.clone, offspring))
			for o in offspring:
				toolbox.mutate(o)
				# TODO only reset fitness to zero when mutation changes individual
				# Implement DEAP built in functionality
				o.fitness = 0
			fitnesses = toolbox.map(toolbox.evaluate, offspring) # list(map(toolbox.evaluate, offspring))

			fitness_values = []
			for ind, fit in zip(offspring, fitnesses):
				ind.fitness = fit
				fitness_values.append(fit)

			population = offspring
			min = np.min(fitness_values)
			max = np.max(fitness_values)
			mean = np.mean(fitness_values)
			dt = datetime.datetime.now()-self.time
			self.time = datetime.datetime.now()
			writer.write("Generation %d evaluated ( %s ) : Min %s, Max %s, Avg %s" % (i + 1, dt,min,max,mean))
			if self.headless:
				writer.write("\n")
			self.EVALUATION_NR+=len(population)

			#print(float(self.EVALUATION_NR)/ float(self.TOTAL_EVALUATIONS) * float(100), "%")
			self.fitnessData.addFitnessData(fitness_values,gen)
			if self.SAVEDATA:
				if (i % self.CHECKPOINT_FREQUENCY == 0 or i == N_GENERATIONS):
					#self.fitnessData.save(self.SAVE_FILE_DIRECTORY)
					self.fitnessData.save(self.SAVE_FILE_DIRECTORY)
					pickle.dump(population,open(self.SAVE_FILE_DIRECTORY + self.POPULATION_FILE + str(i), "wb"))

			if self.PLOT_FITNESS:
				self.plotter.plotFitnessProgress(self.fitnessData)
				if (self.PLOT_TREE):
					self.plotter.displayDivs(self.fitnessData)

			# save only the best fit individual; currently, all other individuals of the population are discarded.
			bestfit = 0.0
			bestOffspring = None
			for o in offspring:
				if o.fitness > bestfit:
					bestfit = o.fitness
					bestOffspring = o
					pickle.dump(o,open(self.SAVE_FILE_DIRECTORY + self.BEST_INDIVIDUAL_FILE + str(i), "wb"))

			# To show the best individual
			if (self.show_best == True):
				# Hacky to only  display the best individuals when running in headless mode. 
				# However, when not connecting for a time, the box2D window freezes. TODO
				switch = False
				if (self.headless == True):
					self.headless = False
					switch = True
				# debug prints:
				print("Fitness of best = ", toolbox.evaluate(bestOffspring,INTERVAL=5,HEADLESS=False))
				#print("This fitness is stored as: ", bestOffspring.fitness)
				#print(toolbox.evaluate(bestOffspring))
				if switch == True:
					self.headless = True

			if (datetime.datetime.now() - self.start_time).seconds > int(config.get("ea", "wallclock_time_limit")):
				print("Reached wall-clock time limit. Stopping evolutionary run")
				break

def evaluate(individual, EVALUATION_STEPS= 10000, HEADLESS=True, INTERVAL=100, ENV_LENGTH=100, TREE_DEPTH = None, CONTROLLER = None):
	env = getEnv()
	if TREE_DEPTH is None:
		try:
		   TREE_DEPTH = individual.tree_depth
		except:
			raise Exception("Tree depth not defined in evaluation")
	tree = individual.genome.create(TREE_DEPTH)
	env.seed(4)
	env.reset(tree=tree, module_list=individual.genome.moduleList)

	it = 0
	fitness = 0
	for i in range(EVALUATION_STEPS):
		it+=1
		if it % INTERVAL == 0 or it == 1:
			if not HEADLESS:
				env.render()

		action = np.ones_like(env.action_space.sample())
		
		observation, reward, done, info  = env.step(action)
		
		if reward< -10:
			break
		elif reward > ENV_LENGTH:
			reward += (EVALUATION_STEPS-i)/EVALUATION_STEPS
			fitness = reward
			break
		if reward > 0:
			fitness = reward

	# Not necessary, just to keep the keep evolutionary algorithm lingo
	return fitness


def setup():
	parser = argparse.ArgumentParser(description='Process arguments for configurations.')
	parser.add_argument('--file',type = str, help='config file', default="0.cfg")
	parser.add_argument('--seed',type = int, help='seed', default=0)
	parser.add_argument('--headless',type = int, help='headless mode', default=0)
	parser.add_argument('--n_processes',type = int, help='number of processes to use', default=1)
	parser.add_argument('--output',type = str, help='output directory', default='')
	parser.add_argument('--wallclock-time-limit', type=int, help='wall-clock limit in seconds', default=sys.maxsize)
	args = parser.parse_args()
	random.seed(int(args.seed))
	np.random.seed(int(args.seed))

	config = configparser.ConfigParser()

	directory = os.path.dirname(os.path.abspath(__file__))
	orig_cwd = os.getcwd()
	print("original CWD:", orig_cwd)
	os.chdir(directory)

	expnr = int(args.seed)
	if int(expnr) < 0:
		raise("invalid experiment number")

	config_to_read = os.path.join(directory,str(args.file))
	print('reading: ', config_to_read)
	if not os.path.isfile(config_to_read):
		sys.exit("Could not find configuration file, quitting")
	config.read(config_to_read)

	general_config = os.path.join(directory , '_g.cfg')
	print('reading: ', general_config)
	if not os.path.isfile(general_config):
		print("No common configuration specified")
	config.read(general_config)

	print("working from ", directory)
	for each_section in config.sections():
		for (each_key, each_val) in config.items(each_section):
			print(each_key, each_val)

	newdir = os.path.join(orig_cwd, args.output)
	if not os.path.exists(newdir):
		os.makedirs(newdir)
		print("created the ", newdir)

	config.set("ea", "wallclock_time_limit", str(args.wallclock_time_limit))
	return config, newdir

if __name__ == "__main__":
	config, dir = setup()
	experiment = run2D(config,dir)
	experiment.run(config)
