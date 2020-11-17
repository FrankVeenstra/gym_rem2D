import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import cm
import matplotlib.transforms as transforms
import math
import os
import configparser
import scipy.stats as st


""" for loading data from a single path """
def load_data(path, filename) :
	if path is None:
		raise Exception("No path specified for 'load_data' ")
	if not os.path.exists(path):
		print("Directory does not exist, trying working directory ", path)
		path = ""

	file_name = path + filename
	try:
		f = open(file_name)
		f.close()
	except:
		raise Exception("Cannot find file : ", file_name)
	try:
		with open(file_name) as f:
			fit_data = pickle.load(open(file_name,"rb"))
			# add associated configuration file
			print("loaded fitness data")
			return fit_data
	except:
		raise Exception("Could not unpickle the fitness data of experiment : ", file_name)


""" A helper class that simply stores some values that can easily be plotted
	Note: just to use to plot fitness over time while the program is running, 
	all other data of the runs will be stored anyway. """
class FitnessData:
	def __init__(self):
		self.p_0 = []		# 0th percentile
		self.p_25 = []		# 25th percentile
		self.p_50 = []		# 50th percentile
		self.p_75 = []		# 75th percentile
		self.p_100 = []		# 100th percentile
		self.avg = []		# average
		self.divValues =[]	# diversity values
	def save(self, saveFile, num = ''):
		pickle.dump(self,open(saveFile + str(num),"wb"))
	def addFitnessData(self,fitnesses, gen):
		self.avg.append(np.average(fitnesses))
		self.p_0.append(np.percentile(fitnesses,0))
		self.p_25.append(np.percentile(fitnesses,25))
		self.p_50.append(np.percentile(fitnesses,50))
		self.p_75.append(np.percentile(fitnesses,75))
		self.p_100.append(np.percentile(fitnesses,100))

""" Very simple plotter, plots the percentiles and average fitness """ 

def plot_basic_fitness(fitness_data):
	cmap = cm.viridis
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	x = range(0,len(fitness_data.avg))
	ax.plot(fitness_data.avg,color = 'black')
	ax.fill_between(x,fitness_data.p_0,fitness_data.p_100, color = 'black', alpha = 0.01)
	ax.fill_between(x,fitness_data.p_25,fitness_data.p_75, color = 'black', alpha = 0.2)
	ax.set_xlabel("Generations")
	ax.set_ylabel("Fitness")


def plot_fitness(path, filename):
	# load the fitness data
	fitness_data = load_data(path,filename)
	# plot the fitness data
	plot_basic_fitness(fitness_data)


if __name__ == "__main__":	
	# add a path where you saved your files, make sure to end with '/'
	path = "e.g. D:/your_file_folder/"
	# add the filename of your data (the name of the pickled FitnessData object)
	filename="s_"
	# simple function to plot the progression of a single evolutionary run
	plot_fitness(path,filename);	
	plt.show()



