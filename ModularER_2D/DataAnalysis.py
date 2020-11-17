import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.patches import Ellipse
from matplotlib import cm
import matplotlib.transforms as transforms
import math
import os
import configparser
import scipy.stats as st
import copy

""" for loading data from multiple paths path. Note that it's quite incromprihensible """
def load_datas(paths = None, loadFromExperimentSubfolder = False):
	if paths is None:
		paths = ['C:/result_temp_final','D:/results/cppn_final/','D:/results/ce_final/' ]
	for directory in paths:
		if not os.path.exists(directory):
			raise Exception("Cannot find directory: ", directory)
	file_present = True
	count = 0
	fitness_datas = []
	sort = None

	path_id = 0
	directory = paths[0]
	while file_present:
		file_config = os.path.join(directory + str(count) + '.cfg')
		config = configparser.ConfigParser()
		try:
			f = open(file_config)
			f.close()
			config.read(file_config)
		except:
			print("Could not find file: " , count, " in path " , directory)

		file_name = directory + 's_'
		if (loadFromExperimentSubfolder is True):
			try:
				f = open(file_name)
				f.close()
			except:
				file_name = directory +str(count)+'/s_'
		else:
			file_name = directory + 's_'
		try:
			with open(file_name) as f:
				fit_data = pickle.load(open(file_name,"rb"))
				# add associated configuration file
				fit_data.config = config
				fitness_datas.append(fit_data)
				print("found file ", count)
			count+=1
				#print(count)
				#plotter.plotFitnessProgress(fitness_datas[0])
		except:
			print("Could not unpickle the fitness data of experiment ", count)
			count+=1
			if (count >30 and count < 60):
				continue
			path_id+=1
			if (path_id < len(paths)):
				directory = paths[path_id]
				count = 0
			else:
			#print(f.readlines())
				file_present = False
		#if count > 256:
		#	file_present = False
	return fitness_datas

""" for loading data from a single path """
def load_data(path, filename) :
	if path is None:
		raise Exception("No path specified for 'load_data' ")
		print("No path specified for 'load_data'")
	if not os.path.exists(path):
		raise Exception("Cannot find directory: ", path)

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

def save_fitness_data(SAVE_FILE_DIRECTORY, fitnessData):
	fd = copy.deepcopy(fitnessData);
	fd.save(SAVE_FILE_DIRECTORY)



def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

class NodeVisualization:
	def __init__(self,position,orientation, index,t, controller):
		self.pos = position
		self.theta = orientation
		self.index = index
		self.handled = False
		self.type = t
		self.controller = controller



def compare_distance(tree,target):
	"""
	Checks tree edit distance. Since every node has a unique position, we know that the node is the
	same when the positions are the same. Hence, a simple method of counting the number of edits 
	one needs to do to create the target tree out of a given tree is equal to the number of positional
	differences. 
	"""
	# check for positional overlap
	edit_value = 0
	for node in target:
		node.found = False
	for node in tree:
		same_node = False
		for t_node in target:
			if node.pos[0] == t_node.pos[0] and node.pos[1] == t_node.pos[1]:
				same_node = True
				t_node.found = True
		if same_node == False:
			edit_value += 1
	# count found
	for node in target:
		if not node.found:
			edit_value += 1
	return edit_value

def get_tree_pos(tree, module_list=None) :
	colorActivation = True
	tree_vis = []
	nodes = tree.getNodes()
	n_nodes = len(nodes)
	# get first module (there should be only one module with no parent; -1 or None)
	for i,n in enumerate(nodes):
		if (n.parent == -1 or n.parent is None):
			pos = np.array([0,0]) # initial node is at position 0,0
			theta = 0
			tree_vis.append(NodeVisualization(pos, theta, i,n.type, n.controller))
			break

	# expand the tree
	for i in range(10):
		vn_size = len(tree_vis)
		for j in range(vn_size):
			vn = tree_vis[j]
			if not vn.handled:
				vn.handled = True
				for n in nodes:
					if (n.parent == vn.index):
						# found parent
						# base the angle on the current angle, the layer depth, and the maximum number of connection sites
						max_connections = 3
						if module_list:
							max_connections = len(module_list[n.type])

						orientation = n.parent_connection_coordinates.value[0] # should be a value between -1 and 1 
						#print(n.type)
						#maxCon = 3 # TODO change to be module specific
						angle = orientation
						#print("i",i)
						theta = (angle / math.pow(2,((i+1.5))) * (2*math.pi)) + vn.theta # absolute orientation
						#print(theta)
						pos = []
						pos.append((math.sin(theta) * 1) + vn.pos[0])
						pos.append((math.cos(theta) * 1) + vn.pos[1])
						tree_vis.append(NodeVisualization(pos,theta,n.index,n.type, n.controller))
						x = []
						x.append(vn.pos[0])
						x.append(pos[0])
						y = []
						y.append(vn.pos[1])
						y.append(pos[1])
	#print ("len vis tree : ", len(tree_vis), " len tree : ", len(nodes))
	
	return tree_vis

def tree_edit_distance(population):
	t_population = []
	for ind in population:
		t_population.append(get_tree_pos(ind.genome.create(ind.tree_depth)))
	divValues = []
	for c_ind in t_population:
		divValue = 0.0
		for t_ind in t_population:
			if t_ind is c_ind:
				continue
			else:
				divValue+=compare_distance(c_ind,t_ind)
		divValues.append(divValue)
	return divValues
	
class Plotter:
	def __init__(self):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(2,2,1)
		self.ax2 = self.fig.add_subplot(2,2,2)
		self.ax3 = self.fig.add_subplot(2,2,3)
		self.ax4 = self.fig.add_subplot(2,2,4)
		self.cmap = plt.get_cmap('viridis')
	def plotFitnessProgress(self,fitnessData,ax=None):
		if ax == None: 
			ax = self.ax2
		ax.clear()
		ax.plot(fitnessData.avg, color = 'black')
		xs =[]
		for v in range(len(fitnessData.p_0)):
			xs.append(v)
		ax.fill_between(xs,fitnessData.p_0,fitnessData.p_100, color = 'black', alpha = 0.1)
		ax.fill_between(xs,fitnessData.p_25,fitnessData.p_75, color = 'black', alpha = 0.3)
		plt.pause(0.001)
		plt.ion()

	def displayDivs(self,fitnessData,ax=None,ax2=None):
		if ax == None:
			ax=self.ax3
		if ax2 == None:
			ax2=self.ax4
		ax.clear()
		ax2.clear()
		prevys = []
		prevxs = []
		interval = 1
		for i,vals in enumerate(fitnessData.divValues):

			xs = []
			ys = []

			color = self.cmap(1-(i/len(fitnessData.divValues)))
			for pv in vals:
				ys.append(pv)

			if i % interval == 0:
				if len(prevys) > 0:
					xs = []
					xs.append(i-interval)
					xs.append(i)
					c_yavg = []
					c_ymax = []
					c_ymin = []
					c_y75 = []
					c_y25 = []
					c_yavg.append(np.average(prevys))
					c_yavg.append(np.average(ys))
					c_ymax.append(np.max(prevys))
					c_ymax.append(np.max(ys))
					c_ymin.append(np.min(prevys))
					c_ymin.append(np.min(ys))
					c_y75.append(np.percentile(prevys,75))
					c_y75.append(np.percentile(ys,75))
					c_y25.append(np.percentile(prevys,25))
					c_y25.append(+np.percentile(ys,25))
				
					ax2.plot(xs,c_yavg, c ='red')
					ax2.fill_between(xs,c_ymin,c_ymax, alpha = 0.1, color ='red')
					ax2.fill_between(xs,c_y25,c_y75, alpha = 0.3, color ='red')

				prevxs = xs
				prevys = ys

	def displayDivsxy(self,fitnessData,ax=None,ax2=None):
		if ax == None:
			ax=self.ax3
		if ax2 == None:
			ax2=self.ax4
		ax.clear()
		ax2.clear()
		prevxs = []
		prevys = []
		interval = 2
		for i,vals in enumerate(fitnessData.divValues):
			xs = []
			ys = []
			color = self.cmap(1-(i/len(fitnessData.divValues)))

			for pv in vals:
				xs.append(pv[0])
				ys.append(pv[1])
				ax.scatter(pv[0],pv[1],c = color, alpha = i/len(fitnessData.divValues))
			confidence_ellipse(np.array(xs),np.array(ys),ax,edgecolor=color,facecolor=color, alpha = 0.25)#i/len(fitnessData.divValues))

			if i % interval == 0:
				if len(prevxs) > 0:
					xs = []
					xs.append(i-interval)
					xs.append(i)
					c_yavg = []
					c_ymax = []
					c_ymin = []
					c_y75 = []
					c_y25 = []
					c_yavg.append(np.average(prevxs) +np.average(prevys))
					c_yavg.append(np.average(xs) + np.average(ys))
					c_ymax.append(np.max(prevxs) +np.max(prevys))
					c_ymax.append(np.max(xs) + np.max(ys))
					c_ymin.append(np.min(prevxs) +np.min(prevys))
					c_ymin.append(np.min(xs) + np.min(ys))
					c_y75.append(np.percentile(prevxs,75)+np.percentile(prevys,75))
					c_y75.append(np.percentile(xs,75)+np.percentile(ys,75))
					c_y25.append(np.percentile(prevxs,25)+np.percentile(prevys,25))
					c_y25.append(np.percentile(xs,25)+np.percentile(ys,25))
				
					ax2.plot(xs,c_yavg, c ='red')
					ax2.fill_between(xs,c_ymin,c_ymax, alpha = 0.1, color ='red')
					ax2.fill_between(xs,c_y25,c_y75, alpha = 0.3, color ='red')

				prevxs = xs
				prevys = ys

	def plotFitness(self,fitnesses, ax,gen):
		x = []
		for f in fitnesses:
			x.append(gen)
		ax.scatter(x,fitnesses)
		plt.pause(0.001)
		plt.ion()

	def setDivValue(self,individual):
		return


class FitnessData:
	# A helper class that simply stores some values that can easily be plotted
	# Note: just to use to plot fitness over time while the program is running, 
	# all other data of the runs will be stored anyway. 
	def __init__(self):
		self.p_0 = []
		self.p_25 = []
		self.p_50 = []
		self.p_75 = []
		self.p_100 = []
		self.avg = []
		self.divValues =[]
	def save(self, saveFile, num = ''):
		pickle.dump(self,open(saveFile + str(num),"wb"))
	def addFitnessData(self,fitnesses, gen):
		self.avg.append(np.average(fitnesses))
		self.p_0.append(np.percentile(fitnesses,0))
		self.p_25.append(np.percentile(fitnesses,25))
		self.p_50.append(np.percentile(fitnesses,50))
		self.p_75.append(np.percentile(fitnesses,75))
		self.p_100.append(np.percentile(fitnesses,100))

#def plot_encoding_data(ax, data, title):

def plot_diversity_data(ax, all_data, encoding_type, sort,plotIndividualLines=False, x_limit = 1000, y_limit = 2000):
	cmap = cm.viridis
	sorts = []
	fit_data_sorted = []
	for data in all_data:
		if data.config['encoding']['type'] == encoding_type:
			if (sort != None):
				value = data.config['ea'][sort]
				if value not in sorts:
					sorts.append(value)
					fit_data_sorted.append([data])
				else:
					a_pos = 0
					for i,val in enumerate(sorts):
						if value == val:
							a_pos = i
							break
					fit_data_sorted[a_pos].append(data)
			else:
				#if ((fit_data_sorted) == 0):
					#fit_data_sorted.append([])
				fit_data_sorted.append([data])

	#ax.set_title(encoding_type)
	labels = []
	axdatas = []
	for i,datas in enumerate(fit_data_sorted):
		color = cmap(i/(len(fit_data_sorted)))
		axdata = []
		for j,data in enumerate(datas):
			divLine = []
			for divs in data.divValues:
				divLine.append(np.average(divs))
			axdata.append(divLine)
			if plotIndividualLines:
				line, = ax.plot(divLine,color = color)
				label = sorts[i]
				if label not in labels:
					labels.append(label)
					line.set_label(label)
		axdatas.append(axdata)
	#return
	for i,datas in enumerate(axdatas):
		color = cmap(float(i)/7.0)
		x = []
		ys = []
		ysmedian = []
		ymax = []
		y75 = []
		ymin = []
		y25 = []
		nda = np.array(datas)
		nd = nda.transpose()
		count = 0
		for vals in nd:
			ys.append(np.average(vals))
			ysmedian.append(np.median(vals))
			ymax.append(np.percentile(vals, 95))
			ymin.append(np.percentile(vals, 5))
			y75.append(np.percentile(vals, 75))
			y25.append(np.percentile(vals, 25))
			x.append(count)
			count +=1
		ax.plot(ys, color = color, alpha=0.8)
		#ax.plot(ysmedian, color = color)
		ax.fill_between(x,ymin,ymax, color = color, alpha = 0.04)
		ax.fill_between(x,y25,y75, color = color, alpha = 0.2)
	#ax.legend()
	ax.set_xlim(0, x_limit)
	ax.set_ylim(0, y_limit)

def plot_diversity_data_together(ax, all_data, sort,plotIndividualLines=False):
	cmap = cm.viridis
	sorts = []
	fit_data_sorted = []
	for data in all_data:
		if (sort != None):
			value = data.config['encoding'][sort]
			if value not in sorts:
				sorts.append(value)
				fit_data_sorted.append([data])
			else:
				a_pos = 0
				for i,val in enumerate(sorts):
					if value == val:
						a_pos = i
						break
				fit_data_sorted[a_pos].append(data)
		else:
			#if ((fit_data_sorted) == 0):
				#fit_data_sorted.append([])
			fit_data_sorted.append([data])
	#ax.set_title(encoding_type)
	labels = []
	axdatas = []
	for i,datas in enumerate(fit_data_sorted):
		color = cmap(i/(len(fit_data_sorted)))
		axdata = []
		for j,data in enumerate(datas):
			divLine = []
			for divs in data.divValues:
				divLine.append(np.average(divs))
			axdata.append(divLine)
			if plotIndividualLines:
				line, = ax.plot(divLine,color = color)
				label = sorts[i]
				if label not in labels:
					labels.append(label)
					line.set_label(label)
		axdatas.append(axdata)
	#return
	for i,datas in enumerate(axdatas):
		color = cmap(float(i)/3.0)
		x = []
		ys = []
		ysmedian = []
		ymax = []
		y75 = []
		ymin = []
		y25 = []
		nda = np.array(datas)
		nd = nda.transpose()
		count = 0
		conf = []
		conf_0 = []
		conf_1 = []
		for vals in nd:
			ys.append(np.average(vals))
			ysmedian.append(np.median(vals))
			ymax.append(np.percentile(vals, 95))
			ymin.append(np.percentile(vals, 5))
			y75.append(np.percentile(vals, 75))
			y25.append(np.percentile(vals, 25))
			x.append(count)
			c = st.t.interval(0.95, len(vals)-1, loc=np.mean(vals), scale=st.sem(vals))
			conf.append(c)
			conf_0.append(c[0])
			conf_1.append(c[1])

			count +=1
		line, = ax.plot(ys, color = color, alpha=0.8)
		#ax.plot(ysmedian, color = color)
		#ax.fill_between(x,ymin,ymax, color = color, alpha = 0.04)
		#ax.fill_between(x,y25,y75, color = color, alpha = 0.2)
		ax.fill_between(x,conf_0,conf_1, color = color, alpha = 0.2)
		label = sorts[i]
		if label not in labels:
			labels.append(label)
			line.set_label(label)
	#ax.legend()
	ax.set_xlim(0, 1000)
	ax.set_ylim(0,2500)


def plot_encoding_data(ax, all_data, encoding_type, sort,plotIndividualLines=False, x_limit = 1000, y_limit = 80):
	title = "NAN"
	if encoding_type == 'lsystem':
		title = "L-System"
	elif encoding_type == 'cppn':
		title = "CPPN"
	elif encoding_type == 'ce':
		title = "CE"
	elif encoding_type == 'direct':
		title = "Direct"
	n_generations = 0
	cmap = cm.viridis
	sorts = []
	fit_data_sorted = []
	for data in all_data:
		if data.config['encoding']['type'] == encoding_type:
			value = data.config['ea'][sort]
			if value not in sorts:
				sorts.append(value)
				fit_data_sorted.append([data])
			else:
				a_pos = 0
				for i,val in enumerate(sorts):
					if value == val:
						a_pos = i
						break
				fit_data_sorted[a_pos].append(data)
	#ax.set_title(title, va='bottom')
	labels = []
	axdatas = []
	for i,datas in enumerate(fit_data_sorted):
		color = cmap(i/(len(fit_data_sorted)))
		axdata = []
		for data in datas:
			axdata.append(data.p_100)
			if plotIndividualLines:
				line, = ax.plot(data.p_100,color = color, linestyle = "-.")
				label = sorts[i]
				if label not in labels:
					labels.append(label)
					line.set_label(label)
		axdatas.append(axdata)
		
	for i,datas in enumerate(axdatas):
		color = cmap(float(i)/7.0)
		x = []
		ys = []
		ysmedian = []
		ymax = []
		y75 = []
		ymin = []
		y25 = []
		nda = np.array(datas)
		nd = nda.transpose()
		count = 0
		for vals in nd:
			ys.append(np.average(vals))
			ysmedian.append(np.median(vals))
			ymax.append(np.percentile(vals, 95))
			ymin.append(np.percentile(vals, 5))
			y75.append(np.percentile(vals, 75))
			y25.append(np.percentile(vals, 25))
			x.append(count)
			count +=1
		if len(ys) > 0:
			n_generations = len(ys)
		line, = ax.plot(ys, color = color)
		#line2, = ax.plot(ysmedian, color = color)
		label = sorts[i]
		if label not in labels:
			labels.append(label)
			line.set_label(label)
		#ax.fill_between(x,ys,ysmedian, color = color, alpha = 0.4)
		ax.fill_between(x,ymin,ymax, color = color, alpha = 0.04)
		ax.fill_between(x,y25,y75, color = color, alpha = 0.2)
	#ax.legend()
	ax.set_xlim(0, x_limit)
	#ax.set_ylim(bottom=0)
	ax.set_ylim(0,y_limit)

def plot_diversity_datas(fitness_datas,sort = None, together = False):
	#fig = plt.figure(figsize = (6,6))
	#ax1 = fig.add_subplot(2,2,1)
	#ax2 = fig.add_subplot(2,2,2)
	#ax3 = fig.add_subplot(2,2,3)
	#ax4 = fig.add_subplot(2,2,4)
	if not together:
		fig, ([ax1,ax2],[ax3, ax4]) = plt.subplots(2,2,sharex=True,sharey=True)
		fig.text(0.5, 0.04, 'common X', ha='center')
		fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
		fig.set_figheight(6)
		fig.set_figwidth(6)
		fit_data_sorted = []
		sorts = []
		plot_diversity_data(ax1,fitness_datas,'direct',sort)
		plot_diversity_data(ax2,fitness_datas,'lsystem',sort)
		plot_diversity_data(ax3,fitness_datas,'cppn',sort)
		plot_diversity_data(ax4,fitness_datas,'ce',sort)
		plt.tight_layout()
		#fig.suptitle(sort + " diversity")
	else:
		fig, (ax1) = plt.subplots(1,1,sharex=True,sharey=True)
		plot_diversity_data_together(ax1,fitness_datas,'type')
		handles, labels = ax1.get_legend_handles_labels()
		ax1.legend(handles,labels)
		ax1.set_xlabel('Generations')
		ax1.set_ylabel('Tree edit distance')

def plot_fitness_datas(fitness_datas,sort = None):
	cmap = cm.viridis
	fig, ([ax1,ax2],[ax3, ax4]) = plt.subplots(2,2,sharex=True,sharey=True)
	fig.set_figheight(6)
	fig.set_figwidth(6)
	fit_data_sorted = []
	sorts = []
	plot_encoding_data(ax1,fitness_datas,'direct',sort)
	plot_encoding_data(ax2,fitness_datas,'lsystem',sort)
	#plot_encoding_data(ax3,fitness_datas,'cppn',sort)
	plot_encoding_data(ax1,fitness_datas,'ce',sort)
	#fig1 = plt.figure(figsize=(6,6))
	handles, labels = ax1.get_legend_handles_labels()
	
	#plt.tight_layout()

	fig.legend(handles,labels,loc='lower center', ncol =4)
	#fig.suptitle(sort)
	#plt.tight_layout()
	return




def plot_fitness_datas_together(fitness_datas,sort = None,plot_individual_lines=False):
	cmap = cm.viridis
	fig, (ax1) = plt.subplots(1,1,sharex=True,sharey=True)
	fig.set_figheight(6)
	fig.set_figwidth(6)
	ax = ax1
	sorts = []
	n_generations = 0
	cmap = cm.viridis
	sorts = []
	fit_data_direct = []
	fit_data_lsystem = []
	fit_data_cppn = []
	fit_data_ce = []
	fit_data_sorted = []
	encoding_types = ['direct', 'lsystem', 'cppn', 'ce']
	for data in fitness_datas:
		if data.config['encoding']['type'] in encoding_types:
			value = data.config['encoding']['type']
			if value not in sorts:
				sorts.append(value)
				fit_data_sorted.append([data])
			else:
				a_pos = 0
				for i,val in enumerate(sorts):
					if value == val:
						a_pos = i
						break
				fit_data_sorted[a_pos].append(data)

			for i,val in enumerate(sorts):
				if value == val:
					a_pos = i
					break
			fit_data_sorted[a_pos].append(data)
	#ax.set_title(title, va='bottom')
	labels = []
	axdatas = []
	for i,datas in enumerate(fit_data_sorted):
		color = cmap(i/(len(fit_data_sorted)))
		axdata = []
		for data in datas:
			axdata.append(data.p_100)
			if plot_individual_lines:
				line, = ax.plot(data.p_100,linestyle='-.')
				label = sorts[i]
				if label not in labels:
					labels.append(label)
					line.set_label(label)
		axdatas.append(axdata)
		
	for i,datas in enumerate(axdatas):
		color = cmap(float(i)/3.0)
		x = []
		ys = []
		ysmedian = []
		ymax = []
		y75 = []
		ymin = []
		y25 = []
		conf = []
		conf_0 = []
		conf_1 = []
		nda = np.array(datas)
		nd = nda.transpose()
		count = 0
		for vals in nd:
			ys.append(np.average(vals))
			ysmedian.append(np.median(vals))
			ymax.append(np.percentile(vals, 100))
			ymin.append(np.percentile(vals, 0))
			y75.append(np.percentile(vals, 75))
			y25.append(np.percentile(vals, 25))
			c = st.t.interval(0.95, len(vals)-1, loc=np.mean(vals), scale=st.sem(vals))
			conf.append(c)
			conf_0.append(c[0])
			conf_1.append(c[1])
			x.append(count)
			count +=1
		if len(ys) > 0:
			n_generations = len(ys)
		line, = ax.plot(ys, color = color)
		#line2, = ax.plot(ysmedian, color = color)
		label = sorts[i]
		if label not in labels:
			labels.append(label)
			line.set_label(label)
		#ax.fill_between(x,ys,ysmedian, color = color, alpha = 0.4)
		#ax.plot(ymin,color = color, alpha = 0.5, linestyle= '--')
		ax.plot(ymax,color = color, alpha = 0.5, linestyle= '-.')
		#ax.fill_between(x,ymin,ymax, color = color, alpha = 0.1)
		#ax.fill_between(x,y25,y75, color = color, alpha = 0.3)
		#ax.fill_between(x,y25,y75, color = color, alpha = 0.3)
		#ax.plot(conf, color='black')
		ax.fill_between(x,conf_0, conf_1, color=color, alpha=0.3);
	#ax.legend()
	ax.set_xlim(0, 1000)
	ax.set_ylim(bottom=0)
	#fig1 = plt.figure(figsize=(6,6))
	handles, labels = ax1.get_legend_handles_labels()
	
	#plt.tight_layout()

	#ax.legend(handles,labels)
	ax.set_xlabel("Generations")
	ax.set_ylabel("Fitness")
	#fig.legend(handles, labels, loc='right')
	#fig.suptitle(sort)
	#plt.tight_layout()
	return

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
	fitness_data = load_data(path,filename)
	plot_basic_fitness(fitness_data)

def plot_comparison_fitness(path = None):
	#paths = ['C:/result_temp_final/','D:/results/cppn_final/','D:/results/ce_final/']
	paths = []
	if path is None:
		paths.append('C:/result_temp_final/')
	else:
		paths.append(path)
	fitness_datas = load_datas(paths)
	plot_fitness_datas_together(fitness_datas,sort=None,plot_individual_lines=True)
	plt.show()
	

def analyze_sort():
	#directory = 'results/cppn_final/'
	directory = ['C:/result_temp_final/']
	fitness_datas = load_datas(directory)
	
	plot_fitness_datas(fitness_datas,sort=sort)
	#plot_diversity_datas(fitness_datas,sort=sort)
	sort = 'mutation_prob'
	plot_fitness_datas(fitness_datas,sort=sort)
	#plot_diversity_datas(fitness_datas,sort=sort)
	plt.show()
	

def diversity_plot():
	paths = ['C:/result_temp_final/','D:/results/cppn_final/','D:/results/ce_final/' ]
	fitness_datas = load_datas(paths)
	plot_diversity_datas(fitness_datas,sort=None, together = True)

	#plot_fitness_datas_together(fitness_datas,sort=sort)
	plt.show()

def diversity_area_plot():
	displayDivsxy()


def plot_sweep_diversity():
	paths = ['C:/results_sweep/','D:/results/cppn_sweep/','D:/results/ce_sweep/']
	fitness_datas = load_datas(paths)
	fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,True,True)
	sort = 'mutation_prob'
	#fig.text(0.5, 0.04, 'Generations', ha='center')
	#fig.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical')
	x_limit = 500
	y_limit = 2500
	plot_diversity_data(ax1,fitness_datas,'direct',sort, x_limit = x_limit, y_limit = y_limit)
	plot_diversity_data(ax2,fitness_datas,'lsystem',sort,x_limit = x_limit, y_limit = y_limit)
	plot_diversity_data(ax3,fitness_datas,'cppn',sort, x_limit = x_limit, y_limit = y_limit)
	plot_diversity_data(ax4,fitness_datas,'ce',sort, x_limit = x_limit, y_limit = y_limit)
	sort = 'morphmutation_prob'
	plt.tight_layout();
	fig2,(axes) = plt.subplots(2,2,True,True)
	#fig2.text(0.5, 0.04, 'Generations', ha='center')
	#fig2.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical')
	plot_diversity_data(axes[0][0],fitness_datas,'direct',sort, x_limit = x_limit, y_limit = y_limit)
	plot_diversity_data(axes[0][1],fitness_datas,'lsystem',sort,x_limit = x_limit, y_limit = y_limit)
	plot_diversity_data(axes[1][0],fitness_datas,'cppn',sort, x_limit = x_limit, y_limit = y_limit)
	plot_diversity_data(axes[1][1],fitness_datas,'ce',sort, x_limit = x_limit, y_limit = y_limit)
	plt.tight_layout();
	#handles, labels = ax1.get_legend_handles_labels()
	#fig.legend(handles, labels, loc='lower center', ncol=4)
	
	plt.show()
	#fitness_datas,sort = None, together = False

def plot_sweep():
	paths = ['C:/results_sweep/','D:/results/cppn_sweep/','D:/results/ce_sweep/']
	fitness_datas = load_datas(paths)
	fig,([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,True,True)
	#fig,(ax1) = plt.subplots(1,1,True,True)
	sort = 'mutation_prob'
	#fig.text(0.5, 0.04, 'Generations', ha='center')
	#fig.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical')
	x_limit = 500
	y_limit = 50
	plot_encoding_data(ax1,fitness_datas,'direct',sort, x_limit = x_limit, y_limit = y_limit,plotIndividualLines = True)
	plot_encoding_data(ax2,fitness_datas,'lsystem',sort,x_limit = x_limit, y_limit = y_limit)
	plot_encoding_data(ax3,fitness_datas,'cppn',sort, x_limit = x_limit, y_limit = y_limit)
	plot_encoding_data(ax4,fitness_datas,'ce',sort, x_limit = x_limit, y_limit = y_limit)
	sort = 'morphmutation_prob'
	plt.tight_layout();
	fig2,(axes) = plt.subplots(2,2,True,True)
	#fig2.text(0.5, 0.04, 'Generations', ha='center')
	#fig2.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical')
	plot_encoding_data(axes[0][0],fitness_datas,'direct',sort, x_limit = x_limit, y_limit = y_limit)
	plot_encoding_data(axes[0][1],fitness_datas,'lsystem',sort,x_limit = x_limit, y_limit = y_limit)
	plot_encoding_data(axes[1][0],fitness_datas,'cppn',sort, x_limit = x_limit, y_limit = y_limit)
	plot_encoding_data(axes[1][1],fitness_datas,'ce',sort, x_limit = x_limit, y_limit = y_limit)
	plt.tight_layout();
	#handles, labels = ax1.get_legend_handles_labels()
	#fig.legend(handles, labels, loc='lower center', ncol=4)
	
	plt.show()
	#fitness_datas,sort = None, together = False

if __name__ == "__main__":	
	# add a path where you saved your files, make sure to end with '/'
	path = "e.g. D:/your_file_folder/"
	# add the filename of your data (the name of the pickled FitnessData object)
	filename="s_"
	# simple function to plot the progression of a single evolutionary run
	plot_fitness(path,filename);	
	plt.show()

	# the following lines of code are quick hacks that were used for last-minute plotting, they might be a bit lacking in user-friendliness
	#plt.rcParams.update({'font.size': 16})
	#plot_sweep()
	#plot_sweep_diversity()
	#plot_comparison_fitness()
	#diversity_plot()
	#analyze_sort()
	#diversity_area_plot() # todo

