import REM2D_main as r2d
import numpy as np

def evaluate(individual, evaluation_steps= 10000, headless=True, render_interval=1, environment_length=100, tree_depth = None, controller = None):
	env = r2d.getEnv()
	if tree_depth is None:
		try:
		   tree_depth = individual.tree_depth
		except:
			raise Exception("Tree depth not defined in evaluation")
	tree = individual.genome.create(tree_depth)
	env.seed(4)
	env.reset(tree=tree, module_list=individual.genome.moduleList)

	fitness = 0
	for i in range(evaluation_steps):
		if i % render_interval == 0:
			if not headless:
				env.render()

		# A list of actions should be returned ideally. 
		# Right now, the tree contains a contoller which is updated.  
		# TODO: 
		action = np.ones_like(env.action_space.sample())	
		observation, reward, done, info  = env.step(action)
		
		if reward< -10: 
			break
		elif reward > environment_length:
			# add a little bit on top of the regular fitness. 
			reward += (evaluation_steps-i)/evaluation_steps
			fitness = reward
			break
		if reward > 0:
			fitness = reward
	return fitness


if __name__=="__main__":
	# how many random individual to create
	n_iterations = 200 
	for i in range(n_iterations):
		individual = r2d.Individual.random(encoding = 'lsystem') # options: direct, cppn, ce, lsystem
		evaluate(individual, headless = False, controller = None)
	
