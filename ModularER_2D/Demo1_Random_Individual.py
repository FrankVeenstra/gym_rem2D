import REM2D_main as r2d
import numpy as np

def evaluate(individual, EVALUATION_STEPS= 10000, HEADLESS=True, INTERVAL=100, ENV_LENGTH=100, TREE_DEPTH = None, CONTROLLER = None):
	env = r2d.getEnv()
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

		# Note that the evaluation function is not designed to be integrated with the action space of OpenAI gym. 
		# TODO: 
		action = np.ones_like(env.action_space.sample())	

		observation, reward, done, info  = env.step(action)
		
		if reward< -10:
			break
		elif reward > ENV_LENGTH:
			# add a little bit on top of the regular fitness. 
			reward += (EVALUATION_STEPS-i)/EVALUATION_STEPS
			fitness = reward
			break
		if reward > 0:
			fitness = reward
	return fitness


if __name__=="__main__":
	# Here we simply create and evaluate a few random individuals. 
	for i in range(200):
		individual = r2d.Individual.random(encoding = 'ce')
		evaluate(individual, HEADLESS = False, CONTROLLER = None)
	
