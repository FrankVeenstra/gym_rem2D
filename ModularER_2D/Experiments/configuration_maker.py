import configparser
import os
# import itertools product TODO for iterating through all parameters automatically

mutation_rates = [0.01]
morphology_mutation_rates = [0.01]
mutation_sigmas = [0.1]
encodings = ['lsystem']

def create(experiment_nr = 0,mr = 0.01,mmr = 0.01,ms = 0.1,enc = 'lsystem',dir=''):
	config = configparser.ConfigParser()
	config['experiment'] = {}
	config['experiment']['checkpoint_frequency'] = '10'
	config['experiment']['save_elite'] = '1'
	config['experiment']['experiment_number'] = str(experiment_nr)
	config['experiment']['directory'] = dir

	config['ea'] = {}
	# total number of evaluations: note that generations is calculated as 'n_evaluations' / 'batch_size'
	config['ea']['n_evaluations'] = '10000'
	# Number of individuals evaluated per generation 
	config['ea']['batch_size'] = '100'
	# probability for controller mutations
	config['ea']['mutation_prob'] = str(mr)
	# probability for morphology mutations
	config['ea']['morphmutation_prob'] = str(mmr)
	# sigma value for both types of mutations
	config['ea']['mutation_sigma'] = str(ms)
	# running in headless mode 
	config['ea']['headless'] = '1'
	# showing the best individual after each run (note, box2D time-out can lead to frozen window)
	config['ea']['show_best'] = '0'
	# In case you simply want to load the best individuals
	config['ea']['load_best'] = '0'
	# number of dedicated CPU cores for the experiments
	config['ea']['n_cores'] = '6'
	# placeholder, not implemented in this version
	#config['ea']['crossover_prob']
	config['ea']['interval'] = '5'


	config['morphology'] = {}
	# A robot can be composed of up to 'max_size' modules
	config['morphology']['max_size'] = '40'
	# The maximum depth of the tree blueprint
	config['morphology']['max_depth'] = '7'
	config['morphology']['m_rectangle'] = '4'
	config['morphology']['m_circular'] = '4'

	config['evaluation'] = {}
	# The speed at which the wall of death moves forward
	config['evaluation']['wod_speed'] = '2'

	config['encoding'] = {}
	config['encoding']['type'] = enc	
	config['control'] = {}
	config['control']['type'] = 'wave'

	config['visualization'] = {}
	config['visualization']['v_tree'] = '0'
	config['visualization']['v_progression'] = '0'
	config['visualization']['v_debug'] = '0'
	return config

def save_config(config):
	print("Saving config in : ", config['experiment']['directory'])
	with open(config['experiment']['directory']+str(config['experiment']['experiment_number'])+'.cfg', 'w') as configfile:
		config.write(configfile)

if __name__ == "__main__":
	directory = 'experiments'
	if not os.path.exists(directory):
	    os.makedirs(directory)
	nr = 0
	n_duplicates = 1
	for enc in encodings:
		for mr in mutation_rates:
			for mmr in morphology_mutation_rates:
				for ms in mutation_sigmas:
					for i in range(n_duplicates):
						config = create(nr, mr,mmr,ms, enc, directory +'/')
						save_config(config)
						nr+=1
