import neat
import os

class CPPN_genome(neat.DefaultGenome):
	def __init__(self,key):
		super().__init__(key)
	def configure_new(self, genome_config):
		super().configure_new(genome_config)
	def configure_crossover(self, genome1, genome2, config):
		super().configure_crossover(genome1, genome2, config)
	def mutate(self,config):
		super().mutate(config)


class CPPN:
	"""The CPPN is extracted from the neat library."""
	def __init__(self, n_inputs,n_outputs,t_config=None):
		# using a simple helper class to manage genomes
		self.genome = CPPN_genome(1)
		local_dir = os.path.dirname(__file__)
		config_path = os.path.join(local_dir, 'config')
		
		self.config = neat.Config(CPPN_genome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
		
		self.config.genome_config.num_inputs = n_inputs
		self.config.genome_config.num_outputs = n_outputs
		self.config.genome_config.input_keys = []
		self.config.genome_config.output_keys = []


		for i in range(n_inputs):
			self.config.genome_config.input_keys.append(-(i+1))
		for i in range(n_outputs):
			self.config.genome_config.output_keys.append((i))

		self.genome.configure_new(self.config.genome_config)
		self.phenotype = self.getPhenotype()
		if (t_config is not None):
			# topological parameters
			self.config.genome_config.conn_add_prob = float(t_config['ea']['morphmutation_prob'])
			self.config.genome_config.conn_delete_prob = float(t_config['ea']['morphmutation_prob'])
			self.config.genome_config.node_add_prob = float(t_config['ea']['morphmutation_prob'])
			self.config.genome_config.node_delete_prob = float(t_config['ea']['morphmutation_prob'])
			
			self.config.genome_config.activation_mutate_rate = float(t_config['ea']['mutation_prob'])
			self.config.genome_config.weight_mutate_power = float(t_config['ea']['mutation_sigma'])
			self.config.genome_config.weight_replace_rate = float(t_config['ea']['mutation_prob'])
			self.config.genome_config.weight_mutate_rate = float(t_config['ea']['mutation_prob'])
			# bias 			
			self.config.genome_config.bias_replace_rate = float(t_config['ea']['morphmutation_prob'])
			self.config.genome_config.bias_mutate_rate = float(t_config['ea']['mutation_prob'])
			self.config.genome_config.bias_mutate_power = float(t_config['ea']['mutation_sigma'])

			# response
			self.config.genome_config.response_replace_rate = float(t_config['ea']['morphmutation_prob'])
			self.config.genome_config.response_mutate_rate = float(t_config['ea']['mutation_prob'])
			self.config.genome_config.response_mutate_power = float(t_config['ea']['mutation_sigma'])

	def update(self,input):
		return(self.phenotype.activate(input))

	def mutate(self):
		self.genome.mutate(self.config.genome_config)

	def crossover(self,other):
		self.genome.configure_crossover(self.genome,other.genome,self.config.genome_config)

	def getPhenotype(self):
		return (neat.nn.FeedForwardNetwork.create(self.genome, self.config))

	def display(self,ax):
		# Helper to display network. ax is a matplotlib sub_plot
		for i,neuron in enumerate(self.phenotype.input_nodes):
			ax.scatter([i-(0.5*len(self.phenotype.input_nodes))],[1])
		for i,neuron in enumerate(self.phenotype.output_nodes):
			ax.scatter([0],[-1])
		for i,neuron in enumerate(self.phenotype.node_evals):
			ax.scatter([i],[0])