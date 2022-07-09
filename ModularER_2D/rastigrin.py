#!/usr/bin/env python

from map_elites import Map, Genome


import numpy as np
import datetime
import sys

try:
    # Optionally use tqdm package for visualization of progress
    from tqdm import trange
except ImportError:
    trange = range


class Rastrigin(object):
    """Class which encapsulate fitness evaluation and behavior
    characteristic"""
    def __init__(self, genome):
        self.genome = genome

    @property
    def fitness(self):
        """Calculate fitness of genome.

        Usually fitness will be a float, but it can be anything supporting
        ordering (>, <, >=, <=)"""
        # Since we are using default genome we know that the genome is in range
        # [0, 1), however, for this problem we need it to be [-5, 5]
        scaled_genome = 10.0 * self.genome - 5.0
        # Fitness calculation starts by calculating '10' times the size of the
        # genome
        fitness = 10.0 * scaled_genome.shape[0]
        # Then we iterate the genome and perform the formula below
        for gene in scaled_genome:
            fitness += gene ** 2 - 10. * np.cos(2. * np.pi * gene)
        # Finally we return the calculated fitness value
        return fitness

    def behavior(self):
        """Return the behavior characteristic of the solution.

        Must be array like with range [0, 1]."""
        # For this simple problem we return the first two numbers of the
        # unscaled genome. For more complex cases this might be expensive to
        # calculate. Note also that this function should be called after
        # 'fitness()' if behavior is collected during evaluation
        return self.genome[0:2]

genome = Genome(10,0.1)#generate different genomes
storage = Map((128,128))

for _ in range(100):
    evalauator = Rastrigin(genome.rand())

    storage.eval_individual(evalauator,behaviour=evalauator.behavior())

iter = 0
for _ in trange(25000):
    iter+=1
    offspring = [storage.random_select() for _ in range(100)]
    fitness_values = []

    for i,o in enumerate(offspring):
        offspring[i] = genome.variation(o.genome)
        evaluator = Rastrigin(offspring[i])
        fitness_values.append(evaluator.fitness)
        storage.eval_individual(evaluator,behaviour=evaluator.behavior())

elites = storage.get_elites()
fitness_values = []

for e in elites:
    fitness_values.append(e.fitness)

min = np.min(fitness_values)
max = np.max(fitness_values)
mean = np.mean(fitness_values)
sys.stdout.write("Generation %d evaluated : Min %s, Max %s, Avg %s" % (iter,min,max,mean))
sys.stdout.write("\nTotal map coverage : %s" % (storage.coverage))
sys.stdout.write("\nPrecision for map : %s" % (storage.precision))
sys.stdout.write("\nReliability for map : %s" % (storage.reliability))
sys.stdout.write("\nQD-score : %s" % (storage.QD_score))
storage.plot_heat()
