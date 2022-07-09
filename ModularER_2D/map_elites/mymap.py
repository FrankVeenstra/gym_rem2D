#!/usr/bin/env python

from typing import Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy



class Map():

    class Cell():

        def __init__(self):
            self.individual = None

        def set_cell(self,individual):
            self.individual = individual



    def __init__(self,dims):
        
        if type(dims) == tuple:
            self._dims = tuple(dims)
            self._map = [[None] * dims[0] for _ in range(dims[1])]
        else:
            self._dims = dims
            self._map = [None]*dims
            self._timeseries = []

    @property
    def shape(self):
        return self._dims

    def __len__(self):
        """Length of filled entries"""
        entries = self.get_elites()
        return len(entries)

    def eval_individual(self,individual,TREE_DEPTH=1,behaviour=None):
        """
        Find the phenotype of the individual and determine if it should be added to cell.
        The phenotype in this particular case is the number of leaf nodes in the tree structure.
        """
        if type(self._dims) is tuple:
            index1 = int(behaviour[0]*(self._dims[0]-1))
            index2 = int(behaviour[1]*(self._dims[1]-1))

            if self._map[index1][index2] is None:
                self._map[index1][index2] = self.Cell()
                self._map[index1][index2].set_cell(individual)
            elif self._map[index1][index2].individual.fitness < individual.fitness:
                self._map[index1][index2].set_cell(individual)
        else:
            phenotype = individual.genome.create(TREE_DEPTH).get_leaves(individual.genome.expressed_nodes)

            if self._map[phenotype] is None:
                self._map[phenotype] = self.Cell()
                self._map[phenotype].set_cell(individual)
            elif self._map[phenotype].individual.fitness < individual.fitness:
                self._map[phenotype].set_cell(individual)

        return self._map

    def random_select(self):
        """
        Keep looking for a random cell with an elite until one is found or until iter limit is reached.
        """
        if type(self._dims) is tuple:
            index1 = 0
            index2 = 0
            iter = 0
            while True:
                if iter == 10000:
                    print("Could not find a non-empty cell after 10000 iterations (map might be empty)")
                    break
                iter+=1
                rand1 = np.random.random()
                rand2 = np.random.random()
                index1 = int(rand1*(self._dims[0]-1))
                index2 = int(rand2*(self._dims[1]-1))
                if self._map[index1][index2] is None:
                    continue
                

                return self._map[index1][index2].individual
            
        else:
            iter = 0
            while True:
                if iter == 10000:
                    print("Could not find a non-empty cell after 10000 iterations (map might be empty)")
                    break
                iter+=1
                number = np.random.randint(0,len(self._map))
                if self._map[number] is None:
                    continue
                
                return self._map[number].individual

    def store_timeseries(self):
        self._timeseries.append(copy.deepcopy(self._map))

    def get_elites(self):
        """
        Get all elites in map
        """
        elites = []

        if type(self._dims) is tuple:
            for C1 in range(0,self._dims[0]-1):
                for C2 in range(0,self._dims[1]-1):
                    if self._map[C1][C2] is None: continue
                    elites.append(self._map[C1][C2].individual)
        else:
            for C in self._map:
                if C is None: continue
                elites.append(C.individual)
        
        return elites

    def get_map(self):
        """
        Get a pointer to the map
        """
        return self._map

    def get_best_elite(self):
        best_elite = None
        elites = self.get_elites()
        for e in elites:
            if best_elite is None:
                best_elite = e
            elif e.fitness > best_elite.fitness:
                best_elite = e
        return best_elite

    def get_5_best_elites(self):
        elites = self.get_elites()
        elites.sort(key=lambda x: x.fitness, reverse=True)

        return elites[:5]

    def plot_heat(self):
        if type(self._dims) is tuple:
            heatmap = np.zeros(self._dims)
            for row,i in enumerate(self._map):
                for col,j in enumerate(i):
                    if j is None:
                        heatmap[row][col] = 0
                        continue
                    heatmap[row][col] = j.individual.fitness
            df = pd.DataFrame(heatmap)
            #result = df.pivot(index="Yrows",columns="Xcols",values="Change")
            fig, ax = plt.subplots()
            #title = "Heatmap for rastigrin"
            #plt.title(title, fontsize=18)
            sns.heatmap(df, fmt="", linewidths=0, ax=ax)#, cbar_kws={'label': 'fitness'})
            #ax.set_ylabel('b1')
            #ax.set_xlabel('b2')
            #plt.show()
            plt.savefig("rastigrin_25000eval.pdf")

        else:
            timeseries_dims = tuple((len(self._timeseries), self._dims))
            heatmap = np.zeros(timeseries_dims)
            for row,i in enumerate(self._timeseries):
                for col,j in enumerate(i):
                    if j is None:
                        heatmap[row][col] = 0
                        continue
                    heatmap[row][col] = j.individual.fitness
            df = pd.DataFrame(heatmap)
            fig, ax = plt.subplots(figsize=(12,7))
            sns.heatmap(df, fmt="", linewidths=0, ax=ax)
            plt.show()

    @property      
    def coverage(self):
        """Number of filled cells divided by total number of cells"""
        if type(self._dims) is tuple:
            return len(self)/np.product(self.shape)
        else:
            return len(self)/self.shape
            
    
    @property
    def precision(self, max_fitness = None):

        ents = self.get_elites()

        if max_fitness is not None:
            ssum = sum([e.fitness / max_fitness for e in ents])
        else:
            ssum = sum([e.fitness for e in ents])
        return ssum / float(len(ents))

    @property
    def reliability(self, max_fitness = None):

        ents = self.get_elites()

        if max_fitness is not None:
            ssum = sum([e.fitness / max_fitness for e in ents])
        else:
            ssum = sum([e.fitness for e in ents])
        return ssum / np.product(self.shape)

    @property
    def QD_score(self):

        ents = self.get_elites()

        QD = 0
        for e in ents:
            QD+=e.fitness

        return QD
        
        

        

        
    