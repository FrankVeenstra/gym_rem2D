#!/usr/bin/env python

"""
MAP-Elites implementation in Python
"""

from .genome import Genome
from .mymap import Map
import numpy as np


class Initializer(object):
    def __init__(self, genome, evaluator, storage_map):
        self._gen = genome
        self._eval = evaluator
        self._map = storage_map

    def __call__(self, num_init):
        # Initialize random starting genomes
        initial = [self._gen.rand() for _ in range(num_init)]
        # Evaluate fitness and behavior
        entries = self._eval(initial)
        # Add entries to map
        for entry in entries:
            self._map[entry.behavior] = entry
        # Return private MapElites class for further work
        return _MapElites(self._gen, self._eval, self._map)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['random_state'] = np.random.get_state()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        np.random.set_state(state['random_state'])


class _MapElites(object):
    def __init__(self, genome, evaluator, storage_map):
        self._gen = genome
        self._eval = evaluator
        self._map = storage_map
        self._batches = 0

    def batch(self, size=100):
        """Compute a new batch"""
        # Random selection
        entries = self._map.entries
        selection = [entries[i] for i in np.random.randint(0, len(entries),
                                                           size=size)]
        # Apply variation operator to selection
        to_eval = [self._gen.variation(s.genome) for s in selection]
        # Evaluate selection
        entries = self._eval(to_eval)
        for entry in entries:
            self._map[entry.behavior] = entry
        # Bookkeeping
        self._batches += 1

    def __getstate__(self):
        state = self.__dict__.copy()
        state['random_state'] = np.random.get_state()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        np.random.set_state(state['random_state'])

    @property
    def batches(self):
        """Number of batches performed"""
        return self._batches

    @property
    def map(self):
        """Get the underlying map"""
        return self._map

    @property
    def evaluator(self):
        """Get the evaluator used by this Map"""
        return self._eval

    @property
    def genome(self):
        """Get the genome used by this Map"""
        return self._gen


# To aid users we rename the Initializer class to 'MapElites' so that code
# looks better with correct names, however, the naming in this file better
# represents the actual function of each class
MapElites = Initializer

__all__ = [MapElites, Genome, Map]
