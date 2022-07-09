#!/usr/bin/env python

"""
Default Genome implementation based on numpy array with range [0, 1]
"""

import numpy as np


class Genome(object):
    """Default implementation of a genome factory"""
    def __init__(self, size, sigma):
        self._size = size
        self.sigma = sigma

    def rand(self):
        """Generate a random valid genome"""
        return np.random.random(self._size)

    def variation(self, genome):
        """Randomly perturb the given genome"""
        # Apply variation
        mutated = np.random.normal(genome, self.sigma)
        # Ensure that genome is always in range [0, 1]
        clipped = np.clip(mutated, 0, 1)
        return clipped
