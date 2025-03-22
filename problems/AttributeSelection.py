# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:14:57 2022

@author: Raquel
"""

# Problem definition and fitness function for all variables in D
from platypus import *
import numpy as np
import config.config as config


class AttributeSelection(Problem):

    def __init__(self, model, nVar = 2, nobjs = 2):
        super(AttributeSelection, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.model = model


    def evaluate(self, solution):
        maskInt = np.array([int(sol[0]) for sol in solution.variables], dtype=int)
        N = maskInt.sum()
        last_sum = maskInt[-config.N_STEPS:].sum()

        solution.objectives[:] = [1000 if (N == 0 or last_sum == 0) else self.model.predict(maskInt.reshape(1, -1))[0], N]

