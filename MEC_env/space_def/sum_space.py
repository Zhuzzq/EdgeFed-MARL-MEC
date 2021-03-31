# -*- coding: UTF-8 -*-
import numpy as np

from gym.spaces import Space
from gym import logger


class SumOne(Space):
    '''
    An n-dimensional sum-1 space. 

    '''

    def __init__(self, n):
        self.n = n
        super(SumOne, self).__init__((self.n,), np.int8)

    def sample(self):
        vec = np.random.rand(self.n)
        vec = vec/sum(vec)
        return vec

    def contains(self, x):
        return (all(x >= 0) & sum(x) <= 1)

    def __repr__(self):
        return "SumOne({})".format(self.n)

    def __eq__(self, other):
        return isinstance(other, SumOne) and self.n == other.n
