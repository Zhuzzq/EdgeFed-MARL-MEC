# -*- coding: UTF-8 -*-
import numpy as np

from gym.spaces import Space
from gym import logger


class OneHot(Space):
    '''
    An n-dimensional onehot space. 

    '''

    def __init__(self, n):
        self.n = n
        super(OneHot, self).__init__((self.n,), np.int8)

    def sample(self):
        vec = [0]*self.n
        vec[np.random.randint(n)] = 1
        return vec

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return (sum(x == 0) == self.n-1 & sum(x == 1) == 1)

    def __repr__(self):
        return "OneHot({})".format(self.n)

    def __eq__(self, other):
        return isinstance(other, OneHot) and self.n == other.n
