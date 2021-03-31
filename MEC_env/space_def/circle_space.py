# -*- coding: UTF-8 -*-
import numpy as np

from gym.spaces import Space
from gym import logger


class Circle(Space):
    def __init__(self, r):
        assert r >= 0
        self.r = r
        super(Circle, self).__init__((), np.float)

    def sample(self):
        x = self.r*2*(np.random.rand()-0.5)
        y = np.sqrt(self.r**2-x**2)*2*(np.random.rand()-0.5)
        return (x, y)

    def contains(self, x):
        return (x[0]**2+x[1]**2 <= self.r**2)

    def __repr__(self):
        return "Circle(%f)" % self.r

    def __eq__(self, other):
        return isinstance(other, Circle) and self.r == other.r


class Discrete_Circle(Space):
    def __init__(self, r):
        assert r >= 0
        self.r = r
        super(Discrete_Circle, self).__init__((), np.float)

    def sample(self):
        x = np.random.randint(-self.r, self.r+1)
        y = np.random.randint(-np.sqrt(self.r**2-x**2),
                              np.sqrt(self.r**2-x**2)+1)
        return (x, y)

    def contains(self, x):
        return (x[0]**2+x[1]**2 <= self.r**2)

    def __repr__(self):
        return "Discrete_Circle(%f)" % self.r

    def __eq__(self, other):
        return isinstance(other, Discrete_Circle) and self.r == other.r
