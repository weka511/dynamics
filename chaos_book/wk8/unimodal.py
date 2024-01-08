#!/usr/bin/env python

import numpy as np

class Unimodal:
    '''
    symbolic dynamics of a unimodal map.
    input parameters:
         returnMap: the return map itself
         C : critical point of this map
    '''
    def __init__(self, returnMap, C):
        self.returnMap = returnMap
        self.C         = C

    def returnMap_iter(self, x0, n):
        '''
        multiple iterations of this map from point x0
        n: the iteration number.
        '''
        x = x0
        future = np.empty((n))
        for i in range(n):
            x = self.returnMap(x)
            future[i] = x

        return future

    def future_symbol(self, x0, n):
        '''
        future itinerary of point x0.
        n : the iteration number.
        '''
        x = x0
        future = np.empty((n))
        for i in range(n):
            x = self.returnMap(x)
            future[i] = int(x > self.C)

        return future
