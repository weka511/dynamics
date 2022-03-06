###################################################
# This file contains all the related functions to
# investigate the escape rate in Logistic map.
# please complete the experiment part
###################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

class Logistic:
    def __init__(self, A):
        self.A = A

    def oneIter(self, x):
        return self.A * (1.0 - x) * x

    def multiIters(self, x, n):
        y = x
        tmpx = x
        for i in range(n):
            tmpx = self.oneIter(tmpx)
            y.append(tmpx)
            
        return y

    def doesEscape(self, x, n):
        """
        determine whether the mapping sequence is escaping or not
        parameters: 
              x  initial point
              n  number of iteration
        return :
             a vector indicating whether the corresponding iteration
             has escapted rigion [0, 1] or not. '1' indicates escapted.
        """
        tmpx = x
        escape = np.ones(n)
        for i in range(n):
            tmpx = self.oneIter(tmpx)
            if tmpx <= 1 and tmpx >=0:
                escape[i] = 0
            else: 
                break

        return escape


if __name__ == '__main__':
    """
    experiment
    """
    # start with a large number of initial conditions and iterate
    # for a certain number of steps, then find out the escape ratio.
    #  repeat this process for server different iteration steps.
    
    
