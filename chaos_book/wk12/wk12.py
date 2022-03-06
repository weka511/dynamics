###################################################
# This file contains all the related functions to
# investigate the escape rate in Logistic map using
# cycle expansion.
# 
# The experiments will utilize symbolic calculation
# package "SymPy", but only a little bit. You can
# find some short tutorial online.
# 
# please complete the experimental part
###################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp

class Logistic:
    def __init__(self, A):
        self.A = A

    def oneIter(self, x):
        return self.A * (1.0 - x) * x

    def multiIters(self, x, n):
        y = np.zeros(n+1)
        y[0] = x
        tmpx = x
        for i in range(n):
            tmpx = self.oneIter(tmpx)
            y[i+1]=tmpx
            
        return y

    def df(self, x):
        return self.A*(1-2*x)
    
    def dfn(self, x):
        n = np.size(x)
        multiplier = 1.0
        for i in range(n):
            multiplier *= self.df(x[i])

        return multiplier


if __name__ == '__main__':
    """
    experiment
    """
    case = 4

    if case == 1:
        """
        try to find the periodic orbits up to length 4 
        { {0, 1}, {01}, {001, 011}, {0001, 0011, 0111} }
        and their Floquet multipliers. 
        Hint: scipy.optimize.fsolve or numpy.roots
        """
        A = 6.0
        lm = Logistic(A)
        
        # write your code here

    if case == 2:
        """
        Cycle expand dynamical zeta function.
        Follow formula (23.9) to update dynamical zeta 
        function inside the loop and discard the higher
        orders.
        """
        # Floquet multipliers for the periodic orbits up to length 4
        # { {0, 1}, {01}, {001, 011}, {0001, 0011, 0111} }
        mp = [[None, None], 
              [None],    
              [None, None],
              [None, None, None]
              ]
        lm = Logistic(6.0)
        z = sp.Symbol('z')
        zeta = 1 # dynamical zeta function
        order = 4 # cycle expansion order N
        for i in range(order):
            for j in range(np.size(mp[i])):
                # complete the following line
                zeta = zeta * None
                # remove higher orders 
                zeta = ( zeta.expand() + sp.O(z**(order+1)) ).removeO() 
        print "zeta function at order: ", order 
        print zeta
        
        # for efficicy, we choose to use np.roots() instead sp.solve() to find the zero points
        coe = sp.Poly(zeta, z).coeffs() # get the coefficients => a_n, a_{n-1}, ..., a_0
        zps = np.roots(coe) # find the zeros points of zeta function

        # try to get the leading eigenvalue and escape rate
        # do it here

    if case == 3:
        """
        Cycle expand spectral determinant.
        Follow formula that below (23.13) and formula (23.16) 
        to update spectral determinant.
        """
        # Floquet multipliers for the periodic orbits up to length 4
        # { {0, 1}, {01}, {001, 011}, {0001, 0011, 0111} }
        mp = [[None, None], 
              [None],    
              [None, None],
              [None, None, None]
              ]
        
        lm = Logistic(6.0)
        z = sp.Symbol('z')
        trace = 0 # trace of evolution operatior
        order = 4 # cycle expansion order N
        
        # your code to caculate expansion coefficients of trace formula here
        # use C to denote these coefficients
        
        # get the expansion coefficients of spectral determinant here by formula (20.15)
        # use Q to denote these coefficients 
        
        # find the zeros points of zeta function
        # get the leading eigenvalues
        # print out the escapte rate

    if case == 4:               
        """
        for A = 5.0, expand the spectral determinant to get the escape rate.
        """
        # Floquet multipliers for the periodic orbits up to length 5
        # { {0, 1}, {01}, {001, 011}, {0001, 0011, 0111},
        # {00001, 00011, 00101, 00111, 01011, 01111} }     
        # multipliers for A = 5.0  
        mp = [[None, None], 
              [None],    
              [None, None],
              [None, None, None],
              [None, None, None, None, None, None]
              ] 
