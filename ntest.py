#!/usr/bin/env python
'''Numba is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code. '''
from numba import jit

@jit
def f(x, y):
    # A somewhat trivial example
    return x + y

print (f(2,3))
