'''Full tent map (Chapter 14 Example 14.8, 14.10)'''

# falls apart after 50 iterations, regardless of whether multiply in side abs or outside
from numpy import abs

def tent(gamma):
    '''
    tent map: one iteration
    '''
    return 1 - 2*abs(gamma-0.5)

def symbol(gamma):
    return 0 if gamma<0.5 else 1

if __name__ == '__main__':
    gamma = 0.8
    for i in range(128):
        print (i,gamma,symbol(gamma))
        gamma = tent(gamma)
