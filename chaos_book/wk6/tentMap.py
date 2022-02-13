import numpy as np

def tent(x):
    """
    tent map: one iteration
    """
    return 1 - 2 * np.abs(x-0.5)

def tent_iter(x, n):
    """
    tent map: n iterations
    """
    y = []
    y.append(x)
    for i in range(n):
        tmp = tent(x)
        y.append(tmp)
        x = tmp

    return y

if __name__ == '__main__':
    base = 2**6+1
    for i in range(1, base):
        x = float(i)/base
        y = tent_iter(x, 6)
        print( i, x , y[-1])
