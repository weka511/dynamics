'''Full tent map (Chapter 14 Example 14.8, 14.10)'''

from numpy import abs

def tent(x):
    '''
    tent map: one iteration
    '''
    return 1 - 2 * abs(x-0.5)

def tent_iter(x, n):
    '''
    tent map: n iterations
    '''
    y = [x]
    for i in range(n):
        y.append(tent(y[-1]))

    return y

if __name__ == '__main__':
    base = 2**6+1
    for i in range(1, base):
        x = float(i)/base
        y = tent_iter(x, 6)
        print( i, x , y[-1])
