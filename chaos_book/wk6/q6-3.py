'''Full tent map (Chapter 14 Example 14.8, 14.10)'''

# Without multiple precision, falls apart after 50 iterations, regardless of whether multiply inside abs or outside

from mpmath import mp,nstr

def tent(gamma):
    def abs(x):
        return x if x>=0 else -x
    '''
    tent map: one iteration
    '''
    return mp.mpf(1) - abs(mp.mpf(2)*gamma - mp.mpf(1))

def symbol(gamma):
    return 0 if 2*gamma<1 else 1

if __name__ == '__main__':
    mp.dps = 256

    gamma = (mp.mpf(63)/4095)#mp.mpf(8)/10
    S = []
    for i in range(256):
        print (f'{i} {nstr(gamma,12)} {symbol(gamma)}')
        gamma = tent(gamma)
        S.append(symbol(gamma))

    print (S)
