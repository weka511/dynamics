#!/usr/bin/env python

import matplotlib.pyplot as plt

if __name__=='__main__':
    R0           = 2.9
    R1           = 3.57
    N            = 2000
    M            = 2000
    L            = 10000
    epsilon      = 0.00000001
    x_init       = 0.5
    step         = (R1-R0)/N
    bifurcations = []
    cycle_length0 = 1
    cycle_length  = None
    for i in range(N):
        r  = R0+i*step
        x  = x_init
        rs = []
        xs = []
        for j in range(L):
            x=r*x*(1-x)
        x0 = x
        for j in range(M):
            x=r*x*(1-x)
            rs.append(r)
            xs.append(x)
            if abs(x-x0)<epsilon:
                cycle_length = j+1
                break
        if cycle_length != cycle_length0:
            bifurcations.append((r,cycle_length))
            cycle_length0 = cycle_length
        plt.scatter(rs,xs,s=1,edgecolors='b',c='b')

    print (bifurcations)
    for i in range(len(bifurcations)-2):
        a,_=bifurcations[i]
        b,_=bifurcations[i+1]
        c,_=bifurcations[i+2]
        print ((a-b)/(b-c))
    plt.show()
