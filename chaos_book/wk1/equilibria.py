#!/usr/bin/env python

from matplotlib.pyplot import figure, show
from math import sqrt
import numpy  as np
from scipy.integrate   import odeint
from scipy.linalg      import eig
from Rossler           import Velocity, StabilityMatrix

def get_equilibrium(a = 0.2,
                    b = 0.2,
                    c = 5.7):
    srqt1 = sqrt(1 - 4 * a*b/(c*c))
    mult1 = 0.5 + 0.5*srqt1
    mult2 = 0.5 - 0.5*srqt1
    pt1   = (mult1*c, -mult1*c/a, mult1*c/a)
    pt2   = (mult2*c, -mult2*c/a, mult2*c/a)
    return (pt1,pt2)

def subset(s,i,step=1):
    m,_ = s.shape
    return [s[j,i] for j in range(0,m,step)]

def plot_solution(s,
                  c    = 'xkcd:blue',
                  step = 1):
    fig       = figure()
    ax        = fig.add_subplot(1,1,1,projection='3d')
    xt        = subset(s,0,step=step)
    yt        = subset(s,1,step=step)
    zt        = subset(s,2,step=step)
    A         = StabilityMatrix(s[0,:])
    floquet,_ = eig(A)
    ax.plot(xt,yt,zt,
            c  = c,
            ms = 1)
    ax.set_title(f'{floquet}')

u1,u2  = get_equilibrium()

plot_solution(odeint(Velocity, u1,  np.linspace(0, 600.0, 10000)))
plot_solution(odeint(Velocity, u2,  np.linspace(0, 600.0, 10000)),
              c    = 'xkcd:red',
              step = 100)


show()
