from numpy             import array, linspace
from scipy.integrate   import odeint
from matplotlib.pyplot import figure, show
from math              import sqrt
from Rossler           import Velocity

def get_equilibrium(a = 0.2,b = 0.2,c = 5.7):
    srqt1 = sqrt(1 - 4 * a*b/(c*c))
    mult1 = 0.5 +0.5*srqt1
    mult2 = 0.5 - 0.5*srqt1
    # pt = (mult1*c, -mult1*c/a, mult1*c/a)
    pt1 = (mult1*c, -mult1*c/a, mult1*c/a)
    pt2 = (mult2*c, -mult1*c/a, mult1*c/a)
    return (pt1,pt2)

u1,u2 = get_equilibrium()
tArray = linspace(0, 250, 10000)
s1 = odeint(Velocity, u1, tArray)
s2 = odeint(Velocity, u2, tArray)

fig    = figure()
ax     = fig.gca(projection='3d')
ax.plot(s1[:,0],s1[:,1],s1[:,2],c='xkcd:blue',markersize=1)
fig    = figure()
ax     = fig.gca(projection='3d')
ax.plot(s2[:,0],s2[:,1],s2[:,2],c='xkcd:red',markersize=1)
show()
