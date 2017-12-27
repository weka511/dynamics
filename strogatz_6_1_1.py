import numpy as np, matplotlib.pyplot as plt,math

from scipy.integrate import odeint
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def f(x,t):
    return [x[0]+math.exp(-x[1]),-x[1]]

t = np.linspace(0, 10, 101)
cs = ['r','b','g','m','c','y']
i=0
Y, X = np.mgrid[-3:3, -3:3]
U=X+np.exp(-Y)
V=-Y
plt.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)

for xy0 in [[-2,1],[-0.5,1],[-1.1,1],[-2,-2],[-2,-1],[-2,-3]]:
    xy = odeint(f, xy0, t)
    plt.plot(xy[:,0],xy[:,1],c=cs[i%len(cs)],label='({0},{1})'.format(xy0[0],xy0[1]))
    plt.xlim(-20,20)
    plt.ylim(-3,3)
    plt.title('Example 6.1.1')
    i+=1
plt.legend(loc='best')    
plt.show()