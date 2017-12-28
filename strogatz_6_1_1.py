import numpy as np, matplotlib.pyplot as plt

from scipy.integrate import odeint
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def ex(x,y):
    return x+np.exp(-y),-y

def ff(x,t,f=ex):
    u,v=f(x[0],x[1])
    return [u]+[v]

nx, ny = 64, 64
x = np.linspace(-5, 5, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y)
U,V=ex(X,Y)

plt.streamplot(X, Y, U, V, color=U, linewidth=1, cmap=plt.cm.autumn)
t = np.linspace(0, 10, 101)
cs = ['r','b','g','m','c','y']
i=0

for xy0 in [[-2,3],[-0.5,3],[-1.1,3],[-2,-3],[-2,-3],[-2,-3]]:
    xy = odeint(ff, xy0, t)
    plt.plot(xy[:,0],xy[:,1],c=cs[i%len(cs)],label='({0},{1})'.format(xy0[0],xy0[1]),linewidth=3)
    plt.xlim(-5,5)
    plt.ylim(-3,3)
    plt.title('Example 6.1.1')
    i+=1
plt.legend(loc='best')    
plt.show()