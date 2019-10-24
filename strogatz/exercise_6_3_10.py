import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
    return (x*y,x*x-y)

xs   = np.arange(-1, 1, 0.1)
ys   = np.arange(-1, 1, 0.1)
X, Y = np.meshgrid(xs,ys)
U, V = f(X,Y)

plt.quiver(X,Y,U,V,angles='xy')

plt.show()