import numpy as np  # Import NumPy
import Rossler  # Import Rossler module
from scipy.integrate import odeint  # Import odeint from scipy.integrate
                                    # package
from scipy.optimize import fsolve  # Import fsolve from scipy.optimize for
                                   # numerical root finding

#Numerically find the equilibrium of the Rossler system close to the
#origin:
eq0 = fsolve(Rossler.Velocity, np.array([0, 0, 0], float), args=(0,))
#We input args=(0,) in the arguments of fsolve, this is because the function
#Velocity takes two inputs (ssp, t) so that it is compatible for integration
#using odeint. Here, we send simply t=0 in fsolve to fix its value while
#searching for the equilibrium. The value we send has no importance since the
#velocity function is in fact time independent.

#Evaluate the stability matrix (Rossler.Stability) at eq0:
Aeq0 = None  # COMPLETE THIS LINE
#Find eigenvalues and eigenvectors of the stability matrix at eq0:
#We are going to use eig function from np.linalg module, which will return
#eigenvalues as a 3-dimensional numpy array as its first output and eigenvectors
#in the columns of 3x3 numpy array as its second output.
#See
#http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
#for a detailed reference.
eigenValues, eigenVectors = np.linalg.eig(Aeq0)
print(eigenValues)

#Let's take the real and imaginary parts of eigenvector corresponding to
#eigenvalue with largest real part:
v1 = np.real(eigenVectors[:, 0])
v2 = None  # COMPLETE THIS LINE, HINT: Use np.imag()
#Normalize these eigenvectors:
v1 = v1 / np.linalg.norm(v1)
v2 = None  # COMPLETE THIS LINE
#Define the initial condition as a slight perturbation to the eq0 in v1
#direction:
ssp0 = eq0 + 1e-6 * v1
#Now we will integrate this point for a short time to confirm that locally
#solution spirals out in a two dimensional surface spanned by v1 and v2

tInitial = 0  # Initial time
tFinal = 50  # Final time
Nt = 5000  # Number of time points to be used in the integration
# Time array for solution:
tArray = np.linspace(None, None, None)  # COMPLETE THIS LINE

sspSolution = odeint(None, None, None)  # COMPLETE THIS LINE

xt = sspSolution[:, 0]  # Read x(t)
yt = sspSolution[:, 1]  # Read y(t)
zt = sspSolution[:, 2]  # Read z(t)

#Import plotting functions:
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure()  # Create a figure instance
ax = fig.gca(projection='3d')  # Get current axes in 3D projection
ax.plot(xt, yt, zt)  # Plot the solution
#We will draw vectors v1 and v2 as arrows to see that the solution starts on the
#plane spanned by them. Following code is adapted from:
#http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a
#-3d-cube-a-sphere-and-a-vector


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

#Define the arrow objects:
scaleFactor = 1e-2
v1scaled = scaleFactor * v1  # Scale v1 for visibility
v2scaled = scaleFactor * v2  # Scale v2 for visibility
v1arrow = Arrow3D([eq0[0], eq0[0] + v1scaled[0]],
                  [eq0[1], eq0[1] + v1scaled[1]],
                  [eq0[2], eq0[2] + v1scaled[2]],
                   mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
v2arrow = Arrow3D([eq0[0], eq0[0] + v2scaled[0]],
                  [eq0[1], eq0[1] + v2scaled[1]],
                  [eq0[2], eq0[2] + v2scaled[2]],
                   mutation_scale=20, lw=2, arrowstyle="-|>", color="k")
ax.add_artist(v1arrow)
ax.add_artist(v2arrow)

ax.set_xlabel('x')  # Set x label
ax.set_ylabel('y')  # Set y label
ax.set_zlabel('z')  # Set z label

ax.view_init(25, -120)
plt.show()  # Show the figure