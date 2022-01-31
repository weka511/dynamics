import numpy as np  # Import NumPy
import Rossler
from scipy.integrate import odeint  # Import odeint from scipy.integrate package

#Initial condition for the shortest periodic orbit of the Rossler system:
po1 = np.array([9.2690828474963798e+00,
                0.0e+00,
                2.5815927750254137e+00], float)
period = 5.8810885346818402e+00  # Period of po1
#Initial condition for the Jacobian is identity:
Jacobian0 = None  # COMPLETE THIS LINE. HINT: Use np.identity(DIMENSION)
#Initial condition for Jacobian integral is a d+d^2 dimensional matrix
#formed by concatenation of initial condition for state space and the
#Jacobian:
sspJacobian0 = np.zeros(3 + 3 ** 2)  # Initiate
sspJacobian0[0:3] = po1  # First 3 elemenets
sspJacobian0[3:] = np.reshape(Jacobian0, 9)  # Remaining 9 elements
tInitial = 0  # Initial time
tFinal = period  # Final time
Nt = 500  # Number of time points to be used in the integration

tArray = np.linspace(tInitial, tFinal, Nt)  # Time array for solution

sspJacobianSolution = odeint(Rossler.JacobianVelocity, sspJacobian0, tArray)

xt = sspJacobianSolution[:, 0]  # Read x(t)
yt = sspJacobianSolution[:, 1]  # Read y(t)
zt = sspJacobianSolution[:, 2]  # Read z(t)

#Read the Jacobian for the periodic orbit:
Jacobian = sspJacobianSolution[-1, 3:].reshape((3, 3))
#We used -1 index for referring the final element of the solution array
#and 3: for referring 9 elements corresponding to those of Jacobian
#reshaping this vector into a 3x3 matrix gives us the Jacobian for the
#periodic orbit

#Find eigenvalues and eigenvectors of the Jacobian:
eigenValues, eigenVectors = None  # COMPLETE THIS LINE.
                                  # HINT: Use np.linalg.eig()
#Print them for submission:
print(eigenValues)
#Read eigenvectors into 3 vectors:
v1 = np.real(eigenVectors[:, 0])
v2 = np.real(eigenVectors[:, 1])
v3 = np.real(eigenVectors[:, 2])

#Import plotting functions:
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure()  # Create a figure instance
ax = fig.gca(projection='3d')  # Get current axes in 3D projection
ax.plot(xt, yt, zt)  # Plot the solution
#Mark our initial point for the periodic orbit:
ax.scatter([po1[0]], [po1[1]], [po1[2]], color="c", s=20)

#We will draw vectors v1 and v2 as arrows to see that the solution starts
#on the plane spanned by them. Following code is adapted from:
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
scaleFactor = 3.0
v1scaled = scaleFactor * v1  # Scale v1 for visibility
v2scaled = scaleFactor * v2  # Scale v2 for visibility
v3scaled = scaleFactor * v3  # Scale v3 for visibility
v1arrow = Arrow3D([po1[0], po1[0] + v1scaled[0]],
                  [po1[1], po1[1] + v1scaled[1]],
                  [po1[2], po1[2] + v1scaled[2]],
                   mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
v2arrow = Arrow3D([po1[0], po1[0] + v2scaled[0]],
                  [po1[1], po1[1] + v2scaled[1]],
                  [po1[2], po1[2] + v2scaled[2]],
                   mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
v3arrow = Arrow3D([po1[0], po1[0] + v3scaled[0]],
                  [po1[1], po1[1] + v3scaled[1]],
                  [po1[2], po1[2] + v3scaled[2]],
                   mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
ax.add_artist(v1arrow)
ax.add_artist(v2arrow)
ax.add_artist(v3arrow)

ax.set_xlabel('x')  # Set x label
ax.set_ylabel('y')  # Set y label
ax.set_zlabel('z')  # Set z label

ax.view_init(25, -120)
plt.show()  # Show the figure