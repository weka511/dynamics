from matplotlib.pyplot    import figure, show
from matplotlib.patches   import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from numpy.linalg         import eig,norm
from numpy                import array, real, imag, linspace
from Rossler              import StabilityMatrix, Velocity
from scipy.integrate      import odeint
from scipy.optimize       import fsolve


class Arrow3D(FancyArrowPatch):
     '''
     We will draw vectors v1 and v2 as arrows to see that the solution starts on the
     plane spanned by them. Following code is adapted from:
     http://stackoverflow.com/questions/11140163/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector
     '''
     def __init__(self, xs, ys, zs, *args, **kwargs):
          FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
          self._verts3d = xs, ys, zs
     def draw(self, renderer):
          xs3d, ys3d, zs3d = self._verts3d
          xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
          self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
          FancyArrowPatch.draw(self, renderer)

eq0 = fsolve(Velocity, array([0, 0, 0], float), args=(0,)) # find the equilibrium of the Rossler system close to the origin
#We input args=(0,) in the arguments of fsolve, this is because the function
#Velocity takes two inputs (ssp, t) so that it is compatible for integration
#using odeint. Here, we send simply t=0 in fsolve to fix its value while
#searching for the equilibrium. The value we send has no importance since the
#velocity function is in fact time independent.

#Evaluate the stability matrix (Rossler.Stability) at eq0:
Aeq0 = StabilityMatrix(eq0)
#Find eigenvalues and eigenvectors of the stability matrix at eq0:
#We are going to use eig function from np.linalg module, which will return
#eigenvalues as a 3-dimensional numpy array as its first output and eigenvectors
#in the columns of 3x3 numpy array as its second output.
#See
#http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
#for a detailed reference.
eigenValues, eigenVectors = eig(Aeq0)
print(eigenValues)

#Let's take the real and imaginary parts of eigenvector corresponding to
#eigenvalue with largest real part:
v1 = real(eigenVectors[:, 0])
v2 = imag(eigenVectors[:, 0])
v1 = v1 / norm(v1) #Normalize these eigenvectors:
v2 = v2/ norm(v2)

#Define the initial condition as a slight perturbation to the eq0 in v1
#direction:
ssp0 = eq0 + 1e-6 * v1
#Now we will integrate this point for a short time to confirm that locally
#solution spirals out in a two dimensional surface spanned by v1 and v2

tInitial    = 0  # Initial time
tFinal      = 50  # Final time
Nt          = 5000  # Number of time points to be used in the integration
tArray      = linspace(tInitial, tFinal, Nt)  # Time array for solution:
sspSolution = odeint(Velocity, ssp0, tArray)  # COMPLETE THIS LINE

fig         = figure()  # Create a figure instance
ax          = fig.gca(projection='3d')  # Get current axes in 3D projection
ax.plot(sspSolution[:, 0], sspSolution[:, 1], sspSolution[:, 2])  # Plot the solution

scaleFactor = 1e-2
v1scaled    = scaleFactor * v1  # Scale v1 for visibility
v2scaled    = scaleFactor * v2  # Scale v2 for visibility

ax.add_artist( Arrow3D([eq0[0], eq0[0] + v1scaled[0]],
                       [eq0[1], eq0[1] + v1scaled[1]],
                       [eq0[2], eq0[2] + v1scaled[2]],
                       mutation_scale = 20,
                       lw             = 2,
                       arrowstyle     = "-|>",
                       color          = "k"))
ax.add_artist(Arrow3D([eq0[0], eq0[0] + v2scaled[0]],
                      [eq0[1], eq0[1] + v2scaled[1]],
                      [eq0[2], eq0[2] + v2scaled[2]],
                      mutation_scale = 20,
                      lw             = 2,
                      arrowstyle     = "-|>",
                      color          = "k"))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(25, -120)
show()
