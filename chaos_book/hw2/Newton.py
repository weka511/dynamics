import numpy as np  # Import NumPy
from numpy import pi  # Import pi from numpy
from scipy.integrate import odeint  # Import odeint from scipy.integrate
from scipy.optimize import fsolve  # Import fsolve from scipy.optimize
import Rossler  # Import Rossler module
import Poincare  # Import Poincare module

#Set the angle between the Poincare section hyperplane and the x-axis:
thetaPoincare = 0.0

#Define vectors which will be on and orthogonal to the Poincare section
#hyperplane:
e_x = np.array([1, 0, 0], float)  # Unit vector in x-direction
#Template vector to define the Poincare section hyperplane:
sspTemplate = None  # COMPLETE THIS LINE. HINT: See Poincare.py
#Normal to this plane will be equal to template vector rotated pi/2 about
#the z axis:
nTemplate = None  # COMPLETE THIS LINE. HINT: See Poincare.py

#Define the Poincare section hyperplane equation as a Lambda function based on
#the UPoincare from Poincare module, using our new sspTemplate and nTemplate:
UPoincare = lambda ssp: Poincare.UPoincare(ssp, sspTemplate, nTemplate)

#We will first run a long trajectory of the Rossler system by starting
#close to the eq0 in order to include its unstable manifold on the Poincare
#section. Let us start by repeating what we have done in the stability
#exercise and construct this initial condition:
#Numerically find the equilibrium of the Rossler system close to the
#origin:
eq0 = None  # COMPLETE THIS LINE. HINT: See Stability.py
#Evaluate the stability matrix at eq0:
Aeq0 = None  # COMPLETE THIS LINE. HINT: See Stability.py
#Find eigenvalues and eigenvectors of the stability matrix at eq0:
eigenValues, eigenVectors = None  # COMPLETE THIS LINE. HINT: See Stability.py
#Read the real part of the leading eigenvector into the vector v1:
v1 = None  # COMPLETE THIS LINE. HINT: See Stability.py
#Normalize v1:
v1 = None  # COMPLETE THIS LINE. HINT: See Stability.py
#Initial condition as a slight perturbation to the eq0 in v1 direction:
ssp0 = None  # COMPLETE THIS LINE. HINT: See Stability.py

tInitial = 0  # Initial time
tFinal = 1000  # Final time
Nt = 100000  # Number of time points to be used in the integration

# Time array for solution:
tArray = None  # COMPLETE THIS LINE. HINT: See previous exercises
#Integration:
sspSolution = None  # COMPLETE THIS LINE. HINT: See previous exercises

#Now let us look for the intersections with the Poincare section over the
#solution. We first create an empty array to which we will append the
#points at which the flow pierces the Poincare section:
sspSolutionPoincare = np.array([], float)
#FILL IN sspSolutionPoincare, HINT: You can copy/paste corresponding block of
#code from Poincare.py

#At this point sspSolutionPoincare is a long vector each three elements
#corresponding to one intersection of the flow with the Poincare section
#we reshape it into an N x 3 form where each row corresponds to a different
#intersection:
sspSolutionPoincare = None  # COMPLETE THIS LINE. HINT: See Poincare.py
#Unit vectors which will span the Poincare section hyperplane are the
#template vector and the unit vector at z. Let us construct a matrix which
#projects state space vectors onto these basis:
e_z = np.array([0, 0, 1], float)  # Unit vector in z direction
ProjPoincare = np.array([None, None, None], float)  # COMPLETE THIS LINE.
                                                    # HINT: See Poincare.py
#sspSolutionPoincare has column vectors on its rows. We act on the
#transpose of this matrix to project each state space point onto Poincare
#basis by a simple matrix multiplication:
PoincareSection = np.dot(None, None)  # COMPLETE THIS LINE.
                                      # HINT: See Poincare.py
#We return to the usual N x 3 form by another transposition:
PoincareSection = None  # COMPLETE THIS LINE. HINT: Use .transpose()
#Third column of this matrix should be zero if everything is correct, so we
#discard it:
PoincareSection = PoincareSection[:, 0:2]

#We are now going to compute the pairwise distances between PoincareSection
#elements in order to sort them according to the increasing distance

#In previous exercise, we constructed a Poincare return map using the
#radial distance from the origin, this is an easy, however not a good way
#of attacking this problem since the return map you get would be multivalued
#if you are not lucky. Now we are going to do a better job and parametrize
#the Poincare section intersections according to arc lengths.

#We start by importing pdist and square form functions (see
#http://docs.scipy.org/doc/scipy-0.14.0/reference/spatial.distance.html for
#their documentation) to compute pairwise distances between Poincare
#section elements:
from scipy.spatial.distance import pdist, squareform
Distance = squareform(pdist(PoincareSection))
#Distance is a matrix which contains Euclidean distance between ith and
#jth elements of PoincareSection in its element [i,j]

#Now we are going to Sort the elements of the Poincare section according to
#increasing distances:
SortedPoincareSection = PoincareSection.copy()  # Copy PoincareSection into
                                                # a new variable
#Create a zero-array to assign arclengths of the Poincare section points
#after sorting:
ArcLengths = np.zeros(np.size(SortedPoincareSection, 0))
#Create another zero-array to assign the arclengths of the Poincare section
#points keeping their dynamical order for use in the return map
sn = np.zeros(np.size(PoincareSection, 0))
for k in range(np.size(SortedPoincareSection, 0) - 1):
    #Find the element which is closest to the kth point:
    m = np.argmin(Distance[k, k + 1:]) + k + 1
    #Hold the (k+1)th row in the dummy vector:
    dummyPoincare = SortedPoincareSection[k + 1, :].copy()
    #Replace (k+1)th row with the closest point:
    SortedPoincareSection[k + 1, :] = SortedPoincareSection[m, :]
    #Assign the previous (k+1)th row to the mth row:
    SortedPoincareSection[m, :] = dummyPoincare

    #Rearrange the distance matrix according to the new form of the
    #SortedPoincareSection array:
    dummyColumn = Distance[:, k + 1].copy()  # Hold (k+1)th column of the
                                             # distance matrix in a dummy
                                             # array
    Distance[:, k + 1] = Distance[:, m]  # Assign mth column to kth
    Distance[:, m] = dummyColumn

    dummyRow = Distance[k + 1, :].copy()  # Hold (k+1)th row in a dummy
                                          # array
    Distance[k + 1, :] = Distance[m, :]
    Distance[m, :] = dummyRow

    #Assign the arclength of (k+1)th element:
    ArcLengths[k + 1] = ArcLengths[k] + Distance[k, k + 1]
    #Find this point in the PoincareSection array and assign sn to its
    #corresponding arclength:
    sn[np.argwhere(PoincareSection[:, 0]
                   == SortedPoincareSection[k + 1, 0])] = ArcLengths[k + 1]

#We are now going to make a parametric spline interpolation to this curve.
#First we import the scipy.interpolate module, see its documentation at:
#http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/interpolate.html
from scipy import interpolate
#Parametric spline interpolation to the Poincare section:
tckPoincare, u = interpolate.splprep([SortedPoincareSection[:, 0],
                                      SortedPoincareSection[:, 1]],
                                      u=ArcLengths, s=0)

#Define the parametric representation of the Poincare section as a function:


def fPoincare(s):
    """
    Parametric interpolation to the Poincare section
    Inputs:
    s: Arc length which parametrizes the curve, a float or dx1-dim numpy
       array
    Outputs:
    xy = x and y coordinates on the Poincare section, 2-dim numpy array
       or (dx2)-dim numpy array
    """
    interpolation = interpolate.splev(s, tckPoincare)
    xy = np.array([interpolation[0], interpolation[1]], float).transpose()
    return xy

#Compute the interpolation:
#Create an array of arc lengths for which the Poincare section will be
#interpolated:
sArray = np.linspace(np.min(ArcLengths), np.max(ArcLengths), 1000)
#Evaluate the interpolation:
InterpolatedPoincareSection = fPoincare(sArray)

#We can now construct the return map over arclengths, we have already
#computed the array where arclengths of the Poincare section intersections
#are ordered in their dynamical order, we separate it into two parts to use
#them as the data to the return map:
sn1 = sn[0:-1]
sn2 = sn[1:]

#In order to be able to interpolate to the data, we need to arrange it such
#that the x-data is from the smallest to the largest:

#Indices on the order of which the sn1 is sorted from its smallest to the
#largest element:
isort = None  # COMPLETE THIS LINE. HINT: See Poincare.py, use np.argsort()

sn1 = sn1[isort]  # sort radii1
sn2 = sn2[isort]  # sort radii2

# Construct tck of the spline rep
tckReturn = None  # COMPLETE THIS LINE. HINT: See Poincare.py.
snPlus1 = interpolate.splev(sArray, tckReturn)  # Evaluate spline

# Finally, find the fixed point of this map:
# In order to solve with fsolve, construct a lambda function which would be
# zero at the fixed points of the return map:
ReturnMap = lambda s: None  # COMPLETE THIS LINE. HINT: See Poincare.py
sfixed = fsolve(ReturnMap, 10.0)[0]  # Change this initial guess by looking at

#We now have a candidate arclength that should be near to that of a fixed point
#of the return map, and fixed point of the return map must be a periodic orbit
#of the full flow!
#In order to find the state space point corresponding to this arc length, first
#we should find it on the Poincare section, luckily, we already have a function
#for that:
PoincareSectionfixed = fPoincare(sfixed)
#We then back-project from the Poincare section into the full state space
#using our Projection matrix:
sspfixed = np.dot(np.append(PoincareSectionfixed, 0.0), ProjPoincare)
#Note that we appended a 0 to the end of the PoincareSectionfixed vector
#in order to be able to multiply it with the projection matrix. This zero
#corresponds to the direction which is perpendicular to the Poincare
#section

#We will now find how much time does it takes for a trajectory which starts
#at this point to intersect the Poincare section for a second time, to do
#that we update our lambda function fdeltat for sspfixed:
fdeltat = lambda deltat: UPoincare(Rossler.Flow(sspfixed, deltat))
#In order to solve for this time, we need an initial guess. We guess it by
#dividing the total integration time of our simulation by the number of
#intersections with the Poincare section:
Tguess = tFinal / np.size(PoincareSection, 0)
Tnext = fsolve(fdeltat, Tguess)[0]  # Note that we pick the 0th element of the
                                    # array returned by fsolve, this is because
                                    # fsolve returns a numpy array even if the
                                    # problem is one dimensional
#Let us integrate to see if this was a good guess:
#Time array for solution from 0 to Tnext:
tArray = np.linspace(None, None, None)  # COMPLETE THIS LINE
#Integration:
sspfixedSolution = odeint(Rossler.Velocity, sspfixed, tArray)

#Lastly, we are going to refine our guesses to find the periodic orbit
#exactly. We are going to use Newton-Raphson scheme
#(eq. 13.9 from ChaosBook.org version14.5.7)
#We start by setting our tolerance, and calculating initial error:
tol = 1e-9
period = Tnext.copy()  # copy Tnext to a new variable period
error = np.zeros(4)  # Initiate the error vector
Delta = np.zeros(4)  # Initiate the delta vector
error[0:3] = Rossler.Flow(sspfixed, period) - sspfixed
Newton = np.zeros((4, 4))  # Initiate the 4x4 Newton matrix
#We are going to iterate the newton method until the maximum value of the
#absolute error meets the tolerance:
k = 0  # Counter for the Newton solver
kmax = 20  # Maximum steps for Newton solver to terminate if it is not
           # converging
while np.max(np.abs(error)) > tol:
    k += 1
    print(k)
    #Compute the Newton matrix:
    #First 3x3 block is 1 - J^t(x)
    Newton[0:3, 0:3] = None  # COMPLETE THIS LINE
    #Fourth column is the negative velocity at time T: -v(f^T(x))
    Newton[0:3, 3] = None  # COMPLETE THIS LINE
    #Fourth row is the Poincare section constraint:
    Newton[3, 0:3] = None  # COMPLETE THIS LINE
    #Now we will invert this matrix and act on the error vector to find the
    #updates to our guesses:
    Delta = np.dot(np.linalg.inv(Newton), error)
    #Update our guesses:
    sspfixed = sspfixed + Delta[0:3]
    period = period + Delta[3]
    #Compute the new errors:
    error[0:3] = Rossler.Flow(sspfixed, period) - sspfixed
    if k > kmax:
        print("Passed the maximum number of iterations")
        break
print("Shortest peridoic orbit is at: ", sspfixed[0],
                                         sspfixed[1],
                                         sspfixed[2])
print("Period:", period)

#Let us integrate the periodic orbit:
tArray = np.linspace(0, period, 1000)  # Time array for solution
#Integration:
periodicOrbit = odeint(Rossler.Velocity, sspfixed, tArray)

#Import plotting functions:
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig1 = plt.figure(1)  # Create a figure instance
ax = fig1.gca(projection='3d')  # Get current axes in 3D projection
# Plot the solution:
ax.plot(sspSolution[:, 0], sspSolution[:, 1], sspSolution[:, 2], linewidth=0.5)
ax.set_xlabel('$x$')  # Set x label
ax.set_ylabel('$y$')  # Set y label
ax.set_zlabel('$z$')  # Set z label
ax.hold('True')  # Set hold true in order to plot the Poincare section on
                 # top of the solution
ax.plot(sspSolutionPoincare[:, 0],
        sspSolutionPoincare[:, 1],
        sspSolutionPoincare[:, 2], '.r', markersize=4)

fig2 = plt.figure(2)  # Create another figure instance
ax = fig2.gca()  # Get current axes
ax.plot(PoincareSection[:, 0], PoincareSection[:, 1], '.r', markersize=3)
ax.hold('True')
ax.plot(SortedPoincareSection[:, 0], SortedPoincareSection[:, 1])
ax.plot(InterpolatedPoincareSection[:, 0],
        InterpolatedPoincareSection[:, 1])
ax.set_xlabel('$\\hat{x}\'$')  # Set x label
ax.set_ylabel('$z$')  # Set y label

fig3 = plt.figure(3, figsize=(8, 8))  # Create another figure instance
ax = fig3.gca()  # Get current axes
ax.set_aspect('equal')  # Set the aspect ratio of the plot to the square
ax.plot(sn1, sn2, '.r', markersize=5)
ax.hold('True')  # Set hold true to plot the interpolation on top of the
                 # data points
ax.plot(sArray, snPlus1)
ax.plot(sArray, sArray, 'k')
ax.set_xlabel('$s_n$')  # Set x label
ax.set_ylabel('$s_{n+1}$')  # Set y label

fig4 = plt.figure(4)  # Create another figure instance
ax = fig4.gca(projection='3d')  # Get current axes in 3D projection
# Plot the solution:
ax.plot(sspfixedSolution[:, 0],
        sspfixedSolution[:, 1],
        sspfixedSolution[:, 2])
ax.set_xlabel('$x$')  # Set x label
ax.set_ylabel('$y$')  # Set y label
ax.set_zlabel('$z$')  # Set z label
ax.hold('True')  # Set hold true in order to plot the Poincare section on

# Plot the periodic orbit:
ax.plot(periodicOrbit[:, 0],
        periodicOrbit[:, 1],
        periodicOrbit[:, 2])

plt.show()  # Show the figures