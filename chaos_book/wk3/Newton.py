'''Q3.1 Shortest periodic orbit of the Rössler system '''

from matplotlib.pyplot      import figure, show
from numpy                  import abs, append, argmin, argsort, argwhere, array, dot, cos, linspace, max, pi, real, sin, size, zeros
from numpy.linalg           import eig, inv, norm
from scipy.integrate        import odeint
from scipy.optimize         import fsolve
from Rossler                import Flow, StabilityMatrix, Velocity, Jacobian
from scipy.interpolate      import splev,  splprep, splrep
from scipy.spatial.distance import pdist, squareform

class Poincare:
    def __init__(self,
                 thetaPoincare = 0.0#Angle between the Poincare section hyperplane and the x-axis
                ):
    #Define vectors which will be on and orthogonal to the Poincare section
    #hyperplane:
        self.e_x         = array([1, 0, 0], float)  # Unit vector in x-direction
        self.e_z         = array([0, 0, 1], float)  # Unit vector in z direction
        self.sspTemplate = dot(self.zRotation(thetaPoincare), self.e_x)  #Template vector to define the Poincare section hyperplane
        self.nTemplate   = dot(self.zRotation(pi/2), self.sspTemplate)  #Normal to this plane will be equal to template vector rotated pi/2 about the z axis

    def zRotation(self,theta):
        """
        Rotation matrix about z-axis
        Input:
        theta: Rotation angle (radians)
        Output:
        Rz: Rotation matrix about z-axis
        """
        return array([[cos(theta), -sin(theta), 0],
                      [sin(theta), cos(theta),  0],  # Simon
                      [0,          0,           1]],
                     float)  # Simon

    #Define the Poincare section hyperplane equation as a Lambda function based on
    #the UPoincare from Poincare module, using our new sspTemplate and nTemplate:
    # UPoincare = lambda ssp: Poincare.UPoincare(ssp, sspTemplate, nTemplate)

    def UPoincare(self,ssp,
                  sspTemplate=None,
                  nTemplate=None):
        """
        Plane equation for the Poincare section hyperplane which includes z-axis
        and makes an angle theta with the x-axis see ChaosBook ver. 14, fig. 3.2
        Inputs:
        ssp: State space point at which the Poincare hyperplane equation will be
             evaluated
        Outputs:
        U: Hyperplane equation which should be satisfied on the Poincare section
           U = (ssp - sspTemplate) . nTemplate (see ChaosBook ver. 14, eq. 3.6)
        """

        return dot((ssp - (self.sspTemplate if sspTemplate==None else sspTemplate)),
                   self.nTemplate if nTemplate==None else nTemplate)

    def ProjPoincare(self):
        '''Unit vectors which will span the Poincare section hyperplane are the
        template vector and the unit vector at z. Let us construct a matrix which
        projects state space vectors onto these basis'''

        return  array([self.sspTemplate,
                             self.e_z,
                             self.nTemplate], float)

    def get_intersections(self,sspSolution):
        sspSolutionPoincare = array([], float)
        for i in range(size(sspSolution, 0) - 1):
            #Look at every instance from integration and search for Poincare
            #section hyperplane crossings:
            if self.UPoincare(sspSolution[i]) < 0 and self.UPoincare(sspSolution[i+1]) > 0:  #If the hyperplane equation is lesser than zero at one instance
                                                                                    #and greater than zero at the next, this implies that there is a
                                                                                    #zero in between
                sspPoincare0        = sspSolution[i]  # Initial point for the `fine' integration

                deltat0             = (tArray[i + 1] - tArray[i]) / 2       #Initial guess for the how much time one needs to integrate
                                                                            #starting at sspPoincare0 in order to exactly land on the Poincare
                                                                            #section

                fdeltat             = lambda deltat: self.UPoincare(Flow(sspPoincare0, deltat))  #Define the equation for deltat which must be solved as a lambda function

                deltat              = fsolve(fdeltat, deltat0)       #Find deltat at which fdeltat is 0:
                sspPoincare         = Flow(sspPoincare0, deltat)    #Now integrate deltat from sspPoincare0 to find where exactly the
                                                                    #flow pierces the Poincare section:
                sspSolutionPoincare = append(sspSolutionPoincare, sspPoincare)

        #At this point sspSolutionPoincare is a long vector each three elements
        #corresponding to one intersection of the flow with the Poincare section
        #we reshape it into an N x 3 form where each row corresponds to a different
        #intersection:
        return sspSolutionPoincare.reshape(size(sspSolutionPoincare, 0) // 3, 3)

# We will first run a long trajectory of the Rossler system by starting
# close to the eq0 in order to include its unstable manifold on the Poincare
# section. Let us start by repeating what we have done in the stability
# exercise and construct this initial condition:
# Numerically find the equilibrium of the Rossler system close to the
# origin:
eq0                       = fsolve(Velocity, array([0, 0, 0], float), args=(0,))
Aeq0                      = StabilityMatrix(eq0)
eigenValues, eigenVectors = eig(Aeq0)
v1                        = real(eigenVectors[:, 0]) #Real part of the leading eigenvector
v1                        = v1 / norm(v1)
ssp0                      = eq0 + 1e-6 * v1  #Initial condition as a slight perturbation to the eq0 in v1 direction

tInitial                  = 0
tFinal                    = 1000
Nt                        = 100000
tArray                    = linspace(tInitial, tFinal, Nt)
sspSolution               = odeint(Velocity, ssp0, tArray)

#Now let us look for the intersections with the Poincare section over the
#solution. We first create an empty array to which we will append the
#points at which the flow pierces the Poincare section:

poincare            = Poincare()
sspSolutionPoincare = poincare.get_intersections(sspSolution)
ProjPoincare        = poincare.ProjPoincare()

#sspSolutionPoincare has column vectors on its rows. We act on the
#transpose of this matrix to project each state space point onto Poincare
#basis by a simple matrix multiplication:
PoincareSection = dot(ProjPoincare, sspSolutionPoincare.transpose())
PoincareSection =  PoincareSection.transpose()   #We return to the usual N x 3 form by another transposition
#Third column of this matrix should be zero if everything is correct, so we
#discard it:

PoincareSection = PoincareSection[:, 0:2]  # Does this actually do anything?

# We are now going to compute the pairwise distances between PoincareSection
# elements in order to sort them according to the increasing distance

# In previous exercise, we constructed a Poincare return map using the
# radial distance from the origin, this is an easy, however not a good way
# of attacking this problem since the return map you get would be multivalued
# if you are not lucky. Now we are going to do a better job and parametrize
# the Poincare section intersections according to arc lengths.

Distance = squareform(pdist(PoincareSection))
#Distance is a matrix which contains Euclidean distance between ith and
#jth elements of PoincareSection in its element [i,j]

#Now we are going to Sort the elements of the Poincare section according to
#increasing distances:
SortedPoincareSection = PoincareSection.copy()  # Copy PoincareSection into
                                                # a new variable
#Create a zero-array to assign arclengths of the Poincare section points
#after sorting:
ArcLengths = zeros(size(SortedPoincareSection, 0))
# Create another zero-array to assign the arclengths of the Poincare section
# points keeping their dynamical order for use in the return map
sn = zeros(size(PoincareSection, 0))
for k in range(size(SortedPoincareSection, 0) - 1):
    #Find the element which is closest to the kth point:
    m = argmin(Distance[k, k + 1:]) + k + 1
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
    sn[argwhere(PoincareSection[:, 0]
                   == SortedPoincareSection[k + 1, 0])] = ArcLengths[k + 1]



#Parametric spline interpolation to the Poincare section:
tckPoincare, u = splprep([SortedPoincareSection[:, 0],
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
    interpolation = splev(s, tckPoincare)
    xy = array([interpolation[0], interpolation[1]], float).transpose()
    return xy

#Compute the interpolation:
#Create an array of arc lengths for which the Poincare section will be
#interpolated:
sArray = linspace(min(ArcLengths), max(ArcLengths), 1000)
#Evaluate the interpolation:
InterpolatedPoincareSection = fPoincare(sArray)

# We can now construct the return map over arclengths, we have already
# computed the array where arclengths of the Poincare section intersections
# are ordered in their dynamical order, we separate it into two parts to use
#them as the data to the return map:
sn1 = sn[0:-1]
sn2 = sn[1:]

#In order to be able to interpolate to the data, we need to arrange it such
#that the x-data is from the smallest to the largest:

#Indices on the order of which the sn1 is sorted from its smallest to the
#largest element:

isort = argsort(sn1)
sn1   = sn1[isort]  # sort radii1
sn2   = sn2[isort]  # sort radii2

# Construct tck of the spline rep
tckReturn = splrep(sn1,sn2)
snPlus1 = splev(sArray, tckReturn)  # Evaluate spline

# Finally, find the fixed point of this map:
# In order to solve with fsolve, construct a lambda function which would be
# zero at the fixed points of the return map:
ReturnMap = lambda r: splev(r, tckReturn) - r
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
sspfixed = dot(append(PoincareSectionfixed, 0.0), ProjPoincare)
#Note that we appended a 0 to the end of the PoincareSectionfixed vector
#in order to be able to multiply it with the projection matrix. This zero
#corresponds to the direction which is perpendicular to the Poincare
#section

#We will now find how much time does it takes for a trajectory which starts
#at this point to intersect the Poincare section for a second time, to do
#that we update our lambda function fdeltat for sspfixed:
fdeltat = lambda deltat: poincare.UPoincare(Flow(sspfixed, deltat))
#In order to solve for this time, we need an initial guess. We guess it by
#dividing the total integration time of our simulation by the number of
#intersections with the Poincare section:
Tguess = tFinal / size(PoincareSection, 0)
Tnext = fsolve(fdeltat, Tguess)[0]  # Note that we pick the 0th element of the
                                    # array returned by fsolve, this is because
                                    # fsolve returns a numpy array even if the
                                    # problem is one dimensional
#Let us integrate to see if this was a good guess:
#Time array for solution from 0 to Tnext:
tArray = linspace(0, Tnext, 100)
#Integration:
sspfixedSolution = odeint(Velocity, sspfixed, tArray)

#Lastly, we are going to refine our guesses to find the periodic orbit
#exactly. We are going to use Newton-Raphson scheme
#(eq. 13.9 from ChaosBook.org version14.5.7)
#We start by setting our tolerance, and calculating initial error:
tol        = 1e-9
period     = Tnext.copy()  # copy Tnext to a new variable period
error      = zeros(4)  # Initiate the error vector
Delta      = zeros(4)  # Initiate the delta vector
error[0:3] = Flow(sspfixed, period) - sspfixed
Newton     = zeros((4, 4))  # Initiate the 4x4 Newton matrix

k    = 0
kmax = 20
#We are going to iterate the newton method until the maximum value of the
#absolute error meets the tolerance:
while max(abs(error)) > tol:
    if k > kmax:
        print("Passed the maximum number of iterations")
        break
    k += 1
    print(f'Iteration {k}')
    Newton[0:3, 0:3] = 1-Jacobian(sspfixed,Tnext)     #First 3x3 block is 1 - J^t(x)
    Newton[0:3, 3]  = -Velocity(sspfixed,Tnext)   #Fourth column is the negative velocity at time T: -v(f^T(x))
    Newton[3, 0:3]  = poincare.nTemplate# dot(nTemplate,error[0:2])   #Fourth row is the Poincare section constraint:
    Delta           = dot(inv(Newton), error)     #Now we will invert this matrix and act on the error vector to find the updates to our guesses:
    sspfixed        = sspfixed + Delta[0:3]   #Update our guesses:
    period          = period + Delta[3]
    error[0:3]      = Flow(sspfixed, period) - sspfixed #Compute the new errors:


print("Shortest periodic orbit is at: ", sspfixed[0],
                                         sspfixed[1],
                                         sspfixed[2])
print("Period:", period)

#Let us integrate the periodic orbit:
tArray        = linspace(0, period, 1000)  # Time array for solution integration
periodicOrbit = odeint(Velocity, sspfixed, tArray)

fig  = figure(figsize=(12,12))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(sspSolution[:, 0], sspSolution[:, 1], sspSolution[:, 2],
        linewidth = 0.5,
        label     = 'Rossler')

ax.plot(sspSolutionPoincare[:, 0],
        sspSolutionPoincare[:, 1],
        sspSolutionPoincare[:, 2], '.r',
        markersize = 4,
        label      = 'Recurrences')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_title('Poincare Recurrences for Rossler')
ax.legend()

ax = fig.add_subplot(2, 2, 2)
ax.plot(PoincareSection[:, 0], PoincareSection[:, 1], '.r',
        markersize = 5,
        label      = 'Poincare Section')

ax.plot(SortedPoincareSection[:, 0], SortedPoincareSection[:, 1],'.b',
         markersize = 3,
        label = 'Sorted Poincare Section')

ax.plot(InterpolatedPoincareSection[:, 0], InterpolatedPoincareSection[:, 1],'.g',
        markersize = 1,
        label = 'Interpolated Poincare Section')

ax.set_title('Poincare Section, showing Interpolation')
ax.set_xlabel('$\\hat{x}\'$')
ax.set_ylabel('$z$')
ax.legend()

ax = fig.add_subplot(2, 2, 3)
ax.set_aspect('equal')
ax.plot(sn1, sn2, '.r',
        markersize = 5,
        label='sn1:sn2')

ax.plot(sArray, snPlus1,'.b',
        label = 'sArray:snPlus1')
ax.plot(sArray, sArray, 'k',
        label= '$s_n=s_{n+1}$')
ax.set_xlabel('$s_n$')
ax.set_ylabel('$s_{n+1}$')  # Set y label
ax.set_title('Return map')
ax.legend(loc = 'right')

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.set_title('Periodic Orbit')
ax.plot(sspfixedSolution[:, 0],
        sspfixedSolution[:, 1],
        sspfixedSolution[:, 2],
        color='xkcd:green',
        label='sspfixedSolution')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')


# Plot the periodic orbit:
ax.plot(periodicOrbit[:, 0],
        periodicOrbit[:, 1],
        periodicOrbit[:, 2],
        color='xkcd:purple',
        label='Periodic Orbit')
ax.legend()

show()  # Show the figures
