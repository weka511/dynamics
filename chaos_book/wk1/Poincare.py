# -*- coding: iso-8859-1 -*-
# Q1.5 Poincaré sections and return maps of the Rössler system

from matplotlib.pyplot import figure, show
from numpy             import append, argsort, array, cos, dot, linspace, max, min, pi, sin, size
from scipy.integrate   import odeint
from scipy.optimize    import fsolve
from scipy.interpolate import splev,  splrep
from Rossler           import Flow, Velocity

#Define the matrix which rotates vectors about z-axis


def zRotation(theta):
    """
    Rotation matrix about z-axis
    Input:
    theta: Rotation angle (radians)
    Output:
    Rz: Rotation matrix about z-axis
    """
    return array([[cos(theta), -sin(theta), 0],
                    [sin(theta), cos(theta), 0],  # Simon
                    [0, 0, 1]], float)  # Simon


#Set the angle between the Poincare section hyperplane and the x-axis:
thetaPoincare = -pi / 2.0

#Define vectors which will be on and orthogonal to the Poincare section
#hyperplane:
e_x = array([1, 0, 0], float)  # Unit vector in x-direction
#Template vector to define the Poincare section hyperplane:
sspTemplate = dot(zRotation(thetaPoincare), e_x)  # Matrix multiplication in
                                                     # numpy is handled by the
                                                     #`dot' function, see numpy
                                                     # reference to learn more
#Normal to this plane will be equal to template vector rotated pi/2 about
#the z axis:
nTemplate = dot(zRotation(pi/2), sspTemplate)  # COMPLETE THIS LINE

#Define the Poincare section hyperplane equation


def UPoincare(ssp, sspTemplate=sspTemplate, nTemplate=nTemplate):
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
    return dot((ssp - sspTemplate) , nTemplate)  # Simon



if __name__ == "__main__":
    #We will first run an ergodic trajectory on the Rossler attractor:
    tInitial = 0  # Initial time
    tFinal = 200  # Final time
    Nt = 10000  # Number of time points to be used in the integration

    tArray = linspace(tInitial, tFinal, Nt)  # Time array for solution
    #Initial condition on the attractor:
    ssp0 = array([9.64832079329, -3.51977184351, 0.811107128559], float)
    sspSolution = odeint(Velocity, ssp0, tArray)

    #Now let us look for the intersections with the Poincare section over the
    #solution. We first create an empty array to which we will append the
    #points at which the flow pierces the Poincare section:
    sspSolutionPoincare = array([], float)
    for i in range(size(sspSolution, 0) - 1):
        #Look at every instance from integration and search for Poincare
        #section hyperplane crossings:
        if UPoincare(sspSolution[i]) < 0 and UPoincare(sspSolution[i+1]) > 0: #Simon
            #COMPLETE THE LINE ABOVE, HINT:
            #If the hyperplane equation is lesser than zero at one instance
            #and greater than zero at the next, this implies that there is a
            #zero in between
            sspPoincare0 = sspSolution[i]  # Initial point for the `fine'
                                           # integration
            #Initial guess for the how much time one needs to integrate
            #starting at sspPoincare0 in order to exactly land on the Poincare
            #section
            deltat0 = (tArray[i + 1] - tArray[i]) / 2
            #Define the equation for deltat which must be solved as a lambda function
            fdeltat = lambda deltat: UPoincare(Flow(sspPoincare0, deltat))
            #Find deltat at which fdeltat is 0:
            deltat = fsolve(fdeltat, deltat0)
            #Now integrate deltat from sspPoincare0 to find where exactly the
            #flow pierces the Poincare section:
            sspPoincare = Flow(sspPoincare0, deltat)
            sspSolutionPoincare = append(sspSolutionPoincare, sspPoincare)
    #At this point sspSolutionPoincare is a long vector each three elements
    #corresponding to one intersection of the flow with the Poincare section
    #we reshape it into an N x 3 form where each row corresponds to a different
    #intersection:

    sspSolutionPoincare = sspSolutionPoincare.reshape(
                                            size(sspSolutionPoincare, 0) // 3,
                                                      3)
    #Unit vectors which will span the Poincare section hyperplane are the
    #template vector and the unit vector at z. Let us construct a matrix which
    #projects state space vectors onto these basis:
    e_z          = array([0, 0, 1], float)  # Unit vector in z direction
    ProjPoincare = array([sspTemplate,
                             e_z,
                             nTemplate], float)
    #sspSolutionPoincare has state space vectors on its rows. We act on the
    #transpose of this matrix to project each state space point onto Poincare
    #basis by a simple matrix multiplication:
    PoincareSection = dot(ProjPoincare, sspSolutionPoincare.transpose())
    #We return to the usual N x 3 form by another transposition:
    PoincareSection = PoincareSection.transpose()  # Third column of this
                                                   # matrix should be zero if
                                                   # everything is correct.
    #Now let us try to construct a return map of radii. We have already
    #computed radii of points on the Poincare section by projecting them onto
    #the Poincare section basis. Projection onto the first Poincare section
    #basis, in this particular cases, is equal to the radial distance of the
    #Poincare section intersection to the origin. Let's first construct two
    #arrays: One of them will contain the radii of the Poincare section points
    #except the last one, the will contain the radii of the Poincare section
    #points except the first one. We will obtain the return map by plotting
    #the former versus the latter.
    radii1 = PoincareSection[0:-1, 0]
    radii2 = PoincareSection[1:, 0]

    #Finally, let us interpolate to this return map with splines and find where
    #it returns to itself

    #See http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    #to learn more on interpolation using scipy.interpolate
    #In order to be able to interpolate to the data, we need to arrange it such
    #that the x-data is from the smallest to the largest. This is not
    #necessarily what we get from the flow itself, so we have to rearrange
    #data arrays radii1 and radii2:
    isort = argsort(radii1)  # Indices on the order of which the radii1 is
                                # sorted from its smallest to the largest
                                # element
    radii1 = radii1[isort]  # sort radii1
    radii2 = radii2[isort]  # sort radii2

    tck = splrep(radii1, radii2)  # Construct tck of the spline
                                              # representation
    #Construct a radius array to input the interpolation function:
    rn      = linspace(min(radii1), max(radii1), 100)
    rnPlus1 = splev(rn, tck)  # Evaluate spline representation

    #Finally, find the fixed point of this map:
    #In order to solve with fsolve, construct a lambda function which would be
    #zero at the fixed points of the return map:
    ReturnMap = lambda r: splev(r, tck) - r
    #UNCOMMENT FOLLOWING TWO LINES AFTER READING INITIAL GUESS FOR THE SOLVER
    #FROM THE RETURN MAP
    rfixed = fsolve(ReturnMap, 8.3871)
    print(rfixed)  #  8.38094142

    fig1 = figure(1)  # Create a figure instance
    ax = fig1.gca(projection='3d')  # Get current axes in 3D projection
    #Plot the solution:
    ax.plot(sspSolution[:, 0], sspSolution[:, 1], sspSolution[:, 2])
    ax.set_xlabel('$x$')  # Set x label
    ax.set_ylabel('$y$')  # Set y label
    ax.set_zlabel('$z$')  # Set z label

    ax.plot(sspSolutionPoincare[:, 0],
            sspSolutionPoincare[:, 1],
            sspSolutionPoincare[:, 2], '.r', markersize=4)

    fig2 = figure(2)  # Create another figure instance
    ax = fig2.gca()  # Get current axes
    ax.plot(PoincareSection[:, 0], PoincareSection[:, 1], '.r', markersize=3)
    ax.set_xlabel('$\\hat{x}\'$')  # Set x label
    ax.set_ylabel('$z$')  # Set y label

    fig3 = figure(3, figsize=(8, 8))  # Create another figure instance
    ax = fig3.gca()  # Get current axes
    ax.set_aspect('equal')  # Set the aspect ratio of the plot to the square
    ax.plot(radii1, radii2, '.r', markersize=5)

    ax.plot(rn, rnPlus1)
    ax.plot(rn, rn, 'k')
    ax.set_xlabel('$r_n$')  # Set x label
    ax.set_ylabel('$r_{n+1}$')  # Set y label
    show()
