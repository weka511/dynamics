from matplotlib.pyplot      import figure, show
from numpy                  import abs, append, argmin, argsort, argwhere, array, dot, cos, linspace, max, pi, real, sin, size, zeros
from numpy.linalg           import eig, inv, norm
from scipy.integrate        import odeint
from scipy.optimize         import fsolve
from Rossler                import Flow, StabilityMatrix, Velocity, Jacobian
from scipy.interpolate      import splev,  splprep, splrep
from scipy.spatial.distance import pdist, squareform

def zRotation(theta):
    '''
    Rotation matrix about z-axis
    Input:
    theta: Rotation angle (radians)
    Output:
    Rz: Rotation matrix about z-axis
    '''
    return array([[cos(theta), -sin(theta), 0],
                  [sin(theta), cos(theta),  0],
                  [0,          0,           1]],
                 float)


def create_trajectory(epsilon  =1e-6,
                      tInitial = 0,
                      tFinal   = 1000,
                      Nt       = 100000):
    '''
    We will first run a long trajectory of the Rossler system by starting
    close to the eq0 in order to include its unstable manifold on the Poincare
    section. Let us start by repeating what we have done in the stability
    exercise and construct this initial condition:
    Numerically find the equilibrium of the Rossler system close to the
    origin
    '''
    eq0                       = fsolve(Velocity, array([0, 0, 0], float), args=(0,))
    Aeq0                      = StabilityMatrix(eq0)
    eigenValues, eigenVectors = eig(Aeq0)
    v1                        = real(eigenVectors[:, 0]) #Real part of the leading eigenvector
    v1                        = v1 / norm(v1)
    ssp0                      = eq0 + epsilon * v1  #Initial condition as a slight perturbation to the eq0 in v1 direction

    tArray                    = linspace(tInitial, tFinal, Nt)
    sspSolution               = odeint(Velocity, ssp0, tArray)
    return tArray, sspSolution

if __name__=='__main__':
    thetaPoincare = 0.0 #Angle between the Poincare section hyperplane and the x-axis:

    #Define vectors which will be on and orthogonal to the Poincare section
    #hyperplane:

    e_x         = array([1, 0, 0], float)  # Unit vector in x-direction
    sspTemplate = dot(zRotation(thetaPoincare), e_x)  #Template vector to define the Poincare section hyperplane
    nTemplate   = dot(zRotation(pi/2), sspTemplate)  #Normal to this plane will be equal to template vector rotated pi/2 about the z axis

    #Define the Poincare section hyperplane equation as a Lambda function based on
    #the UPoincare from Poincare module, using our new sspTemplate and nTemplate:
    # UPoincare = lambda ssp: Poincare.UPoincare(ssp, sspTemplate, nTemplate)

    def UPoincare(ssp, sspTemplate=sspTemplate, nTemplate=nTemplate):
        '''
        Plane equation for the Poincare section hyperplane which includes z-axis
        and makes an angle theta with the x-axis see ChaosBook ver. 14, fig. 3.2
        Inputs:
        ssp: State space point at which the Poincare hyperplane equation will be
             evaluated
        Outputs:
        U: Hyperplane equation which should be satisfied on the Poincare section
           U = (ssp - sspTemplate) . nTemplate (see ChaosBook ver. 14, eq. 3.6)
        '''

        return dot((ssp - sspTemplate) , nTemplate)

    tArray, sspSolution = create_trajectory()

    #Now let us look for the intersections with the Poincare section over the
    #solution. We first create an empty array to which we will append the
    #points at which the flow pierces the Poincare section:

    sspSolutionPoincare = array([], float)
    for i in range(size(sspSolution, 0) - 1):
        #Look at every instance from integration and search for Poincare
        #section hyperplane crossings:
        if UPoincare(sspSolution[i]) < 0 and UPoincare(sspSolution[i+1]) > 0:  #If the hyperplane equation is lesser than zero at one instance
                                                                                #and greater than zero at the next, this implies that there is a
                                                                                #zero in between
            sspPoincare0        = sspSolution[i]  # Initial point for the `fine' integration

            deltat0             = (tArray[i + 1] - tArray[i]) / 2       #Initial guess for the how much time one needs to integrate
                                                                        #starting at sspPoincare0 in order to exactly land on the Poincare
                                                                        #section

            fdeltat             = lambda deltat: UPoincare(Flow(sspPoincare0, deltat))  #Define the equation for deltat which must be solved as a lambda function

            deltat              = fsolve(fdeltat, deltat0)       #Find deltat at which fdeltat is 0:
            sspPoincare         = Flow(sspPoincare0, deltat)    #Now integrate deltat from sspPoincare0 to find where exactly the
                                                                #flow pierces the Poincare section:
            sspSolutionPoincare = append(sspSolutionPoincare, sspPoincare)

    #At this point sspSolutionPoincare is a long vector each three elements
    #corresponding to one intersection of the flow with the Poincare section
    #we reshape it into an N x 3 form where each row corresponds to a different
    #intersection:
    sspSolutionPoincare = sspSolutionPoincare.reshape(
                                                size(sspSolutionPoincare, 0) // 3,
                                                3)


    fig  = figure(figsize=(12,12))
    ax   = fig.gca(projection='3d')
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
    show()
