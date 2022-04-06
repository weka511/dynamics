from matplotlib.pyplot      import figure, show
from numpy                  import abs, append, argmin, argsort, argwhere, array, dot, cos, linspace, max, pi, real, sin, size, zeros
from numpy.linalg           import eig, inv, norm
from scipy.integrate        import odeint
from scipy.optimize         import fsolve
from Rossler                import Flow, StabilityMatrix, Velocity, Jacobian
from scipy.interpolate      import splev,  splprep, splrep
from scipy.spatial.distance import pdist, squareform


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

class UPoincare:
    '''
    Define the Poincare section hyperplane equation as a Lambda function based on
    the UPoincare from Poincare module, using our new sspTemplate and nTemplate:
    UPoincare = lambda ssp: Poincare.UPoincare(ssp, sspTemplate, nTemplate)
    '''
    @classmethod
    def zRotation(cls,theta):
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

    def __init__(self,thetaPoincare):
        '''
        Define vectors which will be on and orthogonal to the Poincare section hyperplane:
        '''
        e_x         = array([1, 0, 0], float)  # Unit vector in x-direction
        sspTemplate = dot(UPoincare.zRotation(thetaPoincare), e_x)  #Template vector to define the Poincare section hyperplane
        nTemplate   = dot(UPoincare.zRotation(pi/2), sspTemplate)  #Normal to this plane will be equal to template vector rotated pi/2 about the z axis
        self.sspTemplate = sspTemplate
        self.nTemplate   = nTemplate

    def UPoincare(self,ssp):
        '''
        Plane equation for the Poincare section hyperplane which includes z-axis
        and makes an angle theta with the x-axis see ChaosBook ver. 14, fig. 3.2
        Parameters:
            ssp: State space point at which the Poincare hyperplane equation will be evaluated
        Returns:
            U: Hyperplane equation which should be satisfied on the Poincare section
               U = (ssp - sspTemplate) . nTemplate (see ChaosBook ver. 14, eq. 3.6)
        '''

        return dot((ssp - self.sspTemplate) , self.nTemplate)

def get_crossing(sspSolution,i,upoincare,tArray):
    '''
    find where exactly the flow pierces the Poincare section:
    '''

    return Flow(sspSolution[i],
                fsolve(lambda deltat: upoincare.UPoincare(Flow(sspSolution[i] , deltat)),
                                 (tArray[i + 1] - tArray[i]) / 2) )    #Now integrate deltat from sspPoincare0 to find where exactly the
                                                       #flow pierces the Poincare section:

def create_section(thetaPoincare,tArray, sspSolution):
    upoincare   = UPoincare(thetaPoincare)

    return array([get_crossing(sspSolution,i,upoincare,tArray)
                  for i in range(size(sspSolution, 0) - 1)
                  if upoincare.UPoincare(sspSolution[i]) < 0 and upoincare.UPoincare(sspSolution[i+1]) > 0],
                 float)

if __name__=='__main__':
    Angles              = [0, 60, 120, -60]
    tArray, sspSolution = create_trajectory()

    PoincareSections    = [create_section((angle/60) * (pi/3),tArray, sspSolution) for angle in Angles]

    fig                 = figure(figsize=(12,12))
    ax                  = fig.gca(projection='3d')
    ax.plot(sspSolution[:, 0], sspSolution[:, 1], sspSolution[:, 2],
            linewidth = 0.5,
            label     = 'Rossler')

    styles = ['.r', '.g', '.c', '.m']
    for i in range(len(Angles)):
        text = 'Section' if i==0 else '   "   '
        ax.plot(PoincareSections[i][:, 0],
                PoincareSections[i][:, 1],
                PoincareSections[i][:, 2],
                styles[i],
                markersize = 4,
                label      = f'{text} {Angles[i]:4d}')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_title('Poincare Recurrences for Rossler')
    ax.legend()
    show()
