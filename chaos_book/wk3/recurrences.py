from matplotlib.pyplot      import figure, savefig, show
from numpy                  import abs, append, argmin, argsort, argwhere, array, dot, cos, linspace, max, pi, real, sin, size, zeros
from numpy.linalg           import eig, inv, norm
from scipy.integrate        import odeint
from scipy.optimize         import fsolve
from Rossler                import Flow, StabilityMatrix, Velocity, Jacobian
from scipy.interpolate      import splev,  splprep, splrep
from scipy.spatial.distance import pdist, squareform


def create_trajectory(epsilon  = 1e-6,
                      tInitial = 0,
                      tFinal   = 1000,
                      Nt       = 100000):
    '''
    Create a long trajectory of the Rossler system by starting lose to the eq0
    in order to include its unstable manifold on the Poincare
    section.
    '''
    def get_normalized_real(V):
        '''Used to get normalized real part of a vector'''
        v1 = real(V)
        return v1 / norm(v1)

    # Find the equilibrium of the Rossler system close to the origin
    eq0             = fsolve(Velocity, array([0, 0, 0], float), args = (0,))
    _, eigenVectors = eig(StabilityMatrix(eq0))
    tArray          = linspace(tInitial, tFinal, Nt)
    return tArray, odeint(Velocity,  eq0 + epsilon * get_normalized_real(eigenVectors[:, 0]), tArray)

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
        e_x              = array([1, 0, 0], float)  # Unit vector in x-direction
        sspTemplate      = dot(UPoincare.zRotation(thetaPoincare), e_x)  #Template vector to define the Poincare section hyperplane
        nTemplate        = dot(UPoincare.zRotation(pi/2), sspTemplate)  #Normal to this plane will be equal to template vector rotated pi/2 about the z axis
        self.sspTemplate = sspTemplate
        self.nTemplate   = nTemplate
        self.e_z         = array([0, 0, 1], float)

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

    def do_stuff(self,sspSolutionPoincare):
        ProjPoincare = array([self.sspTemplate,
                              self.e_z,
                              self.nTemplate], float)
        #sspSolutionPoincare has column vectors on its rows. We act on the
        #transpose of this matrix to project each state space point onto Poincare
        #basis by a simple matrix multiplication:
        PoincareSection = dot(ProjPoincare, sspSolutionPoincare.transpose())
        PoincareSection =  PoincareSection.transpose()   #We return to the usual N x 3 form by another transposition
        #Third column of this matrix should be zero if everything is correct, so we
        #discard it:

        PoincareSection = PoincareSection[:, 0:2]
        return PoincareSection

    def foo(self,PoincareSection):
        Distance = squareform(pdist(PoincareSection))
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
        return SortedPoincareSection

def get_crossing(sspSolution,i,upoincare,tArray):
    '''
    find where exactly the flow pierces the Poincare section:
    '''
    return Flow(sspSolution[i],
                fsolve(lambda deltat: upoincare.UPoincare(Flow(sspSolution[i] , deltat)),
                                 (tArray[i + 1] - tArray[i]) / 2) )    #Now integrate deltat from sspPoincare0 to find where exactly the
                                                       #flow pierces the Poincare section:


def create_section(upoincare,tArray, sspSolution):

    return array([get_crossing(sspSolution,i,upoincare,tArray)
                  for i in range(size(sspSolution, 0) - 1)
                  if upoincare.UPoincare(sspSolution[i]) < 0 and upoincare.UPoincare(sspSolution[i+1]) > 0],
                 float)

if __name__=='__main__':
    Angles              = [-60, 0, 60, 120]
    tArray, sspSolution = create_trajectory()
    UPoincares          = [UPoincare((angle/60) * (pi/3))  for angle in Angles]
    PoincareSections    = [create_section(upoincare,tArray, sspSolution) for upoincare in UPoincares]
    PoincareSections2   = [upoincare.do_stuff(sspSolutionPoincare) for upoincare,sspSolutionPoincare in zip(UPoincares,PoincareSections)]
    PoincareSectionsS   = [upoincare.foo(X) for upoincare,X in zip(UPoincares,PoincareSections2)]

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
                label      = f'{text} {Angles[i]:4d}' + r'$^{\circ}$')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_title('Poincare Recurrences for Rossler')
    ax.legend( prop={'family': 'monospace'})
    savefig('recurrences')

    fig = figure(figsize=(12,12))
    axes = fig.subplots(nrows = 3,
                        ncols = len(Angles))
    for i in range(len(Angles)):
        axes[0][i].plot(PoincareSections2[i][:, 0], PoincareSections2[i][:, 1], styles[i],
                markersize = 5,
                label      = 'Poincare Section')
        axes[0][i].set_title(f'{Angles[i]}')
        axes[1][i].plot(PoincareSectionsS[i][:, 0], PoincareSectionsS[i][:, 1], styles[i],
                markersize = 5,
                label      = 'Poincare Section')
    show()
