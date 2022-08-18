#!/usr/bin/env python

'''Poincare Sections'''

from math                   import isqrt
from matplotlib.pyplot      import figure, savefig, show, tight_layout
from numpy                  import abs, append, argmin, argsort, argwhere, array, dot, cos, linspace, max, pi, real, sin, size, zeros
from numpy.linalg           import eig, inv, norm
from re                     import split
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
    in order to include its unstable manifold on the Poincare section.
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
        self.e_x         = array([1, 0, 0], float)  # Unit vector in x-direction
        self.sspTemplate = dot(UPoincare.zRotation(thetaPoincare), self.e_x)  #Template vector to define the Poincare section hyperplane
        self.nTemplate   = dot(UPoincare.zRotation(pi/2), self.sspTemplate)  #Normal to this plane: template vector rotated pi/2 about the z axis
        self.e_z         = array([0, 0, 1], float)  # Unit vector in z-direction

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

    def ProjectPoincare(self,sspSolutionPoincare):
        '''
        sspSolutionPoincare has column vectors on its rows. We act on the
        transpose of this matrix to project each state space point onto Poincare
        basis by a simple matrix multiplication:
        '''
        self.ProjPoincare = array([self.sspTemplate,
                              self.e_z,
                              self.nTemplate],
                             float)

        PoincareSection = dot(self.ProjPoincare, sspSolutionPoincare.transpose())
        return  PoincareSection.transpose()   #We return to the usual N x 3 form by another transposition


    def SortPoincareSection(self,PoincareSection):
        Distance              = squareform(pdist(PoincareSection))
        self.SortedPoincareSection = PoincareSection.copy()
        ArcLengths            = zeros(size(self.SortedPoincareSection, 0)) #arclengths of the Poincare section points after sorting:
        self.sn               = zeros(size(PoincareSection, 0)) # arclengths of the Poincare section
        for k in range(size(self.SortedPoincareSection, 0) - 1):
            m                               = argmin(Distance[k, k + 1:]) + k + 1 # element which is closest to the kth point
            dummyPoincare                   = self.SortedPoincareSection[k + 1, :].copy() # #Hold the (k+1)th row in the dummy vector:
            self.SortedPoincareSection[k + 1, :] = self.SortedPoincareSection[m, :] #Replace (k+1)th row with the closest point:
            self.SortedPoincareSection[m, :]     = dummyPoincare #Assign the previous (k+1)th row to the mth row:

            dummyColumn        = Distance[:, k + 1].copy()  # Hold (k+1)th column of the distance matrix in a dummy array
            Distance[:, k + 1] = Distance[:, m]  # Assign mth column to kth
            Distance[:, m]     = dummyColumn

            dummyRow           = Distance[k + 1, :].copy()  # Hold (k+1)th row in a dummy array
            Distance[k + 1, :] = Distance[m, :]
            Distance[m, :]     = dummyRow

            ArcLengths[k + 1] = ArcLengths[k] + Distance[k, k + 1] #Assign the arclength of (k+1)th element:
            #Find this point in the PoincareSection array and assign sn to its
            #corresponding arclength:
            self.sn[argwhere(PoincareSection[:, 0] == self.SortedPoincareSection[k + 1, 0])] = ArcLengths[k + 1]

        self.tckPoincare, u = splprep([self.SortedPoincareSection[:, 0],
                                  self.SortedPoincareSection[:, 1]],
                                 u = ArcLengths,
                                 s = 0)
        # def fPoincare(s):
            # interpolation = splev(s, tckPoincare)
            # return array([interpolation[0], interpolation[1]], float).transpose()

        self.sArray                      = linspace(min(ArcLengths), max(ArcLengths), 1000)
        self.InterpolatedPoincareSection = self.fPoincare(self.sArray)

        self.sn1 = self.sn[0:-1]
        self.sn2 = self.sn[1:]

        #In order to be able to interpolate to the data, we need to arrange it such
        #that the x-data is from the smallest to the largest:

        #Indices on the order of which the sn1 is sorted from its smallest to the
        #largest element:

        isort = argsort(self.sn1)
        self.sn1  = self.sn1[isort]  # sort radii1
        self.sn2  = self.sn2[isort]  # sort radii2

        # Construct tck of the spline rep
        self.tckReturn = splrep(self.sn1,self.sn2)
        self.snPlus1 = splev(self.sArray, self.tckReturn)  # Evaluate spline

    def fPoincare(self,s):
        interpolation = splev(s, self.tckPoincare)
        return array([interpolation[0], interpolation[1]], float).transpose()

def get_crossing(sspSolution,i,upoincare,tArray):
    '''
    find where exactly the flow pierces the Poincare section:
    '''
    return Flow(sspSolution[i],
                fsolve(lambda deltat: upoincare.UPoincare(Flow(sspSolution[i] , deltat)),
                                 (tArray[i + 1] - tArray[i]) / 2) )


def get_all_crossings(upoincare,tArray, sspSolution):

    return array([get_crossing(sspSolution,i,upoincare,tArray)
                  for i in range(size(sspSolution, 0) - 1)
                  if upoincare.UPoincare(sspSolution[i]) < 0 and upoincare.UPoincare(sspSolution[i+1]) > 0],
                 float)

def get_angle_as_text(angle,text=''):
    '''Format angle for display'''
    return f'{text} {angle:4d}' + r'$^{\circ}$'

def generate_subplots(n,
                  name       = 'figure',
                  width      = 12,
                  height     = 12,
                  title      = None,
                  projection = None):
    '''
    Create a matrix of subplots, iterate through them, and save figure

    Parameters:
        n           Number of subplots
        name        Used to save plot as file
        width       Width of figure in inches
        height      Height of figure in inches
        title       Supertitle shared by all subplots
        projection  Used to specify projection for each subplot
    '''
    fig = figure(figsize=(width,height))
    if title != None:
        fig.suptitle(title)
    nrows = isqrt(n)
    ncols = n//nrows
    while nrows*ncols<n:
        ncols += 1
    axes = fig.subplots(nrows      = nrows,
                        ncols      = ncols,
                        subplot_kw = dict(projection=projection))
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if k < n:
                yield axes[i][j],k
                if k==0:
                    ax.legend()
                k += 1
            else:
                axes[i][j].set_axis_off()

    tight_layout(h_pad = 3.0) # ensure label and title don't overlap
    savefig(name)

class CycleFinder:
    '''
    This class finds periodic orbits
    '''
    def __init__(self,UPoincare,PoincareSection,tFinal=1000):
        self.UPoincare = UPoincare
        self.tFinal    = tFinal
        self.PoincareSection = PoincareSection

    def get_fixed_solution(self):
        ReturnMap            = lambda r: splev(r, self.UPoincare.tckReturn) - r
        sfixed               = fsolve(ReturnMap, 10.0)[0]
        PoincareSectionfixed = self.UPoincare.fPoincare(sfixed)
        self.sspfixed        = dot(append(PoincareSectionfixed, 0.0), self.UPoincare.ProjPoincare)
        fdeltat              = lambda deltat: self.UPoincare.UPoincare(Flow(self.sspfixed, deltat))
        Tguess               = self.tFinal / size(self.PoincareSection, 0)
        self.Tnext           = fsolve(fdeltat, Tguess)[0]
        tArray               = linspace(0, self.Tnext, 100)
        return odeint(Velocity, self.sspfixed, tArray)

    def refine(self,
               tolerance  = 1e-9,
               kmax       = 20):
        period     = self.Tnext.copy()
        error      = zeros(4)
        Delta      = zeros(4)
        error[0:3] = Flow(self.sspfixed, period) - self.sspfixed
        Newton     = zeros((4, 4))
        k          = 0

        while max(abs(error)) > tolerance:
            if k > kmax: raise Exception(f'Could not reduce error below {tolerance} within {kmax} iterations')
            k += 1
            Newton[0:3, 0:3] = 1 - Jacobian(self.sspfixed,self.Tnext)
            Newton[0:3, 3]   = -Velocity(self.sspfixed,self.Tnext)
            Newton[3, 0:3]   = self.UPoincare.nTemplate
            Delta            = dot(inv(Newton), error)
            self.sspfixed    = self.sspfixed + Delta[0:3]
            period           += Delta[3]
            error[0:3]       = Flow(self.sspfixed, period) - self.sspfixed
        return period, odeint(Velocity, self.sspfixed, linspace(0, period, 1000))

def create_xkcd_colours(file_name = 'rgb.txt',
                        prefix    = 'xkcd:',
                        filter    = lambda R,G,B:True):
    '''  Create list of XKCD colours
         Keyword Parameters:
            file_name Where XKCD colours live
            prefix    Use to prefix each colour with "xkcd:"
            filter    Allows us to exclude some colours based on RGB values
    '''
    with open(file_name) as colours:
        for row in colours:
            parts = split(r'\s+#',row.strip())
            if len(parts)>1:
                rgb  = int(parts[1],16)
                B    = rgb%256
                rest = (rgb-B)//256
                G    = rest%256
                R    = (rest-G)//256
                if filter(R,G,B):
                    yield f'{prefix}{parts[0]}'

if __name__=='__main__':
    tFinal              = 1000
    Angles              = [0, 45, 90, 135]
    colours             = [colour for colour in create_xkcd_colours()][::-1]
    tArray, sspSolution = create_trajectory(tFinal=tFinal)

    fig                 = figure(figsize=(12,12))
    ax                  = fig.gca(projection='3d')
    ax.plot(sspSolution[:, 0], sspSolution[:, 1], sspSolution[:, 2],
            linewidth = 0.1,
            color     = colours[0],
            label     = 'Rossler')

    UPoincares          = [UPoincare((angle/60) * (pi/3))  for angle in Angles]
    Crossings           = [get_all_crossings(upoincare,tArray, sspSolution) for upoincare in UPoincares]
    PoincareSections    = [upoincare.ProjectPoincare(sspSolutionPoincare) for upoincare,sspSolutionPoincare in zip(UPoincares,Crossings)]

    for i in range(len(Angles)):
        text = 'Section' if i==0 else '   "   '
        ax.plot(Crossings[i][:, 0],
                Crossings[i][:, 1],
                Crossings[i][:, 2],
                marker     = 'o',
                color      = colours[i+1],
                linestyle  = '',
                markersize = 2,
                label      = get_angle_as_text(Angles[i],text=text))

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_title('Poincare Recurrences for Rossler')
    ax.legend( prop={'family': 'monospace'})
    fig.tight_layout(h_pad = 3.0)
    savefig('recurrences')

    for upoincare,PoincareSection in zip(UPoincares,PoincareSections):
        upoincare.SortPoincareSection(PoincareSection)
    SortedPoincareSection        = [upoincare.SortedPoincareSection for upoincare in UPoincares]
    InterpolatedPoincareSections = [upoincare.InterpolatedPoincareSection for upoincare in UPoincares]

    for ax,k in generate_subplots(len(Angles),
                              title = 'Poincare Sections',
                              name = 'sections'):
        ax.plot(PoincareSections[k][:, 0], PoincareSections[k][:, 1],
                marker     = 'o',
                color      = 'xkcd:red',
                linestyle  = '',
                markersize = 8,
                label      = 'Poincare Sections')
        ax.set_title(get_angle_as_text(Angles[k]))
        ax.plot(SortedPoincareSection[k][:, 0], SortedPoincareSection[k][:, 1],
                marker     = 'o',
                color      = 'xkcd:blue',
                linestyle  = '',
                markersize = 4,
                label      = 'Sorted Poincare Section')
        ax.plot(InterpolatedPoincareSections[k][:, 0], InterpolatedPoincareSections[k][:, 1],
                marker     = 'o',
                color      = 'xkcd:green',
                linestyle  = '',
                markersize = 1,
                label      = 'Interpolated Poincare Section')
        ax.set_xlabel(r'$\hat{x}$')
        ax.set_ylabel(r'$\hat{y}$')

    sn1s                  = [upoincare.sn1 for upoincare in UPoincares]
    sn2s                  = [upoincare.sn2 for upoincare in UPoincares]
    sArrays               = [upoincare.sArray for upoincare in UPoincares]
    snPlus1s              = [upoincare.snPlus1 for upoincare in UPoincares]


    for ax,k in generate_subplots(len(Angles),
                              title = 'Return maps',
                              name  = 'return'):
        ax.set_aspect('equal')
        ax.set_title(get_angle_as_text(Angles[k]))
        ax.plot(sn1s[k], sn2s[k],
                marker     = 'o',
                color      = 'xkcd:red',
                linestyle  = '',
                markersize = 8,
                label='Return map')
        ax.plot(sArrays[k], snPlus1s[k],
                marker     = 'o',
                color      = 'xkcd:blue',
                linestyle  = '',
                markersize=1,
                label = 'Interpolated')
        ax.plot(sArrays[k], sArrays[k],
                marker     = '',
                color      = 'xkcd:black',
                linestyle  = '--')
        ax.set_xlabel('$s_n$')
        ax.set_ylabel('$s_{n+1}$')

    finder                = CycleFinder(UPoincares[0],PoincareSections[0], tFinal = tFinal)
    sspfixedSolution      = finder.get_fixed_solution()
    period, periodicOrbit = finder.refine()

    print("Shortest periodic orbit is at: ", periodicOrbit[0][0],
                                             periodicOrbit[0][1],
                                             periodicOrbit[0][2])
    print("Period:", period)

    fig = figure(figsize=(12,12))
    ax  = fig.gca(projection='3d')
    fig.suptitle(f'Periodic Orbit, period={period}')
    ax.set_title(f'Starting point: ({periodicOrbit[0][0]}, {periodicOrbit[0][1]}, {periodicOrbit[0][2]})')
    ax.plot(sspfixedSolution[:, 0],
            sspfixedSolution[:, 1],
            sspfixedSolution[:, 2],
            color='xkcd:green',
            label='From Poincare Section')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.plot(periodicOrbit[:, 0],
            periodicOrbit[:, 1],
            periodicOrbit[:, 2],
            color='xkcd:purple',
            label='Newton')

    ax.scatter(periodicOrbit[0][0],
               periodicOrbit[0][1],
               periodicOrbit[0][2])

    ax.legend()
    savefig('periodicorbits')

    show()
