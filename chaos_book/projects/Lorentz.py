'''Investigation of Lorentz Equations'''
from argparse          import ArgumentParser
from matplotlib.pyplot import figure, rcParams, savefig, show, suptitle
from numpy             import arange, array, cos, dot, pi, sin, sqrt
from numpy.random      import rand
from os.path           import join, basename, split
from pathlib           import Path
from scipy.integrate   import solve_ivp
from scipy.linalg      import eig, norm
from scipy.optimize    import fsolve

class Dynamics:

    def get_title(self):
        return fr'{self.name} $\sigma=${self.sigma}, $\rho=${self.rho}, b={self.b}'

    def get_x_label(self):
        return 'x'

    def get_y_label(self):
        return 'y'

    def get_z_label(self):
        return 'z'


class Lorentz(Dynamics):
    def __init__(self,
                 sigma = 10.0,
                 rho   = 28.0,
                 b     = 8.0/3.0):
        self.sigma = sigma
        self.rho   = rho
        self.b     = b
        self.name  = 'Lorentz'

    def create_eqs(self):
        eq0 = [0,0,0]
        if self.rho<1:
            return array([eq0])
        else:
            x = sqrt(self.b*(self.rho-1))
            return array([eq0,
                         [x,x,self.rho-1],
                         [-x,-x,self.rho-1]])

    def velocity(self, t,stateVec):
        '''
        return the velocity field of Lorentz system.
        stateVec : the state vector in the full space. [x, y, z]
        t : time is used since solve_ivp() requires it.
        '''

        x = stateVec[0]
        y = stateVec[1]
        z = stateVec[2]

        return array([self.sigma * (y-x),
                      self.rho*x - y - x*z,
                      x*y - self.b*z])

class PseudoLorentz(Dynamics):
    def __init__(self,
                 sigma = 10.0,
                 rho   = 28.0,
                 b     = 8.0/3.0):
        self.sigma = sigma
        self.rho   = rho
        self.b     = b
        self.name     = 'Pseudo Lorentz'

    def create_eqs(self):
        eq0 = [0,0,0]
        eq1 = [0,2*self.b*(self.rho-1),self.rho-1]
        return  array([eq0,eq1])

    def velocity(self,t,stateVec):
        u = stateVec[0]
        v = stateVec[1]
        z = stateVec[2]
        N = sqrt(u**2 + v**2)
        return array([-(self.sigma+1)*u + (self.sigma-self.rho)*v + (1-self.sigma)*N + v*z,
                      (self.rho-self.sigma)*u - (self.sigma+1)*v + (self.rho+self.sigma)*N - u*z -u*N,
                      v/2 - self.b*z])

    def get_x_label(self):
        return 'u'

    def get_y_label(self):
        return 'v'

class Integrator:
    def __init__(self,dynamics):
        self.dynamics = dynamics


    def integrate(self,init_x, dt, nstp=1):
        '''
        The integrator of the Lorentz system.
        init_x: the intial condition
        dt : time step
        nstp: number of integration steps.

        return : a [ nstp x 3 ] vector
        '''

        bunch = solve_ivp(dynamics.velocity, (0, dt), init_x, t_eval=arange(0,dt,dt/nstp))
        if bunch.status==0:
            return bunch.t, bunch.y
        else:
            raise(Exception(f'{bunch.status} {bunch.message}'))



    def Flow(self,deltat,y):
        bunch = solve_ivp(dynamics.velocity, (0, deltat), y)
        return bunch.t[1],bunch.y[:,1]

class PoincareSection:
    ''' This class represents a Poincare Section'''
    @staticmethod
    def zRotation(theta):
        '''
        Rotation matrix about z-axis
        Input:
        theta: Rotation angle (radians)
        Output:
        Rz: Rotation matrix about z-axis
        '''
        return array([[cos(theta), -sin(theta), 0],
                      [sin( theta), cos(theta),  0],
                      [0,          0,           1]],
                     float)

    def __init__(self,dynamics,integrator,
                 sspTemplate = None,
                 nTemplate   = None,
                 theta       = 0.0,
                 e_x         = array([1, 0, 0], float)):
        self.sspTemplate = dot(PoincareSection.zRotation(theta), e_x)  if len(sspTemplate)==1 else sspTemplate
        self.nTemplate   = dot(PoincareSection.zRotation(pi/2), self.sspTemplate) if len(nTemplate)==1 else nTemplate
        self.integrator  = integrator
        self.dynamics    = dynamics

    def U(self, ssp):
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
        return dot((ssp - self.sspTemplate),self.nTemplate)

    def Flow(self,y0,dt):
        _,y = self.integrator.integrate(y0,dt)
        return y[0]

    def interpolate(self,dt0, y0):
        return self.integrator.Flow(fsolve(lambda t: self.U(self.Flow(y0, t)), dt0)[0],
                                    y0)


    def interections(self,ts, orbit):
        _,n = orbit.shape
        for i in range(n-1):
            if self.U(orbit[:,i])<0 and self.U(orbit[:,i+1])>0:
                yield self.interpolate(0.5*(ts[i+1]-ts[i]), orbit[:,i])

def plot_poincare(ax,section,ts,orbit,s=1):
    for t,point in section.interections(ts,orbit):
        ax.scatter(point[0],point[1],point[2],
                   c      = 'xkcd:green',
                   s      = s,
                   marker = 'o')

    ax.scatter(point[0],point[1],point[2],
               c      = 'xkcd:green',
               s      = s,
               label  = r'Poincar\'e return',
               marker = 'o')


if __name__ == '__main__':
    rcParams['text.usetex'] = True
    parser                  = ArgumentParser(description = __doc__)
    parser.add_argument('action', type = int)
    parser.add_argument('--figs', default = './figs')
    args          = parser.parse_args()
    fig = figure(figsize=(12,12))

    if args.action==1:
        dynamics      = Lorentz()
        integrator    = Integrator(dynamics)
        EQs           = dynamics.create_eqs()
        section       = PoincareSection(dynamics,integrator,
                                        sspTemplate = EQs[2],
                                        nTemplate   = array([1,0,0]))

        x0            = EQs[0,:] + 0.001*rand(3)
        dt            = 0.005
        nstp          = 50.0/dt
        ts,orbit      = integrator.integrate(x0, 50.0, nstp)

        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                c          = 'xkcd:blue',
                label      = 'Orbit',
                markersize = 1)
        ax.scatter(EQs[0,0], EQs[0,1], EQs[0,2], marker='o', c='xkcd:red', label='EQ0')
        ax.scatter(EQs[1,0], EQs[1,1], EQs[1,2], marker='1', c='xkcd:red', label='EQ1')
        ax.scatter(EQs[2,0], EQs[2,1], EQs[2,2], marker='2', c='xkcd:red', label='EQ2')
        plot_poincare(ax,section,ts,orbit)
        ax.set_title(dynamics.get_title())
        ax.set_xlabel(dynamics.get_x_label())
        ax.set_ylabel(dynamics.get_y_label())
        ax.set_zlabel(dynamics.get_z_label())

    if args.action==2:
        dynamics   = PseudoLorentz()
        integrator = Integrator(dynamics)
        EQs        = dynamics.create_eqs()
        section    = PoincareSection(dynamics,integrator,
                                        sspTemplate = EQs[1],
                                        nTemplate   = array([1,0,0]))
        x0         = EQs[0,:] + 0.001*rand(3)
        dt         = 0.001
        nstp       = 50.0/dt
        ts,orbit   = integrator.integrate(x0, 50.0, nstp)

        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                markersize = 1,
                c          = 'xkcd:blue',
                label      = 'Orbit')
        ax.scatter(EQs[0,0], EQs[0,1], EQs[0,2], marker='o', c='xkcd:red', label='EQ0')
        ax.scatter(EQs[1,0], EQs[1,1], EQs[1,2], marker='1', c='xkcd:red', label='EQ1')
        plot_poincare(ax,section,ts,orbit)
        ax.set_title(dynamics.get_title())
        ax.set_xlabel(dynamics.get_x_label())
        ax.set_ylabel(dynamics.get_y_label())
        ax.set_zlabel(dynamics.get_z_label())

    ax.legend()
    savefig(join(args.figs,f'{Path(__file__).stem}{args.action}'))
    show()
