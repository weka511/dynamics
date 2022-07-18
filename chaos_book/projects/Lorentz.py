from argparse          import ArgumentParser
from matplotlib.pyplot import figure, savefig, show, suptitle
from numpy             import arange, array, sqrt
from numpy.random      import rand
from os.path           import join, basename, split
from pathlib           import Path
from scipy.integrate   import solve_ivp
from scipy.linalg      import eig, norm

class Dynamics:

    def get_title(self):
        return fr'{self.name} $\sigma=${self.sigma}, $\rho=${self.rho}, b={self.b}'

    def get_x_label(self):
        return 'x'

    def get_y_label(self):
        return 'y'

    def get_z_label(self):
        return 'z'

sigma = 10.0
rho   = 28.0
b     = 8.0/3.0
figs  = './figs'

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

        return array([sigma * (y-x),
                      rho*x - y - x*z,
                      x*y - b*z])

class PseudoLorentz(Dynamics):
    def __init__(self,
                 sigma = 10.0,
                 rho   = 28.0,
                 b     = 8.0/3.0):
        self.sigma = sigma
        self.rho   = rho
        self.b     = b
        self.name     = 'Pseudo Lorentz'

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


    def integrate(self,init_x, dt, nstp):
        '''
        The integrator of the Lorentz system.
        init_x: the intial condition
        dt : time step
        nstp: number of integration steps.

        return : a [ nstp x 3 ] vector
        '''

        return solve_ivp(dynamics.velocity, (0, dt), init_x, t_eval=arange(0,dt,dt/nstp)).y

def stabilityMatrix(stateVec):
    '''
    return the stability matrix at a state point.
    stateVec: the state vector in the full space. [x, y, z]
    '''

    x = stateVec[0]; y = stateVec[1]; z = stateVec[2];
    # fill out the following matrix.
    stab = array([
            [-sigma, sigma, 0],
            [rho-z,  -1 , -x],
            [y,      x,   -b]
            ])

    return stab

def integrator(init_x, dt, nstp):
    '''
    The integator of the Lorentz system.
    init_x: the intial condition
    dt : time step
    nstp: number of integration steps.

    return : a [ nstp x 3 ] vector
    '''

    return solve_ivp(velocity, (0, dt), init_x, t_eval=arange(0,dt,dt/nstp)).y


def integrator_with_jacob(init_x, dt, nstp):
    '''
    integrate the orbit and the Jacobian as well. The meaning
    of input parameters are the same as 'integrator()'.

    return :
            state: a [ nstp x 3 ] state vector
            Jacob: [ 3 x 3 ] Jacobian matrix
    '''
    def JacobianVelocity(sspJacobian, t):
        ssp        = sspJacobian[0:d]                 # First three elements form the original state space vector
        J          = sspJacobian[d:].reshape((d, d))  # Last nine elements corresponds to the elements of Jacobian.
        velJ       = zeros(size(sspJacobian))         # Initiate the velocity vector as a vector of same size as sspJacobian
        velJ[0:d]  = velocity(ssp, t)
        velTangent = dot(stabilityMatrix(ssp), J)     # Velocity matrix for  the tangent space
        velJ[d:]   = reshape(velTangent, d*d)           # Last dxd elements of the velJ are determined by the action of
                                                      # stability matrix on the current value of the Jacobian:
        return velJ

    d                   = len(init_x)
    Jacobian0           = identity(d)
    sspJacobian0        = zeros(d + d ** 2)
    sspJacobian0[0:d]   = init_x
    sspJacobian0[d:]    = reshape(Jacobian0, d**2)

    sspJacobianSolution = solve_ivp(JacobianVelocity,    #FIXME
                                 arange(0, dt*nstp, dt),
                                 sspJacobian0)
    state = sspJacobianSolution[0:d]
    Jacob = sspJacobianSolution[-1, d:].reshape((d, d))

    return state, Jacob



if __name__ == '__main__':
    parser        = ArgumentParser()
    parser.add_argument('action', type=int)
    args          = parser.parse_args()
    fig = figure(figsize=(12,12))

    if args.action==1:
        dynamics      = Lorentz()
        integrator    = Integrator(dynamics)
        EQs           = dynamics.create_eqs()
        x0            = EQs[0,:] + 0.001*rand(3)
        dt            = 0.005
        nstp          = 50.0/dt
        orbit         = integrator.integrate(x0, 50.0, nstp)

        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                markersize = 1)
        ax.scatter(EQs[0,0], EQs[0,1], EQs[0,2], marker='o', c='xkcd:red', label='EQ0')
        ax.scatter(EQs[1,0], EQs[1,1], EQs[1,2], marker='1', c='xkcd:red', label='EQ1')
        ax.scatter(EQs[2,0], EQs[2,1], EQs[2,2], marker='2',c='xkcd:red', label='EQ2')
        ax.set_title(fr'Lorentz $\sigma=${sigma}, $\rho=${rho}, b={b}')
        ax.set_title(dynamics.get_title())
        ax.set_xlabel(dynamics.get_x_label())
        ax.set_ylabel(dynamics.get_y_label())
        ax.set_zlabel(dynamics.get_z_label())
        ax.legend()

    if args.action==2:
        dynamics = PseudoLorentz()
        integrator = Integrator(dynamics)
        EQs           = create_eqs()
        x =  EQs[0,0]
        y =  EQs[0,1]
        z =  EQs[0,2]
        x0            = array([x**2-y**2, 2*x*y,z]) + 0.001*rand(3)
        dt            = 0.005
        nstp          = 50.0/dt
        orbit = integrator.integrate(x0, 50.0, nstp)

        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
                markersize = 1)

        ax.set_title(dynamics.get_title())
        ax.set_xlabel(dynamics.get_x_label())
        ax.set_ylabel(dynamics.get_y_label())
        ax.set_zlabel(dynamics.get_z_label())

        ax.legend()

    savefig(join(figs,f'{Path(__file__).stem}{args.action}'))
    show()
