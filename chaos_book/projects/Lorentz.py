from argparse          import ArgumentParser
from matplotlib.pyplot import figure, savefig, show, suptitle
from numpy             import arange, array, sqrt
from numpy.random      import rand
from os.path           import join
from scipy.integrate   import solve_ivp
from scipy.linalg      import eig, norm


sigma = 10.0
rho   = 28.0
b     = 8.0/3.0
figs  = './figs'

def velocity( t,stateVec):
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

def create_eqs():
    eq0 = [0,0,0]
    if rho<1:
        return array([eq0])
    else:
        x = sqrt(b*(rho-1))
        return array([eq0,
                     [x,x,rho-1],
                     [-x,-x,rho-1]])

if __name__ == '__main__':
    EQs           = create_eqs()
    x0            = EQs[0,:] + 0.001*rand(3)
    dt            = 0.005
    nstp          = 50.0/dt
    orbit         = integrator(x0, 50.0, nstp)

    fig = figure(figsize=(12,12))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[0,:], orbit[1,:], orbit[2,:],
            markersize = 1)
    ax.scatter(EQs[0,0], EQs[0,1], EQs[0,2], marker='o', c='xkcd:red', label='EQ0')
    ax.scatter(EQs[1,0], EQs[1,1], EQs[1,2], marker='1', c='xkcd:red', label='EQ1')
    ax.scatter(EQs[2,0], EQs[2,1], EQs[2,2], marker='2',c='xkcd:red', label='EQ2')
    ax.set_title(fr'Lorentz $\sigma=${sigma}, $\rho=${rho}, b={b}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    savefig(join(figs,'Lorentz'))
    show()
