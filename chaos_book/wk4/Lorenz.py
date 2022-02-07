from numpy             import arange, array, dot,  identity, reshape, size, zeros, zeros_like
from matplotlib.pyplot import figure, show, suptitle
from numpy.random      import rand
from scipy.integrate   import odeint
from scipy.linalg      import eig, norm
from argparse          import ArgumentParser

sigma = 10.0
rho   = 28.0
b     = 8.0/3.0


C2 = array([ #  C^{1/2} operation matrix for Lorenz system.
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])


def velocity(stateVec, t):
    '''
    return the velocity field of Lorentz system.
    stateVec : the state vector in the full space. [x, y, z]
    t : time is used since odeint() requires it.
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

    return odeint(velocity, init_x, arange(0, dt*nstp, dt))


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

    sspJacobianSolution = odeint(JacobianVelocity,
                                 sspJacobian0,
                                 arange(0, dt*nstp, dt))
    state = sspJacobianSolution[0:d]
    Jacob = sspJacobianSolution[-1, d:].reshape((d, d))

    return state, Jacob

def reduceSymmetry(states):
    '''
    reduce C^{1/2} symmetry of Lorenz system by invariant polynomials.
    (x, y, z) -> (u, v, z) = (x^2 - y^2, 2xy, z)

    states: trajectory in the full state space. dimension [nstp x 3]
    return: states in the invariant polynomial basis. dimension [nstp x 3]
    '''

    x,y,z              = states[:,0], states[:,1], states[:,2]
    (u, v)             = (x*2 - y*2, 2*x*y)
    reducedStates      = zeros_like(states)
    reducedStates[:,0] = u
    reducedStates[:,1] = v
    reducedStates[:,2] = z
    return reducedStates

def plotFig(orbit):
    fig = figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
    show()

def plot_orbits(orbit,reduced_orbit,case=None):
    fig = figure(figsize=(20,20))
    suptitle(f'Case {case}')
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(orbit[:,0], orbit[:,1], orbit[:,2],
            markersize = 1)
    ax.set_title('Orbit')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title('Reduced')
    ax.plot(reduced_orbit[:,0], reduced_orbit[:,1], reduced_orbit[:,2],
            markersize = 1)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('z')
    show()

if __name__ == '__main__':
    parser = ArgumentParser('Q4.3 Symmetry of Lorenz Flow')
    parser.add_argument('case',
                        type    = int,
                        choices = [1,2,3,4])
    args = parser.parse_args()

    # case 1: try a random initial condition
    if args.case == 1:
        x0            = rand(3)
        dt            = 0.005
        nstp          = 50.0/dt
        orbit         = integrator(x0, dt, nstp)
        reduced_orbit = reduceSymmetry(orbit)
        plot_orbits(orbit,reduced_orbit,case=args.case)


    # case 2: periodic orbit
    if args.case == 2:
        x0            = array([ -0.78844208,  -1.84888176,  18.75036186])
        dt            = 0.0050279107820829149
        nstp          = 156
        orbit_double  = integrator(x0, dt, nstp*2)
        orbit         = orbit_double[:nstp, :] # one prime period
        reduced_orbit = reduceSymmetry(orbit)
        plot_orbits(orbit,reduced_orbit,case=args.case)

    # case 3 : calculate Floquet multipliers and Floquet vectors associated
    # with the full periodic orbit in the full state space.
    if args.case == 3:
        x0                        = array([ -0.78844208,  -1.84888176,  18.75036186])
        dt                        = 0.0050279107820829149 # integration time step
        nstp                      = 156 # number of integration steps => T = nstp * dt
        state, Jacob              = integrator_with_jacob(x0, dt, 2*nstp)
        eigenValues, eigenVectors = eig(Jacob)
        vel                       = velocity(state[-1],2*nstp)
        vel                      /= norm(vel)
        print (f'Eigenvalues: {eigenValues}')
        print (f'Gradient,{vel}')
        # Check that one of Floquet vectors is in the same/opposite
        # direction with velocity field at x0.
        for i in range(len(eigenValues)):
            print (eigenVectors[:,i], norm(eigenVectors[:,i]), dot(eigenVectors[:,i],vel))


    # case 4: calculate Floquet multipliers and Floquet vectors associated
    # with the prime period.
    if args.case == 4:
        C                         = array([[-1, 0, 0],
                                           [0,  -1, 0],
                                           [0,  0, 1]])
        x0                        = array([ -0.78844208,  -1.84888176,  18.75036186])
        dt                        = 0.0050279107820829149
        nstp                      = 156
        state, Jacob              = integrator_with_jacob(x0, dt, nstp)
        eigenValues, eigenVectors = eig(dot(C,Jacob))
        print (eigenValues)
