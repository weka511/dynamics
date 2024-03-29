from numpy             import array, dot, identity, linspace, reshape, size, zeros
from scipy.integrate   import odeint
from matplotlib.pyplot import figure, show

#Parameters:
a = 0.2
b = 0.2
c = 5.7


def Velocity(ssp, t):
    '''
    Velocity function for the Rossler flow

    Inputs:
    ssp: State space vector. dx1 NumPy array: ssp=[x, y, z]
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
        vel: velocity at ssp. dx1 NumPy array: vel = [dx/dt, dy/dt, dz/dt]
    '''

    x, y, z = ssp

    dxdt = - y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)

    return array([dxdt, dydt, dzdt], float)  # Velocity vector


def Flow(ssp0, deltat):
    '''
    Lagrangian description of the flow:
    This function integrates Rossler equation starting at ssp0 for deltat, and
    returns the final state space point.

    Parameters:
        ssp0: Initial state space point
        deltat: Integration time

    Returns:
        sspdeltat: Final state space point
    '''

    sspSolution = odeint(Velocity, ssp0, [0.0, deltat])     # Compute a 2 by 3(=d) solution array whose first row contains initial point ssp0,
                                                            # and the last row contains final point
    return sspSolution[-1, :]  # Read the final point to sspdeltat



def StabilityMatrix(ssp):
    '''
    Stability matrix for the Rossler flow

    Inputs:
        ssp: State space vector. dx1 NumPy array: ssp = [x, y, z]
    Outputs:
        A: Stability matrix evaluated at ssp. dxd NumPy array
           A[i, j] = del Velocity[i] / del ssp[j]
    '''

    x, y, z = ssp  # Read state space points


    return array([[0, -1, -1],
                  [1, a, 0],
                  [z, 0, x-c]],
                 float)


def JacobianVelocity(sspJacobian, t):
    '''
    Velocity function for the Jacobian integration

    Inputs:
        sspJacobian: (d+d^2)x1 dimensional state space vector including both the
                     state space itself and the tangent space
        t: Time. Has no effect on the function, we have it as an input so that our
           ODE would be compatible for use with generic integrators from
           scipy.integrate

    Outputs:
        velJ = (d+d^2)x1 dimensional velocity vector
    '''

    ssp        = sspJacobian[0:3]                 # First three elements form the original state space vector
    J          = sspJacobian[3:].reshape((3, 3))  # Last nine elements corresponds to the elements of Jacobian.
    velJ       = zeros(size(sspJacobian))         # Initiate the velocity vector as a vector of same size as sspJacobian
    velJ[0:3]  = Velocity(ssp, t)
    velTangent = dot(StabilityMatrix(ssp), J)     # Velocity matrix for  the tangent space
    velJ[3:]   = reshape(velTangent, 9)           # Last dxd elements of the velJ are determined by the action of
                                                  # stability matrix on the current value of the Jacobian:
    return velJ


def Jacobian(ssp, t,
             Nt = 500):
    '''
    Jacobian function for the trajectory started on ssp, evolved for time t

    Inputs:
        ssp: Initial state space point. dx1 NumPy array: ssp = [x, y, z]
        t: Integration time
    Outputs:
        J: Jacobian of trajectory f^t(ssp). dxd NumPy array
    '''

    Jacobian0 = identity(3)
    # Initial condition for Jacobian integral is a d+d^2 dimensional matrix
    # formed by concatenation of initial condition for state space and the Jacobian:
    sspJacobian0        = zeros(3 + 3 ** 2)  # Initiate
    sspJacobian0[0:3]   = ssp  # First 3 elemenets
    sspJacobian0[3:]    = reshape(Jacobian0, 9)  # Remaining 9 elements
    tInitial            = 0
    tFinal              = t

    tArray              = linspace(tInitial, tFinal, Nt)  # Time array for solution

    sspJacobianSolution = odeint(JacobianVelocity, sspJacobian0, tArray)

    return sspJacobianSolution[-1, 3:].reshape((3, 3))

if __name__ == '__main__':

    tInitial = 0  # Initial time
    tFinal   = 250  # Final time
    Nt       = 25000  # Number of time points to be used in the integration

    tArray   = linspace(tInitial, tFinal, Nt)  # Time array for solution
    ssp0     = array([1.0,
                      1.0,
                      1.0], float)  # Initial condition for the solution

    sspSolution = odeint(Velocity, ssp0, tArray)

    xt = sspSolution[:, 0]
    yt = sspSolution[:, 1]
    zt = sspSolution[:, 2]

    print((xt[-1], yt[-1], zt[-1]))  # Print final point

    fig = figure()
    ax  = fig.add_subplot(1,1,1,projection='3d')
    ax.plot(xt, yt, zt)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Rossler')
    ssp1    = array([9.269082847348976, 1.1000467728916225e-22, 2.58159277507681], float)  # Initial condition for the solution

    sspSolution = odeint(Velocity, ssp1, linspace(tInitial, 5.9, 1000))

    xt = sspSolution[:, 0]
    yt = sspSolution[:, 1]
    zt = sspSolution[:, 2]
    ax.plot(xt, yt, zt,'.r')
    show()
