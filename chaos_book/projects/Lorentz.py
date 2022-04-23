from numpy             import arange, array
from matplotlib.pyplot import figure, show, suptitle
from numpy.random      import rand
from scipy.integrate   import odeint
from scipy.linalg      import eig, norm
from argparse          import ArgumentParser

sigma = 10.0
rho   = 28.0
b     = 8.0/3.0

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

if __name__ == '__main__':
    x0            = rand(3)
    dt            = 0.005
    nstp          = 50.0/dt
    orbit         = odeint(velocity, x0, arange(0, dt*nstp, dt))

    fig = figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[:,0], orbit[:,1], orbit[:,2],
            markersize = 1)
    show()
