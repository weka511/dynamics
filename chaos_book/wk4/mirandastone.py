from numpy             import array, linspace, sqrt
from matplotlib.pyplot import figure, show
from scipy.integrate   import odeint

sigma = 10
beta  = 8/3
rho   = 28

def Velocity(stateVec, t):
    u = stateVec[0]
    v = stateVec[1]
    z = stateVec[2]
    N = sqrt(u*u + v*v)

    return array(
        [
            -(sigma+1)* u + (sigma-rho)*v + (1-sigma)*N + v*z,
            (rho-sigma)*u - (sigma+1)*v + (rho+sigma)*N -(u+N)*z,
            v/2-beta*z
            ],
        float)

if __name__=='__main__':
    tInitial = 0
    tFinal   = 100
    Nt       = 100000

    tArray   = linspace(tInitial, tFinal, Nt)  # Time array for solution
    ssp0     = array([10,
                      0.0,
                      3], float)  # Initial condition for the solution

    sspSolution = odeint(Velocity, ssp0, tArray)
    ut = sspSolution[:, 0]
    vt = sspSolution[:, 1]
    zt = sspSolution[:, 2]

    fig = figure(figsize=(12,12))
    ax  = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot(ut, vt, zt, color='xkcd:purple', linewidth=0.5)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('z')
    ax.set_title('Miranda & Stone proto-Lorentz')
    show()
