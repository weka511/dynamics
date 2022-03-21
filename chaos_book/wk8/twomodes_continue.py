######################################################################
#
# This template code goes through the process of refining an initial
# guess for a periodic orbit after obtaining the return map. We
# continue to use two modes system as our model.
#
# Please set case = 1 and understand the return map. Then fill out
# function shootingMatrix() in file mutlishooting.py, and finish
# case 2.
#
# Note:
# 1 we choose to use our implementation of 4th order Runge-Kutta mthods
#   to integrate two modes system instead of the built-in method odeint.
#   Have a look at integrator_reduced() and integrator_reduced_with_jacob()
#   in this file.
# 2 You do not need to import any function from previous code in hw5.
#   The needed function velocity_reduced() and  stabilityMatrix_reduced()
#   is implemented for you.

######################################################################
from numpy             import arange, argmax, argmin,  array, dot, hstack,  int, load, size,  vstack, zeros
from numpy.linalg      import norm
from matplotlib.pyplot import figure, show
from scipy.interpolate import interp1d
from scipy.optimize    import fsolve
from multishooting     import Multishooting
from unimodal          import Unimodal
from argparse          import ArgumentParser


G_mu1 = -2.8
G_c1  = -7.75
G_a2  = -2.66

def rk4(velo, y0, dt, nstp):
    '''
    4th order Runge-Kutta method
    '''
    y      = zeros( (nstp, size(y0)) )
    y[0,:] = y0
    yt     = y0
    for i in range(1, nstp):
        k1 = velo(yt, None)
        k2 = velo(yt+0.5*dt*k1, None)
        k3 = velo(yt+0.5*dt*k2, None)
        k4 = velo(yt+dt*k3, None)

        yt = yt + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        y[i, :] = yt

    return y

def velocity_reduced(stateVec_reduced, t):
    '''
    velocity in the slice after reducing the continous symmetry

    stateVec_reduced: state vector in slice [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    t: not used
    return: velocity at stateVect_reduced. dimension [1 x 3]
    '''
    x1 = stateVec_reduced[0]
    x2 = stateVec_reduced[1]
    y2 = stateVec_reduced[2]

    velo = array([
            (G_mu1-x1**2)*x1 + G_c1*x1*x2,
            x2 + y2 + x1**2 + G_a2*x2*x1**2 + 2*G_c1*y2**2,
            -x2 + y2 + G_a2*y2*x1**2 - 2*G_c1*x2*y2
            ])

    return velo

def stabilityMatrix_reduced(stateVec_reduced):
    '''
    calculate the stability matrix on the slice

    stateVec_reduced: state vector in slice [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    return: stability matrix. Dimension [3 x 3]
    '''
    x1 = stateVec_reduced[0]
    x2 = stateVec_reduced[1]
    y2 = stateVec_reduced[2]

    stab = array([
            [-3*x1**2 + G_mu1 + G_c1*x2, G_c1*x1, 0],
            [2*x1 + 2*G_a2*x1*x2, 1+G_a2*x1**2, 1+4*G_c1*y2],
            [2*G_a2*x1*y2, -1-2*G_c1*y2, 1+G_a2*x1**2-2*G_c1*x2]
            ])
    return stab

def integrator_reduced(init_state, dt, nstp):
    '''
    integrate two modes system in the slice

    init_state: initial state [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    dt: time step
    nstp: number of time step
    '''

    return rk4(velocity_reduced, init_state, dt, nstp)

def integrator_reduced_with_jacob(stateVec_reduced, dt, nstp):
    def vTotal(y, t):
        x = y[0:3]
        J = y[3:].reshape(3,3)
        v = velocity_reduced(x, None)
        AJ = dot(stabilityMatrix_reduced(x), J)
        return hstack((v, AJ.reshape(9)))

    init_J = eye(3)
    y0     = hstack( (stateVec_reduced, init_J.reshape(9)))
    y      = rk4(vTotal, y0, dt, nstp)
    state  = y[:, 0:3];
    Jacob  = y[-1, 3:].reshape(3,3)

    return state, Jacob

TBD = None

if __name__ == '__main__':
    parser = ArgumentParser('8.1/Q8.2|two modes system continued -- kneading theory')
    parser.add_argument('case',
                        type    = int,
                        choices = [1,2])
    args = parser.parse_args()

    '''
    pre action : load data

    In homework 5, we get the return map on the poincare section in the two modes
    system. This map could be used to locate periodic orbits. Here, we provide
    the raw data of this map.

    Note: the poincare intersection points are recorded in the new coordinates system.
          They should be transformed to the original coordinates to get the inital guess
          of periodic orbit
    '''
    data           = load('data.npz')
    PoincarePoints = data['PoincarePoints'] # Poincare intersection points. dimension [1382 x 2]
    arclength      = data['arclength']      # arclength
    time           = data['time']           # the time stamp for each Poincare intersection point
    req            = data['req']            # relative equilibrium
    Px             = data['Px']             # the Px, Py, Pz axes of the new coordinate system
    Py             = data['Py']
    Pz             = data['Pz']

    if args.case == 1:
        '''
        work with the symbolic dynamics of two modes system.
        You are supposed to get the initial condition of periodic
        orbit with period 4.
        '''
        # interpolate the data points to get a smooth return map
        returnMap = interp1d(arclength[:-1], arclength[1:])

        # locate the critical point in a crude way
        x = arange(0.0001,1.8,0.0001)
        y = returnMap(x)
        C = x[argmax(y)] # critical point

        # get the kneading sequence of this map
        uni = Unimodal(returnMap, C)
        kneading_sequence = uni.future_symbol(C, 10)
        print (C)
        print (kneading_sequence)

        # find a periodic orbit with period 4
        order = 4
        g = lambda x : uni.returnMap_iter(x, order)[-1] - x
        guess =  TBD# choose a guess
        # use fsolve() to get the periodic orbit 1110
        state = TBD # the x coordinate of the point of orbit 1110
        # find the closest Poincare intersection point
        idx   = argmin(abs(arclength - state))
        point = PoincarePoints[idx]

        # transform the point to the original coordinate
        # Note: Poincare section is Px = 0
        original_point = point[0]*Py+point[1]*Pz + req
        # the guess of period
        T0 = time[idx+order]-time[idx]

        nstp = int(T0/0.001)
        dt   = T0/nstp
        orbit = integrator_reduced(original_point, dt, nstp+1)
        print (norm(orbit[-1,:] - orbit[0,:])) # print the error of this guess

        fig = figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(orbit[:,0], orbit[:,1], orbit[:,2])
        ax.scatter(original_point[0], original_point[1], original_point[2], c='r')
        ax.scatter(orbit[-1,0], orbit[-1,1], orbit[-1,2], c='k')



    if args.case == 2:
        '''
        use multishootimg method to refine the orbit you got in case 1
        '''
        x0 = array([]) # copy the initial guess here: 'original_point' in case 1
        T0 =  TBD# copy the initial guess of period here
        M = 4 # choose 4 (= order) points on the orbit for multishooting
        nstp = int(T0/0.001/M)
        dt = T0/nstp/M
        orbit = integrator_reduced(x0, dt, nstp*M)
        states_stack = array([]).reshape(0, 3)
        for i in range(M):
            states_stack = vstack((states_stack, orbit[i*nstp, :]))
        ms = Multishooting(integrator_reduced, integrator_reduced_with_jacob, velocity_reduced)
        xx, tt = ms.findPO(states_stack, dt, nstp, 40, 2e-14)

        # print out the period of orbit
        print (nstp*M*tt)

        # plot this periodic orbit
        fig = figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(M):
            states = integrator_reduced(xx[i,:], tt, nstp+1)
            ax.plot(states[:,0], states[:,1], states[:,2], 'r')

    show()
