'''
Stable and unstable manifold of Henon map (Example 15.5)
'''

from argparse          import ArgumentParser
from numpy             import arange, argmax, array, load, savez,  size, sqrt, vstack, zeros
from matplotlib.pyplot import figure, legend, show
from numpy.random      import rand
from scipy.interpolate import splrep, splev
from scipy.linalg      import eig, norm
from scipy.optimize    import fsolve

TBP = None

class Henon:
    '''
    Class Henon contains functions for both forward and
    backward Henon map iteration.
    '''
    def __init__(self, a=6, b=-1):
        '''
        initialization function which will be called every time you
        create an object instance. In this case, it initializes
        parameter a and b, which are the parameters of Henon map.
        '''
        self.a = a
        self.b = b

    def fixed_points(self):
        term0 = (1-self.b) / (2*self.a)
        term1 = sqrt((1 + ((1-self.b)**2)/(4*self.a))/self.a)
        fixed0 = -term0-term1
        fixed1 = -term0+term1
        return (fixed0,fixed0),(fixed1,fixed1)

    def oneIter(self, stateVec):
        '''
        forward iterate for one step.

        stateVec: the current state. dimension : [1 x 2] numpy.array
        return: the next state. dimension : [1 x 2]
        '''
        x = stateVec[0];
        y = stateVec[1];

        stateNext = [1 - self.a*x*x + self.b*y,x]

        return stateNext

    def multiIter(self, stateVec, NumOfIter):
        '''
        forward iterate for multiple steps.
        stateVec: the current state. dimension : [1 x 2] numpy.array
        NumOfIter: number of iterations

        return: the current state and the furture 'NumOfIter' states.
                dimension [NumOfIter+1 x 2]
        Hint : numpy.vstack()
        '''

        state = zeros((NumOfIter+1 , 2))
        state[0,:]=stateVec
        for i in range(NumOfIter):
            state[i+1,:] = self.oneIter(state[i,:])
        return state

    def Jacob(self, stateVec):
        '''
        The Jacobian for forward map at state point 'stateVec'.

        stateVec: the current state. dimension : [1 x 2] numpy.array
        '''
        x = stateVec[0];
        y = stateVec[1];

        jacobian = array([[-2*self.a*x, self.b],
                          [1,0]])

        return jacobian

    def oneBackIter(self, stateVec):
        '''
        backward iterate for one step.

        stateVec: the current state. dimension : [1 x 2] numpy.array
        return: the previous state. dimension : [1 x 2]
        '''
        x = stateVec[0];
        y = stateVec[1];

        statePrev = (y,-(1-self.a*y*y-x)/self.b)
        return statePrev

    def multiBackIter(self, stateVec, NumOfIter):
        '''
        backward iterate for multiple steps.
        stateVec: the current state. dimension : [1 x 2] numpy.array
        NumOfIter: number of backward iterations

        return: the current state and the pervious 'NumOfIter' states.
                dimension [NumOfIter+1 x 2]
        '''
        state = stateVec
        tmpState = stateVec
        for i in range(NumOfIter):
            tmp = self.oneBackIter(tmpState)
            tmpState = tmp
            state = vstack((state, tmpState))

        return state


if __name__ == '__main__':

    parser = ArgumentParser('Stable and unstable manifold of Henon map (Example 15.5)')
    parser.add_argument('case',
                        type    = int,
                        choices = [1,2,3,4])
    args = parser.parse_args()

    if args.case == 1:
        '''
        Validate your implementation of Henon map.
        Note here we use a=1.4, b=0.3 in this valication
        case. This is the classical Hénon map -- wikipedia.
        For other cases in this homework, we use a=6, b=-1.
        Actually, these two sets of parameters are both important
        since we will come back to this model when discussing invariant measure.
        '''
        henon      = Henon(1.4, 0.3) # create a Henon instance
        states     = henon.multiIter(rand(2), 1000) # forward iterations
        states_bac = henon.multiBackIter(states[-1,:], 10) # backward iterations

        eq0,eq1 = henon.fixed_points()

        fig        = figure(figsize=(6,6))       # check your implementation of forward map
        ax         = fig.add_subplot(111)
        ax.scatter(states[:,0], states[:, 1], edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('(a)')
        show()

        # check the correctness of backward map. The first 10 states_bac should
        # be the last 10 states in reverse order.
        # Note, backward map is very unstable for a=1.4, b=0.3,
        # so we only iterate backward for 10 times.
        print (states[-10:, :])
        print ('======')
        print (states_bac[:10,:])
        print ('======')

        # check the Jacobian matrix at (0.1, 0.2).
        # for a = 1.4, b = 0.3, the output should be
        # [[-0.28  0.3 ]
        #  [ 1.    0.  ]]

        print (henon.Jacob(array([0.1, 0.2])))


    if args.case == 2:
        '''
        Try to obtain the stable and unstable manifold for
        equilibrium '0' in Henon map with parameter a=6, b=-1.

        Plotting unstable/stable manifold is a difficult task in general.
        We need to moniter a lot of variables, like the distances between points
        along the manifold, the angle formed by adjacent 3 points on the manifold, etc.
        However, for Henon map, a simple algorithm is enough for demonstration purpose.
        The algorithm works as follows:

        Unstable manifold: start from a point close to equilibrium '0', in the direction of
        the expanding eigenvector: ' eq0 + r0 * V_e '. Here 'eq0' is the equilibrium,
        'r0' is a small number, 'V_e' is the expanding eigen direction of equilibrium '0'.
        The image of
        this point after one forward iteration should be very close to
        ' eq0 + r0 * \Lambda_e * V_e', where
        '\Lambda_e' is the expanding multiplier. We can confidently think that these two
        points are sitting on the unstable manifold of 'eq0' since 'r0' is very small.
        Now, we interpolate linearly between these two points and get total 'N' points.
        If we iterate these 'N' points forward for 'NumOfIter' times, we get the unstable
        manifold within some length.
        The crutial part of this method is that when these 'N' state points are being iterated,
        the spacings between them get larger and larger, so we need to use a relative large
        value 'N' to ensure that these state points are not too far away from each other.
        The following formula is used to determine 'N' ( please figure out its meaning
        and convince yourself ):

        ( (\Lambda_e - 1) * r0 / N ) * (\Lambda_e)^NumOfIter = tol  (*)

        Here, 'tol' is the tolerance distance between adjacent points in the unstable manifold.

        Stable manifold: If we reverse the dynamics, then the stable direction becomes the
        ustable direction, so we can use the same method as above to obtain stable manifold.

        '''
        henon = Henon() # use the default parameters: a=6, b=-1
        # get the two equilbria of this map. equilibrium '0' should have smaller x coordinate.

        eq0,eq1 = henon.fixed_points()

        # get the expanding multiplier and eigenvectors at equilibrium '0'
        J        = henon.Jacob(eq0)
        w,vl     = eig(J)
        i        = argmax(abs(w))
        Lambda_e = w[i] # expanding multiplier
        Ev       = vl[i] # expanding eigenvector
        assert norm(Ev)==1
        NumOfIter = 5 # number of iterations used to get stable/unstable manifold
        tol       = 0.1 # tolerance distance between adjacent points in the manifold
        r0        = 0.0001 # small length
        N         = int((Lambda_e-1)*r0*(Lambda_e**NumOfIter)/tol)# implement the formula (*). Note 'N' should be an integer.
        delta_r   = (Lambda_e-1)*r0 / N # initial spacing between points in the manifold

        # generate the unstable manifold. Note we do not use Henon.multiIter() here
        # since we want to keep the ordering of the points along the manifold.
        uManifold = eq0
        states    = zeros([N,2])
        for i in range(N): # get the initial N points
            states[i,:] = eq0 + (r0 + delta_r*i)*Ev
        uManifold = vstack((uManifold, states))

        for i in range(NumOfIter):
            for j in range(N): # update these N points along the manifold
                states[j,:] = henon.oneIter(states[j,:]);
            uManifold = vstack((uManifold, states))

        fig = figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r-', lw=2, label=r'$W_u$')
        ax.scatter(eq0[0],eq0[1])
        ax.scatter(eq1[0],eq1[1])
        ax.text(eq0[0], eq0[1], '0')
        ax.text(eq1[0], eq1[1], '1')
        legend()
        show()
        # Please fill out this part to generate the stable manifold.
        # Check whether the stable manifold is symmetric with unstable manifold with
        # diagonal line y = x
        sManifold = TBP


        # get the spline interpolation of stable manifold. Note: unstable manifold are
        # double-valued, so we only interploate stable manifold, and this is
        # enough since unstable manifold and stable manifold is symmetric with line y = x.
        tck = splrep(sManifold[:,0], sManifold[:,1], s=0)

        # use scipy.optimize.fsolve() to obtain intersection points B, C, D
        # hint: use scipy.interpolate.splev() and the fact that stable and unstable
        # are symmetric with y = x
        C = TBP
        D = TBP
        B = TBP

        # save the variables needed for case3
        # if you are using ipython enviroment, you could just keep the varibles in the
        # current session.
        savez('case2', B=B, C=C, D=D, eq0=eq0, eq1=eq1,
                 sManifold=sManifold, uManifold=uManifold, tck=tck)

        # plot the unstable, stable manifold, points B, C, D, equilibria '0' and '1'.
        fig = figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r-', lw=2, label=r'$W_u$')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c-', lw=2, label=r'$W_s$')
        ax.scatter(eq0[0],eq0[1])
        ax.scatter(eq1[0],eq1[1])
        ax.text(C[0], C[1], 'C')
        ax.text(D[0], D[1], 'D')
        ax.text(B[0], B[1], 'B')
        ax.text(eq0[0], eq0[1], '0')
        ax.text(eq1[0], eq1[1], '1')
        legend()
        show()



    if args.case == 3:
        '''
        Try to establish the first level partition of the non-wandering set
        in Henon map. You are going to iterate region 0BCD forward and backward
        for one step.
        '''
        henon = Henon() # use the default parameters: a=6, b=-1
        # load needed variables from case2. If you have kept the variables
        # in the working space, then just comment out the following few lines.
        case2 = load('case2.npz')
        B = case2['B']; C = case2['C']; D = case2['D']; eq0 = case2['eq0']; eq1 = case2['eq1'];
        tck = case2['tck']; uManifold = case2['uManifold']; sManifold = case2['sManifold'];

        # We first make a sampling of region 0BCD.
        # It works like this:
        # we are sure that box [-0.8, 0.8] x [-0.8, 0.8]
        # can cover region OBCD, but not every point inside this box is in 0BCD,
        # so for point (x, y), how to determine whether it is in 0BCD ?
        # The criteria is
        #            y < f(x)  and  x < f(y)
        # Here f() is the interpolation function of stable manifold.
        # It is easy to see that 'y < f(x)' enforces point (x, y) below the
        # stable manifold, but the fact that 'x < f(y)' enforces point (x, y) to be
        # at the left side of the unstable manifold is a little tricky. The answer
        # is that stable manifold and unstable manifold are symmetric with y = x
        # Anyway, this part is implemented for you.
        M = array([]).reshape(0,2) # region 0BCD
        x = arange(-0.8, 0.8, 0.01)
        y = splev(x, tck)
        for i in range(size(x)):
            for j in range(size(x)):
                if x[i] < y[j] and x[j] < y[i]:
                    state = array([x[i], x[j]])
                    M = vstack( (M, state) )

        # please plot out region M to convince yourself that you get region 0BCD
        # Now iterate forward and backward the points in region 0BCD for one step
        Mf1 = TBP# forward iteration points
        Mb1 = TBP# backward iteration points

        # plot out Mf1 and Mb1
        fig = figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot(Mb1[:,0], Mb1[:,1], 'g.')
        ax.plot(Mf1[:,0], Mf1[:,1], 'm.')
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c')
        show()

        # In order to see the pre-images of the boarders of Mf1 and Mb1, please
        # try to plot the images and per-images of 4 edges of region 0BCD.
        # hint: use the interpolation function of stable manifold


    if args.case == 4:
        '''
        We go further into the partition of state space in this case.
        In case3 you have figure out what the pre-images of the border of
        first forward and backward iteration, so we do not need to
        sample the region again, iteration of the boarder is enough.
        In this case we iterate forward and backward for two steps
        '''
        henon = Henon() # use the default parameters: a=6, b=-1
        # load needed variables from case2
        case2 = load('case2.npz')
        B = case2['B']; C = case2['C']; D = case2['D']; eq0 = case2['eq0']; eq1 = case2['eq1'];
        tck = case2['tck']; uManifold = case2['uManifold']; sManifold = case2['sManifold'];

        # initialize the first/second forward/backward iteration of the boarder
        Mf1 = array([]).reshape(0,2) # the first forward iteration of the boarder you got in case 3
        Mf2 = array([]).reshape(0,2) # ... second forward ....
        Mb1 = array([]).reshape(0,2) # ....first backward ....
        Mb2 = array([]).reshape(0,2) # ....second backward ....

        # implement your code here to get Mf1, Mf2, Mb1, Mb2
        # hint: use the interpolation function of stable manifold


        # plot out your result.
        fig = figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c')
        ax.plot(Mf1[:,0], Mf1[:,1], 'm')
        ax.plot(Mf2[:,0], Mf2[:,1], 'g')
        ax.plot(Mb1[:,0], Mb1[:,1], 'b')
        ax.plot(Mb2[:,0], Mb2[:,1], 'y')
        ax.set_title('(d)')
        show()

        # find a point in the top left region (the region which is closest to point D)
        # as the initial condition to find a periodic period with period 4
        # hint: use fsolve()
        guess = array([-0.4, 0.5])
        x = TBP # the initial condition you get from this guess
        print (henon.multiIter(x, 4)) # check whether it is periodic
        # if you like, you can figure out the symbolic representation
        # of this periodic orbit.

