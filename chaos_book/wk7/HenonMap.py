'''
Stable and unstable manifold of Henon map (Example 15.5)
'''

from argparse          import ArgumentParser
from numpy             import arange, argmax, argmin, array, dot, empty_like, load, savez, size, sqrt, vstack, zeros
from matplotlib.pyplot import colorbar, figure, grid, legend, rc, savefig, show
from numpy.random      import rand
from scipy.interpolate import splrep, splev
from scipy.linalg      import eig, norm
from scipy.optimize    import fsolve

rc('text', usetex=True)

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
        term0  = (1-self.b) / (2*self.a)
        term1  = sqrt((1 + ((1-self.b)**2)/(4*self.a))/self.a)
        fixed0 = -term0-term1
        fixed1 = -term0+term1
        return (fixed0,fixed0),(fixed1,fixed1)

    def oneIter(self, stateVec):
        '''
        forward iterate for one step.

        stateVec: the current state. dimension : [1 x 2] numpy.array
        return: the next state. dimension : [1 x 2]
        '''
        x,y = stateVec

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

        state      = zeros((NumOfIter+1 , 2))
        state[0,:] = stateVec
        for i in range(NumOfIter):
            state[i+1,:] = self.oneIter(state[i,:])
        return state

    def Jacob(self, stateVec):
        '''
        The Jacobian for forward map at state point 'stateVec'.

        stateVec: the current state. dimension : [1 x 2] numpy.array
        '''
        x,y = stateVec

        return array([[-2*self.a*x, self.b],
                      [1,0]])



    def oneBackIter(self, stateVec):
        '''
        backward iterate for one step.

        stateVec: the current state. dimension : [1 x 2] numpy.array
        return: the previous state. dimension : [1 x 2]
        '''
        x,y       = stateVec
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
        state    = stateVec
        tmpState = stateVec
        for i in range(NumOfIter):
            tmp      = self.oneBackIter(tmpState)
            tmpState = tmp
            state    = vstack((state, tmpState))

        return state


def get_point(f,x0,tck):
    '''
    Case 2: use scipy.optimize.fsolve() to obtain intersection points B, C, D
    Parameters:
        f     Function to evaluate
        x0    Point to search from
        tck   Spline computed by splrep
    Returns:
        (x,y) such that f(x)=0, y = g(x), where g() is function represented by tck
    '''
    x = fsolve(f,x0)
    y = splev(x,tck)
    return x[0],y[0]


def get_interpolated_line(A, E, n = 25):
    def interpolate(u):
        return (u*A[0]+(1-u)*E[0],u*A[1]+(1-u)*E[1])
    return [interpolate(i/n) for i in range(n+1)]

def get_case2(file = 'case2.npz'):
    '''load needed variables from case2.'''
    case2     = load(file, allow_pickle = True)
    return (case2['B'], case2['C'], case2['D'], case2['eq0'], case2['eq1'], case2['tck'], case2['uManifold'], case2['sManifold'])



# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1,a2, b1,b2) :
    '''Snarfed from https://stackoverflow.com/questions/3252194/numpy-and-line-intersections'''
    def perp( a ) :
        b    = empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b
    da    = a2 - a1
    db    = b2 - b1
    dp    = a1 - b1
    dap   = perp(da)
    denom = dot( dap, db)
    num   = dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def detect4cycle(x,offset=0.1,epsilon=0.1):
    xs = henon.multiIter(x, 4)
    return offset if norm(xs[2]-x)<epsilon else (xs[-1]-x)

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
        eq0,eq1    = henon.fixed_points()

        fig        = figure(figsize=(6,6))       # check your implementation of forward map
        ax         = fig.add_subplot(111)
        ax.scatter(states[:,0], states[:, 1],
                   edgecolor = 'none',
                   s         = 1,
                   c         = 'xkcd:blue')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('(a)')
        savefig('case1')


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
        i_expand = argmax(abs(w))
        Lambda_e = w[i_expand] # expanding multiplier
        Ev       = vl[i_expand] # expanding eigenvector
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

        i_contract = argmin(abs(w))
        Ev         = vl[i_contract] # contracting eigenvector
        sManifold = eq0
        states    = zeros([N,2])
        for i in range(N): # get the initial N points
            states[i,:] = eq0 + (r0 + delta_r*i)*Ev
        sManifold = vstack((sManifold, states))

        for i in range(NumOfIter):
            for j in range(N): # update these N points along the manifold
                states[j,:] = henon.oneBackIter(states[j,:]);
            sManifold = vstack((sManifold, states))


        # get the spline interpolation of stable manifold. Note: unstable manifold are
        # double-valued, so we only interploate stable manifold, and this is
        # enough since unstable manifold and stable manifold is symmetric with line y = x.
        tck = splrep(sManifold[:,0], sManifold[:,1], s=0)
        m,_ = sManifold.shape

        # use scipy.optimize.fsolve() to obtain intersection points B, C, D
        # hint: use scipy.interpolate.splev() and the fact that stable and unstable
        # are symmetric with y = x

        C = get_point(lambda x:splev(x,tck)-x,eq1[0],tck)
        D = get_point(lambda x:splev(splev(x,tck),tck)-x,-0.5,tck)
        B = get_point(lambda x:splev(splev(x,tck),tck)-x,+0.5,tck)

        savez('case2',               # save the variables needed for case3
              B         = B,
              C         = C,
              D         = D,
              eq0       = eq0,
              eq1       = eq1,
              sManifold = sManifold,
              uManifold = uManifold,
              tck       = tck)

        # plot the unstable, stable manifold, points B, C, D, equilibria '0' and '1'.
        fig = figure(figsize=(6,6))
        ax  = fig.add_subplot(111)
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r-', lw=2, label=r'$W_u$')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c-', lw=2, label=r'$W_s$')
        ax.scatter(eq0[0],eq0[1])
        ax.scatter(eq1[0],eq1[1])
        ax.text(C[0], C[1], f' $C_x={C[0]:.4f}$')
        ax.text(D[0], D[1], 'D')
        ax.text(B[0], B[1], 'B')
        ax.text(eq0[0], eq0[1], '0')
        ax.text(eq1[0], eq1[1], '1')
        grid()
        legend()
        ax.set_title('(b)')
        savefig('case2')


    if args.case == 3:
        '''
        Try to establish the first level partition of the non-wandering set
        in Henon map. You are going to iterate region 0BCD forward and backward
        for one step.
        '''
        henon                                   = Henon() # use the default parameters: a=6, b=-1
        B,C,D,eq0, eq1,tck, uManifold,sManifold = get_case2()

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
        # fig = figure(figsize=(6,6))
        # ax  = fig.add_subplot(111)
        # ax.plot(M[:,0],M[:,1])
        # ax.text(C[0], C[1], f' $C_x={C[0]:.4f}$')
        # ax.text(D[0], D[1], 'D')
        # ax.text(B[0], B[1], 'B')
        # ax.text(eq0[0], eq0[1], '0')
        # savefig('RegionM')
        # Now iterate forward and backward the points in region 0BCD for one step

        Mf1 = array([]).reshape(0,2)
        for m in M:
            Mf1 = vstack((Mf1,henon.oneIter(m)))

        Mb1 = array([]).reshape(0,2)
        for m in M:
            Mb1 = vstack((Mb1,henon.oneBackIter(m)))

        state0 = henon.oneIter(eq0)
        stateC = henon.oneIter(C)
        stateB = henon.oneIter(B)
        stateD = henon.oneIter(D)

        # plot out Mf1 and Mb1

        fig = figure(figsize=(6,6))
        ax  = fig.add_subplot(111)
        ax.plot(Mb1[:,0], Mb1[:,1], 'g.',label=r'$M_b$')
        ax.plot(Mf1[:,0], Mf1[:,1], 'm.',label=r'$M_f$')

        ax.plot(uManifold[:,0], uManifold[:, 1], 'r', label=r'$W_u$')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c', label=r'$W_s$')
        ax.text(state0[0],state0[1],"0'")
        ax.text(stateC[0],stateC[1],"C'")
        ax.text(stateB[0],stateB[1],"B'")
        ax.text(stateD[0],stateD[1],"D'")
        ax.scatter(state0[0],state0[1],c='xkcd:blue',zorder=5,marker='x')
        ax.scatter(stateC[0],stateC[1],c='xkcd:blue',zorder=5,marker='x')
        ax.scatter(stateB[0],stateB[1],c='xkcd:blue',zorder=5,marker='x')
        ax.scatter(stateD[0],stateD[1],c='xkcd:blue',zorder=5,marker='x')
        ax.set_title('(c)')
        ax.legend()
        savefig('case3')


        # In order to see the pre-images of the borders of Mf1 and Mb1, please
        # try to plot the images and per-images of 4 edges of region 0BCD.
        # hint: use the interpolation function of stable manifold

        inner_f = [henon.oneIter(p) for p in get_interpolated_line(C,D)]
        inner_b = [henon.oneBackIter(p) for p in get_interpolated_line(C,B)]

        fig = figure(figsize=(6,6))
        ax  = fig.add_subplot(111)
        ax.plot(uManifold[:,0], uManifold[:, 1], 'r-',
                lw    = 2,
                label = r'$W_u$')
        ax.plot(sManifold[:,0], sManifold[:, 1], 'c-',
                lw    = 2,
                label = r'$W_s$')
        ax.plot([x for (x,_) in inner_f], [y for (_,y) in inner_f],'m-',
                lw     =2,
                label = r'$Pre_f$')
        ax.plot([x for (x,_) in inner_b], [y for (_,y) in inner_b],'b-',
                lw     =2,
                label = r'$Pre_b$')

        ax.text(C[0], C[1], '$M_{11}$')
        ax.text(D[0], D[1], '$M_{01}$')
        ax.text(B[0], B[1], '$M_{10}$')
        ax.text(eq0[0], eq0[1], '$M_{00}$')
        ax.legend()
        ax.set_title('(d)')
        savefig('Q7-3')

    if args.case == 4:
        '''
        We go further into the partition of state space in this case.
        In case3 you have figure out what the pre-images of the border of
        first forward and backward iteration, so we do not need to
        sample the region again, iteration of the border is enough.
        In this case we iterate forward and backward for two steps
        '''
        henon                                   = Henon() # use the default parameters: a=6, b=-1
        B,C,D,eq0, eq1,tck, uManifold,sManifold = get_case2()


        # initialize the first/second forward/backward iteration of the border
        Mf1 = [henon.oneIter(p) for p in get_interpolated_line(C,D,n=100)]
        Mf2 = [henon.oneIter(p) for p in Mf1]
        Mb1 = [henon.oneBackIter(p) for p in get_interpolated_line(C,B,n=50)]
        Mb2 = [henon.oneBackIter(p) for p in Mb1]
        # spl_Mb1 = splrep([x for (x,_) in Mb1], [y for (_,y) in Mb1], s=0)
        # implement your code here to get Mf1, Mf2, Mb1, Mb2
        # hint: use the interpolation function of stable manifold

        Mf2_in = [(x,y) for (x,y) in Mf2 if D[0]-0.05<x and x<D[0]+0.1 and D[1]-0.075<y and y<D[1]+0.075]
        Mb2_in = [(x,y) for (x,y) in Mb2 if D[0]-0.1<x and x<D[0]+0.1 and D[1]-0.225<y and y<D[1]+0.225]

        p1 = array( Mf2_in[0] )
        p2 = array( Mf2_in[-1] )
        p3 = array( Mb2_in[0] )
        p4 = array( Mb2_in[-1] )
        opposite_D = seg_intersect( p1,p2, p3,p4)
        # M = array([]).reshape(0,2) # region 0BCD
        x_range = arange(D[0],opposite_D[0], 0.0001)
        y_range = arange(opposite_D[1],D[1], 0.0001)
        a = zeros((size(x_range),size(y_range)))
        for i in range(size(x_range)):
            for j in range(size(y_range)):
                # x = x_range[i]
                # y = y_range[j]
                x,y = henon.multiIter([x_range[i],y_range[i]], 4)[-1]
                # for _ in range(4):
                    # (x,y) = henon.oneIter((x,y))
                a[i,j] = min(1,sqrt((x-x_range[i])**2 + (y-y_range[j])**2))
        # plot out your result.
        fig = figure(figsize=(12,6))
        ax1  = fig.add_subplot(121)
        ax2  = fig.add_subplot(122)
        # ax.plot(M[:,0],M[:,1])
        ax1.plot(uManifold[:,0], uManifold[:, 1], 'r',label = r'$W_u$')
        ax1.plot(sManifold[:,0], sManifold[:, 1], 'c',label = r'$W_s$')
        ax1.plot([x for (x,_) in Mf1], [y for (_,y) in Mf1], 'm',label = r'$Mf_1$')
        ax1.plot([x for (x,_) in Mf2], [y for (_,y) in Mf2], 'g',label = r'$Mf_2$')
        ax2.plot([x for (x,_) in Mf2_in], [y for (_,y) in Mf2_in], 'g',label = r'$Mf_2$')

        ax1.plot([x for (x,_) in Mb1], [y for (_,y) in Mb1], 'b',label = r'$Mb_1$')
        ax1.plot([x for (x,_) in Mb2], [y for (_,y) in Mb2], 'y',label = r'$Mb_2$')
        ax2.plot([x for (x,_) in Mb2_in], [y for (_,y) in Mb2_in], 'y',label = r'$Mf_2$')
        ax2.text(D[0], D[1], '$D$')
        ax2.scatter(opposite_D[0],opposite_D[1],c='xkcd:red',marker='x',s=50)
        # ax2.plot(M[:,0],M[:,1])
        heatmap = ax2.imshow(a,
                             cmap          = 'hot',
                             interpolation = 'nearest',
                             origin        = 'lower',
                             extent        = (x_range[0], x_range[-1], y_range[0], y_range[-1]))
        colorbar(heatmap)

        ax1.set_title('(e)')
        ax1.legend()
        ax1.grid()
        ax2.legend()
        savefig('case4')
        # find a point in the top left region (the region which is closest to point D)
        # as the initial condition to find a periodic period with period 4
        # hint: use fsolve()
        guess = array([-0.4, 0.55]) # [-0.415, 0.546]
        x = fsolve(detect4cycle,guess)
        print ('Found')
        print (henon.multiIter(x, 4)) # check whether it is periodic
        ax2.scatter(x[0],x[1],c='xkcd:green',marker='x',s=50)
        # if you like, you can figure out the symbolic representation
        # of this periodic orbit.

show()


