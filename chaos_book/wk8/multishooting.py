#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm, solve
from scipy.sparse.linalg import gmres

class Multishooting:
    '''
    Multishooting method to refine the initial guess of a periodic orbit.

    After obtaining the Poincare return map of a system, we can locate the
    fixed points of this map and retrive the initial condition for an orbit,
    which is very close to a periodic orbit (shadowing the periodic orbit).
    Then the next task is to refine the initial conditon to let it converge to
    a true periodic orbit. This is done by multishooting method.

    How it works ?

    Let the flow is f(x,t), and choose M points from the guess orbit:
    x_1, x_2, ..., x_M. If this orbit is periodic, then

                     [ x_1 - f(x_M, tau)     ]
    F(x_i, tau)  =   [ x_2 - f(x_1, tau)     ] = 0
                     [    ...                ]
                     [ x_M - f(x_{M-1}, tau) ]

    Take derivative of F, we get updating equation:

                           DF * dx = dF                           ( * )

    Here,
                 [      I                                -J(x_M, tau)  -v(f(x_M, tau))     ]
                 [ -J(x_1, tau)   I                                    -v(f(x_1, tau))     ]
        DF   =   [                .                                                        ]
                 [                 .                                                       ]
                 [                  .                                                      ]
                 [                       -J(x_{M-1}, tau)       I      -v(f(x_{M-1}, tau)) ]


                [dx_1]
        dx   =  [dx_2]
                [... ]
                [dx_M]
                [dtau]

                [ x_1 - f(x_M, tau)     ]
        dF  = - [ x_2 - f(x_1, tau)     ]
                [    ...                ]
                [ x_M - f(x_{M-1}, tau) ]

        J(x_i, tau) is Jacobian at x_i for time period tau. v(f(x_i), tau) is the
        velocity at point f(x_i, tau). In the following, we call DF the
        'multishooting matrix', and dF the 'difference vector'.

        let the guess period is T, then
                                tau = T / M
        is the time that needed for each point to evolve to the next point.
        Here we simplify this method in constrast to Chaobook because we let each point
        to evolve the same time, intead of varying time for each short piece. Also we
        do not integrate constraints into DF, as you have noticed that DF is not a square
        matrix. This is not allowed if we use Newton method to solve matrix function (*),
        but in our case, we use 'levenberg-marquardt algorithm' to solve (*), which
        does not require DF to be a square matrix. We prefer L-M algorithm to Newton since
        the former converges even if the initial guess is far away from the true periodic orbit.
        Anyway, you do not need to understand how L-M works since we have implemented for you.

   '''

    def __init__(self, intgr, intgr_jaco, velo):
        '''
        intgr:         integrator of the system. In this template code, we will pass
                       two modes system integrator
        intgr_jaco:    integrator of orbit and jacobian. We need to pass
                       integrator_reduced_with_jacob() in the experiments
        velo :         velocity field of the system.  velocity_reduced()
        '''
        self.intgr = intgr
        self.intgr_jaco = intgr_jaco
        self.velo = velo

    def shootingMatrix(self, states_stack, dt, nstp):
        '''
        Form the multi shooting matrix for updating the current orbit.
        Parameter:
            states_stack : states points along an orbit. It has dimension [M x N],
                           and each row represents a state point.
            dt           : integration time step
            nstp         : number of integration steps for each state point

        return:
              DF  : multishooting matrix
              dF  : the different vector
        '''
        M, N = states_stack.shape

        DF = np.zeros([N*M, N*M+1])
        dF = np.zeros(N*M)

        # fill out the first row of the multishooting matrix and difference vector
        x_start = states_stack[-1,:]
        x_end = states_stack[0,:]
        states, Jacob = self.intgr_jaco(x_start, dt, nstp+1)
        fx_start = states[-1,:]
        DF[0:N, 0:N] = np.eye(N)
        DF[0:N, -N-1:-1] = -Jacob;
        DF[0:N, -1] = - self.velo(fx_start, None)
        dF[0:N] = - (x_end - fx_start)

        # fill out row 2 to row M of multishooting matrix and difference vector
        for i in range(1, M):
            # iut your implementation here
            pass   # TO DO
        return DF, dF


    def dFvector(self, states_stack, dt, nstp):
        '''
        This function has been implemented for you !

        Similar to shootingMatrix(). Only form the difference matrix
        Parameter: the same as shootingMatrix()
        return: the different vector
        '''
        M, N = states_stack.shape
        xx = states_stack
        dF = np.zeros(M*N)

        x = self.intgr(xx[-1,:], dt, nstp+1)
        dF[0:N] = - (xx[0,:] - x[-1,:])
        for j in range(1, M):
            x = self.intgr(xx[j-1,:], dt, nstp+1)
            dF[N*j: N*(j+1)] = -(xx[j,:] - x[-1,:])

        return dF

    def findPO(self, states_stack, init_dt, nstp, maxIter, tol):
        '''
        This function has been implemented for you !

        implement the levenberg-marquardt algorithm to refine the initial condition
        for an periodic orbit.

        Parameter:
            states_stack : states points along an orbit. It has dimension [M x N],
                           and each row represents a state point.
            init_dt      : the initial guess of integration time step
            nstp         : number of integration steps for each state point
            maxIter      : maximal iteration number for L-M algorithm
            tol          : convergence tolerence
        return:
              x   : multishooting matrix
              dt  : the different vector

        '''
        M, N = states_stack.shape;
        x = states_stack
        dt = init_dt
        lam = 1.0
        x_new = x
        for i in range(maxIter):
            J, dF = self.shootingMatrix(x, dt, nstp);
            print ('iteration number i = ' + str(i) + ' has error: ' + str(norm(dF, np.inf)))
            if( norm(dF, np.inf) < tol ):
                print ('iteration terminates at error : ' + str(norm(dF, np.inf)))
                return x, dt
            JJ  = np.dot(J.T,  J) ;
            JdF = np.dot(J.T, dF);
            H = JJ + lam* np.diag(np.diag(JJ));
            delta_x = solve(H, JdF);
            #tmp =  gmres(H, JdF, tol = 1e-6, restart=30, maxiter = 100)
            #delta_x = tmp[0]; print tmp[1]
            for k in range(M):
                x_new[k,:] = x[k,:] + delta_x[k*N:(k+1)*N]
            dt_new = dt + delta_x[-1] / (nstp-1);
            dF_new = self.dFvector(x_new, dt_new, nstp)
            if norm(dF_new, np.inf) < norm(dF, np.inf):
                x = x_new;
                dt = dt_new
                lam = lam / 10.0;
            else :
                lam = lam * 10.0;
                if lam > 1e10:
                    print ('lam is too large')
                    return x, dt

        return x, dt
