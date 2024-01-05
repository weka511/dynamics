#!/usr/bin/env python

#   Copyright (C) 2024 Simon Crase

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''Chaosbook Exercise 7.2: Inverse Iteration Method for the Hénon repeller'''


from argparse import ArgumentParser, ArgumentTypeError
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from scipy.linalg import eig
from xkcd import create_colour_names

class SymbolicDynamics:
    '''
    A class that looks after symbolic dynamics. The user sees (e.g.) 10100,
    but the Inverse iterations see [+1, -1, +1, -1, -1]
    '''
    @staticmethod
    def sign(s):
        converted = int(s)
        if converted==0: return -1
        elif converted==1: return 1
        else: raise ArgumentTypeError(f'{s} should be 0 or 1')

    @staticmethod
    def get_sign(S,i):
        return S[i%len(S)]

    @staticmethod
    def get_p(S):
        return ''.join(['0' if s==-1 else '1' for s in S ])

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--N', type = int, default = 1000)
    parser.add_argument('--M', type = int, default = 100)
    parser.add_argument('--S', type = SymbolicDynamics.sign, nargs = '+')
    parser.add_argument('action',choices=['explore',
                                          'list',
                                          'prune',
                                          'rational'])
    parser.add_argument('--n', type = int, default = 6)
    parser.add_argument('--a', type = float, default = 6)
    parser.add_argument('--b', type = float, default = -1)
    parser.add_argument('--factorA',  default=False, action='store_true')
    parser.add_argument('--tol', type = float, default = 1e-6)
    return parser.parse_args()

def get_name_for_save(extra = None,
                      sep = '-',
                      figs = './figs'):
    '''
    Extract name for saving figure

    Parameters:
        extra    Used if we want to save more than one figure to distinguish file names
        sep      Used if we want to save more than one figure to separate extra from basic file name
        figs     Path name for saving figure

    Returns:
        A file name composed of pathname for figures, plus the base name for
        source file, with extra distinguising information if required
    '''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)



def calculate_cycle(rng,M,N,S,
                    a = 6,
                    b = -1,
                    atol = 1e-12):
    '''
    Find cycles in inverse Hénon map. We start with N points and iterate using inverse Hénon map.
    The N points are padded to avoid effect of zero boundary conditions (do we really need this?)

    Parameters:
        rng      Random number generator for initialization
        M        Padding
        N        Number of points in space
        S        Symbolic dynamics weights
        a        Hénon parameter
        b        Hénon parameter
        atol     Absolute tolerance. End iteration if solution moves by less than this value
    '''
    X = rng.uniform(0,0.5/len(S),2*M + N)
    for i in range(M):
        delta = 0.
        for j in range(1,X.size-1):
            term = max((-b-X[j-1]-X[j+1])/a,0)  # Issue 50: argument to sqrt sometimes
                                                # negative because of rounding
            x_new =   SymbolicDynamics.get_sign(S,j) * np.sqrt(term)
            delta = max(delta,abs(X[j]-x_new))
            X[j] =  x_new
        if delta<atol:
            break
    Cycle = X[M:M+len(S)]
    return Cycle,X

def generate_cycles(m):
    '''
    Produce all cycles of specified length
    '''
    for i in range(2**m):
        A = []
        for j in range(m):
            A.append(i%2)
            i = i//2
        yield A[::-1]

def factorize(n):
    '''
    Find all pairs of factors of a number

    Parameters:
        n       The number to be factorized

    Returns:
        List of pairs (k,m), such that n == k*m
    '''
    Factors = []
    for k in range(1,n+1):
        m = n//k
        if m*k==n:
            Factors.append((m,k))
    return Factors

def cycles_equal(c1,c2):
    '''
    Determine whether two strings are equal within a cyclic permutation.

    Parameters:
        c1   First string
        c2   The other
    '''
    assert len(c1)==len(c2),'Strings should be same length'
    if c1 == c2: return True
    for i in range(1,len(c1)):
        c = c1[i:] + c1[:i]
        if c == c2: return True
    return False

def anymatch(s,Factors):
    '''
    Verify that string can be broken into a set of substrings,
    such that each substring matches every other to within a cyclic permutation

    Parameters:
        s        The string
        Factors  A list of all integers (m,k) such that m*k = len(s)
    '''
    def matches(sub_cycles):
        for i in range(1,len(sub_cycles)):
            if not cycles_equal(sub_cycles[0],sub_cycles[i]): return False
        return True

    def some_match(m,k):
        if m==1: return False
        sub_cycles=[]
        for i in range(m):
            sub_cycles.append(s[i*k:(i+1)*k])
        return matches(sub_cycles)

    for m,k in Factors:
        if some_match(m,k):
            return True
    return False

def factorizes(s,n):
    '''
    Verify whether string factorizes into substrings that match to within a cycle
    '''
    Factors = factorize(n)
    for Factor in Factors:
        if len(Factors) > 0 and anymatch(s,Factors): return True
    return False

def seen_before(s,Cycles):
    '''
    Verify that a string already appears within a list to within a cycle

    Parameters:
        s       The string
        Cycles  The list to be searched
    '''
    for predecessor in Cycles:
        if cycles_equal(s,predecessor): return True
    return False

def prune(n):
    '''
    Generate strings from symbolic dynamics, subject to pruning rules:
    1.  No string can be a the cyclic permuitation of any other
    2.  No string can consist entirely of substrings that are cyclic permutations of each other.

    Parameters:
        n     Length of each string
    '''
    Cycles = []
    for cycle in generate_cycles(n):
        if not factorizes(cycle,n) and not seen_before(cycle,Cycles):
            Cycles.append(cycle)
            yield cycle

def create_jacobian(X,n,m,a=6,b=-1):
    '''
    Calculate Jacobian using equation (4.44)
    '''
    J = np.eye(2)
    for i in range(n):
        M = np.array([[-2*a*X[m+i],b],[1,0]])
        J = np.dot(M,J)
    return J

def get_lambda(X,n,m,a=6,b=-1):
    '''
    Get expanding eigenvalue
    '''
    w,_ = eig(create_jacobian(X,n,m,a=a,b=b))
    return np.real(w[np.argmax(abs(w))])

if __name__=='__main__':
    start  = time()
    args = parse_args()
    rng = np.random.default_rng()
    match(args.action):
        case 'explore':
            n = len(args.S)
            Cycle,X = calculate_cycle(rng,args.M,args.N,args.S,
                                      a = args.a,
                                      b = args.b)
            Colours = create_colour_names(n=n)
            Lambda = get_lambda(X,n,args.M,
                                a = args.a,
                                b = args.b)

            fig = figure(figsize=(12,12))
            ax1 = fig.add_subplot(1,1,1)
            ax1.scatter(X[args.M-1:-args.M-1],X[args.M:-args.M])
            for i in range(n):
                ax1.scatter(X[args.M-1+i],X[args.M+i],
                            c = Colours[i],
                            label = f'({X[args.M-1+i]:.06f},{X[args.M+i]:.06f})')
                ax1.arrow(X[args.M-1+i],X[args.M+i],X[args.M-1+i+1]-X[args.M-1+i],X[args.M+i+1]-X[args.M+i],
                          length_includes_head = True,
                          facecolor = Colours[(i+1)%n],
                          head_width = 0.03,
                          head_length = 0.05)
            tex_avge =  r'$\Sigma_i x_{p,i}$'
            ax1.set_title(fr'Cycles for Hénon repeller: p={SymbolicDynamics.get_p(args.S)}, $\Lambda_p$={Lambda:.6e}, {tex_avge}={Cycle.sum():.6f}')
            ax1.legend()
            fig.savefig(get_name_for_save())

        case 'list':
            factor = args.a if args.factorA else 1
            factor_string = f'/{args.a}' if args.factorA else ''
            for S in [
                [-1],
                [1],
                [1,-1],
                [1,-1,-1],
                [1,1,-1],
                [1,-1,-1,-1],
                [1,1,-1,-1],
                [1,1,1,-1],
                [1,-1,-1,-1,-1],
                [1,1,-1,-1,-1],
                [1,-1,1,0,0],
                [1,1,1,-1,-1],
                [1,1,-1,1,-1],
                [1,1,1,1,-1],
                [1,-1,-1,-1,-1,-1],
                [1,1,-1,-1,-1,-1],
                [1,-1,1,1,1,1],
                [1,1,1,-1,-1,-1],
                [1,1,-1,1,-1,-1],
                [1,-1,1,1,-1,-1],
                [1,1,1,1,-1,-1],
                [1,1,1,-1,1,-1],
                [1,1,1,1,1,-1]
            ]:
                Cycle,X = calculate_cycle(rng,args.M, args.N,S,
                                      a = args.a,
                                      b = args.b)
                Lambda = get_lambda(X,len(S),args.M,
                                      a = args.a,
                                      b = args.b)
                if args.factorA:
                    numerator = args.a*Cycle.sum()
                    if abs(round(numerator)-numerator) < args.tol:
                        print (f'{SymbolicDynamics.get_p(S):8s}\t{Lambda:9.6e}\t{numerator:9.0f}{factor_string}')
                    else:
                        print (f'{SymbolicDynamics.get_p(S):8s}\t{Lambda:9.6e}\t{numerator:9.06f}{factor_string}')
                else:
                    print (f'{SymbolicDynamics.get_p(S):8s}\t{Lambda:9.6e}\t{Cycle.sum():9.06f}')

        case 'prune':
            for n in range(1,args.n+1):
                for p in prune(n):
                    try:
                        S = [1 if p0==1 else -1 for p0 in p ]
                        Cycle,X = calculate_cycle(rng,args.M, args.N,S,
                                      a = args.a,
                                      b = args.b)
                        Lambda = get_lambda(X,n,args.M,
                                      a = args.a,
                                      b = args.b)
                        print (f'{SymbolicDynamics.get_p(S):8s}\t{Lambda:0.06e}\t{Cycle.sum():9.06f}')
                    except ValueError:
                        print (f'Value error processing {p}')

        case 'rational':
            fig = figure(figsize=(12,12))
            for j,S in enumerate([
                [1,-1],
                [1,1,-1,-1],
                [1,1,-1,1,-1,-1],
                [1,-1,1,1,-1,-1]
            ]):
                n = len(S)
                Colours = create_colour_names(n=n)
                Cycle,X = calculate_cycle(rng,args.M, args.N,S,
                                      a = args.a,
                                      b = args.b)
                ax1 = fig.add_subplot(2,2,1+j)
                for i in range(n):
                    ax1.scatter(X[args.M-1+i],X[args.M+i],
                                c = Colours[i],
                                label = f'({X[args.M-1+i]:.06f},{X[args.M+i]:.06f})')
                    ax1.arrow(X[args.M-1+i],X[args.M+i],X[args.M-1+i+1]-X[args.M-1+i],X[args.M+i+1]-X[args.M+i],
                              length_includes_head = True,
                              facecolor = Colours[(i+1)%n],
                              head_width = 0.03,
                              head_length = 0.05)
                tex_avge =  r'$\Sigma_i x_{p,i}$'
                ax1.set_title(fr'p={SymbolicDynamics.get_p(S)}, {tex_avge}={Cycle.sum():.6f}')
                ax1.legend()
            fig.suptitle('Cycles for Hénon repeller')
            fig.savefig(get_name_for_save())

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
