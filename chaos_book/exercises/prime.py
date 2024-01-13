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

'''Exercise 14.2 Generate all Prime Cycles up to a specified length.'''


from argparse import ArgumentParser
from time import time
import numpy as np


def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default = 5, help='Maximum lenght for cycles.')
    return parser.parse_args()

def tent_map(x):
    '''
    Get next point in tent map

    Parameters:
        x    Current point as a rational a/b represented as (a,b)
    Returns:
        Tuple representing next point
    '''
    a,b = x
    if 2*a < b:
        return (2*a,b)
    else:
        return (2*(b-a),b)

def create_orbit(x0, amap=tent_map):
    '''
    Generate an orbit for some map

    Parameters:
        x    Starting point as a rational a/b represented as (a,b)
        amap Function that maps one rational number into another
    Returns:
        Array containing points for one cycle through orbit
    '''
    orbit = [x0]
    a0,b0 = x0
    while True:  # Loop until first point repeats
        an,bn = amap(orbit[-1])
        if (a0==an and b0 == bn): break
        orbit.append((an,bn))
    return np.array(orbit)

def orbit2itinerary(orbit):
    '''
    Convert an orbit to symbolic dynamics

    Parameters:
        orbit   Points in orbit (tuples representing rationals)
    Returns:
        Intinerary   Array of zeros and ones
    '''
    m,_ = orbit.shape
    itinerary = np.empty(m,dtype=int)
    for i in range(m):
        a,b = orbit[i]
        itinerary[i] = 0 if 2*a < b else 1
    return itinerary

def get_w(s):
    '''
    Calculate digits in accordance with Equation (14.4)

    Parameters:
        s     An itinerary

    Returns:
        Tent map point with future itinerary S, as computed by equation (14.4)
    '''
    n = len(s)
    w = np.empty((n), dtype=int)
    w[0] = s[0]
    for i in range(1,n):
        w[i] = w[i-1] if s[i] == 0 else (1 - w[i-1])
    return w if np.count_nonzero(s)%2==0 else np.concatenate((w,1-w))


def get_gamma(w,as_rational=True):
    '''
    Equation (14.4): convert cycle w to gamma

    Parameters:
        w           As defined in (14.4)
        as_rational Denotes whther to calculate answer as a rational (ordered pair) or a real
    '''
    def cancel(a,b):
        '''
        Reduce orderd pair, representing rational, to fully factored form
        '''
        for i in range(2,int(np.sqrt(a))+1):
            while a%i == 0 and b%i == 0:
                a //= i
                b //= i
        return a,b

    divisor = 2
    sum_to_be_repeated = 0
    for i in range(w.size):
        sum_to_be_repeated *= 2
        sum_to_be_repeated += w[i]
        divisor *= 2
    gamma = cancel(2*sum_to_be_repeated,divisor-2)

    return gamma if as_rational else 2*sum_to_be_repeated/(divisor-2) # See example 14.10


def generate_prime_cycles(n):
    '''
    Generate prime cycles of a specified length

    Parameters:
        n       Lenght of cycles
    '''
    def matches(cycle1,cycle2):
        '''
        Used to determine whether two specified cycles are identical modulo rotation
        '''
        def matches1(k):
            '''
            Used to determine whether two specified cycles are identical modulo
            a rotation by one position
            '''
            for i in range(n):
                if cycle1[i] != cycle2[(i+k)%n]: return False
            return True

        for k in range(1,n):
            if matches1(k): return True
        return False

    def factors(cycle):
        '''
        Used to determine whether a cycle factors into copies of a shorter cycle
        '''
        def found_factorization(i,m):
            '''
            Used to determine whether a cycle factors into a specified number of
            copies of a cycle whose length is specified
            '''
            for j in range(1,m):
                segment1 = cycle[:i]
                segment2 =  cycle[j*i:(j+1)*i]
                if not (segment1 == segment2).all(): return False
            return True

        for i in range(2,len(cycle)):
            m = len(cycle)//i
            if i*m == len(cycle):
                if found_factorization(i,m): return True
        return False

    def some_cycle_factors(equivalent_cycles):
        '''
        Used to determine whether any cycle in a group factors
        into copies of a shorted cycle.
        '''
        for cycle in equivalent_cycles:
            if factors(cycle): return True
        return False

    def create_candidates():
        '''
        Build a list of candidate cycles, i.e. all possible cycles of length n,
        then link each one back to an earlier occurence of a cyclic permutation of itself.
        If there is no earlier occurence, link to itself
        '''
        candidate = np.zeros((2**n,n),dtype=int)
        cycle_indices = np.empty((2**n),dtype=int)
        for i in range(2**n):
            k = i
            for j in range(n):
                candidate[i,n-j-1] = k%2
                k //= 2
            cycle_indices[i] = i
            for j in range(i):
                if matches(candidate[i,:],candidate[j,:]):
                    cycle_indices[i] = j
                    break
        return candidate,cycle_indices

    def create_equivalence_cycles(k,n,candidate,cycle_indices):
        '''
        We need to organize the candidate cycles into equivalence classes, where cycles
        are deemed equivalent if one can be rotated onto another

        Parameters:
            k              Denotes cycle being processes
            n              Length of cycles
            candidate      All possible cycles of length n
            cycle_indices  A link to another cycle that is a rotation of k
        '''
        product = []

        for i in range(1,2**n-1):
            if cycle_indices[i] == k:
                product.append(candidate[i,:])
        return product

    candidate,cycle_indices = create_candidates()

    for k in list(set(cycle_indices)):
        if n == 1:     # For some reason the code below doesn't work for this case
            yield [0]
            yield [1]
            return

        equivalent_cycles = create_equivalence_cycles(k,n,candidate,cycle_indices)

        # Discard any cycles that can be factored into repetitions of a simpler cycle
        if k==0: continue          # n*[0]
        if k==2**n-1: continue     # n*[1]
        if some_cycle_factors(equivalent_cycles): continue

        # Now find the cycle that gives the largest value of gamma - see equation (14.7)
        gammas = []
        for cycle in equivalent_cycles:
            gammas.append(get_gamma(get_w(cycle), as_rational=False))

        yield equivalent_cycles[np.argmax(gammas)]


if __name__=='__main__':
    start  = time()
    args = parse_args()

    for n in range(args.n+1):
        for s in generate_prime_cycles(n):
            w = get_w(s)
            a,b = get_gamma(w)
            print (s, w, f'{a}/{b}')

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
