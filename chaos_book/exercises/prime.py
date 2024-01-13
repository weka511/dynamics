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

'''Exercise 14.2 Generating Prime Cycles'''


from argparse import ArgumentParser
from time import time
import numpy as np


def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default = 5)
    parser.add_argument('case', type=int, choices = [1,2])
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

def create_orbit(x0, map=tent_map):
    '''
    Generate an orbit for tent map

    Parameters:
        x    Starting point as a rational a/b represented as (a,b)
    Returns:
        Point s for one cycle through orbit
    '''
    orbit = [x0]
    a0,b0 = x0
    while True:
        an,bn = map(orbit[-1])
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
    Equation (14.4)

    Parameters:
        s     In itinerary

    Returns:
        Tent map point with future itinerary S, as computed by equation (14.4)
    '''
    n = len(s)
    w = np.empty((n))
    w[0] = s[0]
    for i in range(1,n):
        w[i] = w[i-1] if s[i] == 0 else (1 - w[i-1])
    if np.count_nonzero(s)%2==0:
        return w
    else:
        return np.concatenate((w,1-w))

def evaluate_gamma(w):
    '''
    Equation (14.4): convert cycle w to gamma
    '''
    divisor = 2
    sum = 0
    for i in range(w.size):
        sum += w[i]/divisor
        divisor *= 2
    return sum / (1 - 2/divisor)

# def generate_prime_cycles(N):
    # for n in range(1,N):
        # for m in range(2**n):
            # if n>1 and m == 0: continue
            # if n>1 and m == 2**n-1: continue
            # candidate = format(m,f'0{n}b')
            # s = np.zeros(len(candidate))
            # for i in range(len(candidate)):
                # if candidate[i] == '1':
                    # s[i] = 1
            # w = get_w(s)
            # gamma = evaluate_gamma(w)
            # yield candidate,s,w,gamma

def generate_cycles(n):

    def matches(cycle1,cycle2):
        def matches1(k):
            for i in range(n):
                if cycle1[i] != cycle2[(i+k)%n]: return False
            return True
        for k in range(1,n):
            if matches1(k): return True
        return False

    def factors(cycle):
        def found_factorization(i,m):
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
        for cycle in equivalent_cycles:
            if factors(cycle): return True
        return False

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
    for k in list(set(cycle_indices)):
        if k==0: continue
        if k==2**n-1: continue
        equivalent_cycles = []

        for i in range(1,2**n-1):
            if cycle_indices[i] == k:
                equivalent_cycles.append(candidate[i,:])

        if some_cycle_factors(equivalent_cycles): continue

        gammas = []
        for cycle in equivalent_cycles:
            w = get_w(cycle)
            gammas.append(evaluate_gamma(w))
        yield equivalent_cycles[np.argmax(gammas)]

def format_cycle(cycle):
    '''Used to display cycle'''
    def format(a,b):
        return '0' if 2*a < b else '1'
    return ''.join([format(a,b) for a,b in cycle])

if __name__=='__main__':
    start  = time()
    args = parse_args()

    match args.case:
        case 1:
            for a0,b0 in [(0,1), (2,3), (4,5), (6,7), (8,9), (14,17), (14,15),
                        (16,17), (26,31), (28,33), (28,31), (10,11), (30,31), (32,33)]:
                s = orbit2itinerary(create_orbit((a0,b0)))
                w = get_w(s)
                print (f'{a0}/{b0}', s, w, abs(evaluate_gamma(w) - a0/b0))

        case 2:
            for n in range(args.n):
                for c in generate_cycles(n):
                    print (c)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
