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
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--n', type=int, default = 5)
    parser.add_argument('case', type=int, choices = [1,2])
    return parser.parse_args()

def get_next(x):
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

def get_itinerary(cycle):
    return [0 if 2*a < b else 1 for a,b in cycle]

def format_cycle(cycle):
    def format(a,b):
        return '0' if 2*a < b else '1'
    return ''.join([format(a,b) for a,b in cycle])


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

def generate_prime_cycles(N):
    for n in range(1,args.n+1):
        for m in range(2**n):
            if n>1 and m == 0: continue
            if n>1 and m == 2**n-1: continue
            candidate = format(m,f'0{n}b')
            s = np.zeros(len(candidate))
            for i in range(len(candidate)):
                if candidate[i] == '1':
                    s[i] = 1
            w = get_w(s)
            yield candidate,s,w

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

def create_orbit(x0):
    orbit = [x0]
    a0,b0 = x0
    while True:
        an,bn = get_next(orbit[-1])
        if (a0==an and b0 == bn): break
        orbit.append((an,bn))
    return orbit

if __name__=='__main__':
    start  = time()
    args = parse_args()

    match args.case:
        case 1:
            for a0,b0 in [(0,1), (2,3), (4,5), (6,7), (8,9), (14,17), (14,15),
                        (16,17), (26,31), (28,33), (28,31), (10,11),
                        (30,31), (32,33)]:
                orbit = create_orbit((a0,b0))
                s = get_itinerary(orbit)
                print (f'{a0}/{b0}', s, get_w(s))

        case 2:

            for cycle in generate_prime_cycles(args.n):
                print (cycle)

            w = get_w(np.array([1,1,1]))
    fig = figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,1,1)

    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()

