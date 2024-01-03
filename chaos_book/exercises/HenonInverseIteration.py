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
from xkcd import create_colour_names

def sign(s):
    converted = int(s)
    if converted==0: return -1
    elif converted==1: return 1
    else: raise ArgumentTypeError(f'{s} should be 0 or 1')

def parse_args():
    '''Define and parse command line arguments'''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--N', type = int, default = 1000)
    parser.add_argument('--M', type = int, default = 100)
    parser.add_argument('--S', type = sign, nargs = '+')
    parser.add_argument('action',choices=['explore','list'])
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

def get_sign(S,i):
    return S[i%len(S)]

def get_p(S):
    return ''.join(['0' if s==-1 else '1' for s in S ])

def calculate_cycle(rng,M,N,S,a=6):
    X = rng.uniform(0,0.25,2*M + N)
    for i in range(M):
        for j in range(1,X.size-1):
            X[j] =   get_sign(S,j) * np.sqrt((1-X[j-1]-X[j+1])/a)

    Cycle = X[M:M+len(S)]
    return Cycle,X

if __name__=='__main__':
    start  = time()
    args = parse_args()
    rng = np.random.default_rng()
    match(args.action):
        case 'explore':
            Cycle,X = calculate_cycle(rng,args.M,args.N,args.S)

            Colours = create_colour_names(n=len(args.S))
            fig = figure(figsize=(12,12))
            ax1 = fig.add_subplot(1,1,1)
            ax1.scatter(X[args.M-1:-args.M-1],X[args.M:-args.M])
            for i in range(len(Colours)):
                ax1.scatter(X[args.M-1+i],X[args.M+i],
                            c = Colours[i],
                            label = f'i={i}')
                ax1.arrow(X[args.M-1+i],X[args.M+i],X[args.M-1+i+1]-X[args.M-1+i],X[args.M+i+1]-X[args.M+i],
                          length_includes_head = True,
                          facecolor = Colours[(i+1)%len(Colours)],
                          head_width = 0.03,
                          head_length = 0.05)
            tex_avge =  r'$\Sigma_i x_{p,i}$'
            ax1.set_title(f'Cycles for Hénon repeller: p={get_p(args.S)}, {tex_avge}={Cycle.sum():.6f}')
            ax1.legend()
            fig.savefig(get_name_for_save())

        case 'list':
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
                Cycle,_ = calculate_cycle(rng,args.M, args.N,S)
                print (f'{get_p(S):8s}\t{Cycle.sum():9.06f}')

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
