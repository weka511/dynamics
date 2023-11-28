#!/usr/bin/env python

#   Copyright (C) 2023 Simon Crase

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

'''Template for python script for dynamics'''


from argparse import ArgumentParser
from os.path import  basename,splitext,join
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--show',  default=False, action='store_true', help='Show plots')
    parser.add_argument('--method', default='euler')
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
        source file, with extra ditinguising information if required
    '''
    basic = splitext(basename(__file__))[0]
    name = basic if extra==None else f'{basic}{sep}{extra}'
    return join(figs,name)

def euler(x0,n=1):
    N = 2**n
    h = 1/N
    x = x0
    for i in range(N):
        x -= h*x
    return x

def euler2(x0,n=1):
    N = 2**n
    h = 1/N
    x = x0
    for i in range(N):
        x1 = x - h*x
        x -= 0.5*h*(x+x1)
    return x

if __name__=='__main__':
    start  = time()
    args = parse_args()
    method = euler if args.method=='euler' else euler2
    x = [abs(method(1.0,n)-np.exp(-1)) for n in range(10)]
    fig = figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title('Error')
    ax1.plot(x,label='Error')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(np.log(x),label='Log Error')
    ax2.set_title('Log Error')
    fig.suptitle(args.method)
    fig.savefig(get_name_for_save(extra='method'))
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
