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
from os.path import  basename,splitext
from time import time
import numpy as np
from matplotlib.pyplot import figure, show

def parse_args():
    parser = ArgumentParser(description=__doc__)
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    return basic if extra==None else f'{basic}{sep}{extra}'

if __name__=='__main__':
    start  = time()
    args = parse_args()
    fig = figure()
    ax = fig.add_subplot(2,2,1)
    t = np.linspace(-2,2, 100)
    x = np.cosh(t)
    for i,r in enumerate([2,1,0]):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(t,r - x)
        ax.axhline(0,c='xkcd:red',linestyle=':')
        ax.set_title(f'r={r}')
        ax.set_xlabel('x')
    fig.suptitle(r'3.1.2 $r = \cosh{x}$')
    fig.savefig(get_name_for_save())
    fig.tight_layout()
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    show()
