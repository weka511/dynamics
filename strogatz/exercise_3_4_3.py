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

'''3.4.3 Pitchfork Bifurcations'''

from argparse import ArgumentParser
from os.path import  basename,splitext
from time import time
import numpy as np
from matplotlib.pyplot import figure, show
from bifurcations import sketch_vector_field

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('r', default=[-1.0, 0, 1.0], type=float,nargs='*')
    parser.add_argument('--show', default = False, action='store_true')
    return parser.parse_args()

def get_name_for_save(extra=None,sep='-'):
    '''Extract name for saving figure'''
    basic = splitext(basename(__file__))[0]
    return basic if extra==None else f'{basic}{sep}{extra}'

if __name__=='__main__':
    start  = time()
    args = parse_args()
    fig = figure(figsize=(10,10))
    for i,r in enumerate(args.r):
        fixed_points = [0,0.5*np.sqrt(r),-0.5*np.sqrt(r)] if r>0 else [0]
        sketch_vector_field(r,
                            f = lambda x,r:r*x-4*x**3,
                            df = lambda x,r:r-12*x**2,
                            fixed_points = fixed_points,
                            ax = fig.add_subplot(len(args.r),1,1+i))
    fig.savefig(get_name_for_save())
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    if args.show:
        show()
