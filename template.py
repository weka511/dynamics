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

def get_name_for_save():
    '''Extract name for saving figure'''
    return splitext(basename(__file__))[0]

if __name__=='__main__':
    start  = time()
    args = parse_args()

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
