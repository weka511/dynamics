#!/usr/bin/env python

# Copyright (C) 2022 Simon Crase

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

'''Utility classes for use by other programs'''

from contextlib        import AbstractContextManager
from os.path           import basename, join, splitext
from matplotlib.pyplot import figure,  savefig, show
from numpy             import argwhere, array, dot, linspace, meshgrid, ndarray, stack
from time              import time

class Figure(AbstractContextManager):
    '''Context manager for plotting and saving a figure'''
    def __init__(self,
                 figs     = './',
                 name     = '',
                 dynamics = None,
                 file     = __file__,
                 width    = 12,
                 height   = 12):
        '''
        Initialize Figure

        Parameters:
            figs      Identifies location where figure file will be stored
            file      1st cmponent of figure file name: should be __file__ for top level source file
            dynamics  Used to create 2nd component of figure file name
            name      Used to create 3rd component of figure file name
            width     Width of figure in inches
            height    Height of figure in inches
        '''
        self.figs        = figs
        self.source_file = file
        self.dynamics    = dynamics
        self.name        = name
        self.figsize     = (width,height)

    def __enter__(self):
        self.fig         = figure(figsize=self.figsize)
        return self.fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''Save figure on exit'''
        if exc_type==None and exc_val==None and exc_tb==None:
            savefig(self.get_path_name())
            return True
        else:
            return False

    def get_path_name(self):
        '''
        Identifies path where figure will be stored, comprising:
           1. The folder where figure is to be stored
           2. Name of top level source file for program, passed in a a parameter
           3. Name of Dynamics (e.g. "Lorentz")
           4. An additional compoient of the name.
        '''
        return join(self.figs,f'{splitext(basename(self.source_file))[0]}-{self.dynamics.name}-{self.name}')

class Timer(AbstractContextManager):
    '''
    Context Manager for estimating time
    Prints the elapsed time from __enter__(...) to __exit__(...)
    '''
    def __init__(self,name='Timer',silent=False):
        self.name   = name
        self.silent = silent

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self,exc_type, exc_val, exc_tb):
        if not self.silent:
            print (f'{self.name}: Elapsed time = {time()-self.start:.0f} seconds')
        return exc_type==None and exc_val==None and exc_tb==None

    def get_elapsed(self):
        return time()-self.start

def get_plane( sspTemplate = array([1,1,0]),
               nTemplate   = array([1,-1,0]),
               limits      = [linspace(-1,1,50)]*3):
    '''
    Used to plot section as a surface

    Parameters:
        sspTemplate     Template point for surface
        nTemplate       Normal to surface plotted
        limits          Controls the size of the area to be

    Returns: x,y, and z coordinates for points in a plane
    '''
    class Swapper:
        '''Copy an array and then swap two "rows", i.e. entities along the first axis'''
        def __init__(self,k1,k2):
            self.k1 = k1
            self.k2 = k2

        '''
        Actually swap the two rows

        Parameters:
               A Array whose rows are to be swapped
        Returns:
               A with rows swapped
        '''
        def swap(self,A):
            Product          = A.copy()
            Product[self.k1] = A[self.k2]
            Product[self.k2] = A[self.k1]
            return Product

    def get_z(x,y,
              normal = [1,1,1]):
        '''
        Equation of plane. Since we want to avoid divison by zero, the outer function will
        cyclically permute the coordinates (rolling) to ensure that normal[2] is non zero.

        Parameters:
            x        First independent variable after rolling
            y        Second independent variable after rolling
            normal   This is the normal, with rows swapped so it will match x and y.
                     The inner product dot(sspTemplate,nTemplate) is invariant,
                     so there is no need to swap nTemplate for that calculation.
        '''
        return (dot(sspTemplate,nTemplate) - normal[0]*x - normal[1]*y)/normal[2]

    i_nz     = argwhere(nTemplate)                         # Indices of non-zero components
    m,_      = i_nz.shape                                  # Verify that there is at least one non-zero
    assert m>0,'Normal should not be all zeroes'
    swapper = Swapper(i_nz[[-1]].item(),len(nTemplate)-1)
    limits  = swapper.swap(limits)
    xx,yy   = meshgrid(limits[0],limits[1])
    return swapper.swap(stack([xx,yy,get_z(xx,yy,
                                           normal = swapper.swap(nTemplate))]))

def xkcd_colour_names(file='./rgb.txt', prefix='xkcd'):
    '''A generator to list colour names for rgb.txt'''
    with open(file) as f:
        for line in f:
            name = line.split('#')[0]
            yield(f'{prefix}:{name.strip()}')

if __name__=='__main__':
    for name in xkcd_colour_names():
        print (name)
    with Figure(dynamics=type('Dummy',(object,),{'name':'Dummy'})) as fig:
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Test')

    with Timer():
        x=0

    show()
