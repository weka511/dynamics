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

# This program is intended to support my project from https://chaosbook.org/

from contextlib        import AbstractContextManager
from os.path           import basename, join, splitext
from matplotlib.pyplot import figure,  savefig

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
        self.fig         = figure(figsize=(width,height))

    def __enter__(self):
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
