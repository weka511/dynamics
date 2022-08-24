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
    '''Context manager for plotting a figure'''
    def __init__(self,
                 figs     = './',
                 name     = '',
                 dynamics = None,
                 file     = __file__):
        self.figs     = figs
        self.name     = name
        self.dynamics = dynamics
        self.file     = file

    def __enter__(self):
        self.fig = figure(figsize=(12,12))
        return self.fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type==None and exc_val==None and exc_tb==None:
            savefig(join(self.figs,f'{splitext(basename(self.file))[0]}-{self.dynamics.name}-{self.name}'))
            return True
        else:
            return False
