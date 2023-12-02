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

'''Some functions to faciliate the use of XKCD colours'''


def generate_colour_names(reverse=True):
    '''
    Allow iteration through XKCD colours

    Parameters:
        reverse If set to true, iteration starts at most frequent colour
    '''
    order = reversed if reverse else lambda x: x
    with open('rgb.txt') as xkcd:
        for line in order(list(xkcd)):
            parts = line.rstrip().split('\t#')
            if len(parts)>1:
                yield f'xkcd:{parts[0]}'

def create_colour_names(n=None):
    Product = []
    for i,c in enumerate(generate_colour_names()):
        Product.append(c)
        if n != None and i>n: break

    return Product

if __name__=='__main__':
    for c in generate_colour_names():
        print (c)
