# Copyright (C) 2018 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>

def to_binary(w,n=3):
    s=[]
    while len(s)<2**n:
        s.append(w%2)
        w=w//2
    return s[::-1]

def step(rule,state,n=3):
    new_state=[]
    for i in range(len(state)):
        index=0
        for k in range(i-1,i+1):
            index*=2
            if 0<=k and k<len(state):
                index+=state[k]
        new_state.append(rule[index])
    return new_state

if __name__=='__main__':
    w = 110
    #w=30
    r = to_binary(w)[::-1]
    print (r)

    s = [0]*9+[1,0,0]
    #s = [0]*10+[1]+[0]*10
    print (s)
    #step(r,s)
    for t in range(8+1):
        print (t,s)
        s=step(r,s)
