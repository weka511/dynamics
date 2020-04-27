# Copyright (C) 2020 Greenweaves Software Limited

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

import heapq, random, abc,math
from enum import Enum, unique

N = 25

class Particle:
    def __init__(self,position=[0,0,0],velocity=[1,1,1],radius=1):
        self.position = [p for p in position]
        self.velocity = [v for v in velocity]
        self.radius   = radius

@unique
class Wall(Enum):
    NORTH = 0
    EAST  = 1
    SOUTH = 2
    WEST  = 3
    FRONT = 4
    BACK  = 5
    
class Event(abc.ABC):
    def __init__(self,t):
        self.t = t
    def __lt__(self,other):
            return self.t<other.t 
        
    @abc.abstractmethod
    def act(self):
        pass
    
class HitsWall(Event):
    def __init__(self,i,wall,t):
        super().__init__(t)
        self.i    = i
        self.wall = wall
    def __str__(self):
        return f'({self.i},{self.wall})\t\t{self.t}'
    def act(self):
        super().act()    
    
class Collision(Event):
    def __init__(self,i,j,t):
        super().__init__(t)
        self.i = i
        self.j = j
        
    def __str__(self):
        return f'({self.i},{self.j})\t\t\t{self.t}'
    
    def act(self):
        super().act()       
      

particles = [Particle() for _ in range(N)]
events = []

for i in range(N):
    for wall in Wall:
        events.append(HitsWall(i,wall,math.inf))
    for j in range(i):
        events.append(Collision(i,j,random.random()))
        
heapq.heapify(events)

while (len(events)>0):
    print(heapq.heappop(events))

#t = 0    
#while True:
    #next_events = heapq.heappop(events)
    #t = next_events.t
    #next_events.action()
    