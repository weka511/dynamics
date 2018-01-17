# Simple Python implementation of a Barnes-Hut galaxy simulator.
# This file is part of the exercise series of the University of Geneva's
# MOOC "Simulation and Modeling of Natural Processes".
#
# Author: Jonas Latt
# E-mail contact: jonas.latt@unige.ch
# Important: don't send questions about this code to the above e-mail address.
# They will remain unanswered. Instead, use the resources of the MOOC.
# 
# Copyright (C) 2016 University of Geneva
# 24 rue du Général-Dufour
# CH - 1211 Genève 4
# Switzerland
#
# This code is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# The code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy,numpy as np,matplotlib.pyplot as plt,mpl_toolkits.mplot3d

images='./images/' 

class Node:
# A node represents a body if it is an endnote (i.e. if node.child is None)
# or an abstract node of the quad-tree if it has child.

    def __init__(self, m, x, y,z):
        '''
        The initializer creates a child-less node (an actual body).
        Instead of storing the position of a node, we store the mass times
        position, m_pos. This makes it easier to update the center-of-mass.
        '''
        self.m = m
        self.m_pos = m * np.array([x, y,z])  #mass*position
        self.momentum = np.array([0., 0.,0.])
        self.child = None

    def into_next_octant(self):
        '''
        Place node into next-level octant and return the octant number.
        '''
        self.s = 0.5 * self.s   # s: side-length of current octant.
        return self._subdivide(1) + 2*self._subdivide(0) #self._subdivide(2) + 2*self._subdivide(1) + 4*self._subdivide(0)

    def pos(self):
        '''
        Physical position of node, independent of currently active octant.
        '''
        return self.m_pos / self.m

    def reset_to_0th_octant(self):
        '''
        Re-position the node to the level-0 octant (full domain).
        Side-length of the level-0 octant is 1.
        '''
        self.s = 1.0
        self.relpos = self.pos().copy() # Relative position inside the octant is equal to physical position.

    def dist(self, other):
        '''
        Distance between present node and another node.
        '''
        return np.linalg.norm(other.pos() - self.pos())

    def force_on(self, other):
        '''
        Force which the present node is exerting on a given body.
        '''       
        cutoff_dist = 0.002 # To avoid numerical instabilities, introduce a short-distance cutoff.
        d = self.dist(other)
        if d < cutoff_dist:
            return np.array([0., 0., 0.])
        else: # Gravitational force goes like 1/r**2           
            return (self.pos() - other.pos()) * (self.m*other.m / d**3)

    def _subdivide(self, i):
        '''
        Place node into next-level octant along direction i and recompute
        the relative position relpos of the node inside this octant.
        '''
        self.relpos[i] *= 2.0
        if self.relpos[i] < 1.0:
            octant = 0
        else:
            octant = 1
            self.relpos[i] -= 1.0
        return octant


def add(body, node):
    '''
    Barnes-Hut algorithm: Creation of the quad-tree. This function adds
    a new body into a quad-tree node. Returns an updated version of the node.
    '''
    # 1. If node n does not contain a body, put the new body b here.
    new_node = body if node is None else None
    
    # To limit the recursion depth, set a lower limit for the size of octant.
    smallest_octant = 1.e-4
    if node is not None and node.s > smallest_octant:
        # 3. If node n is an external node, then the new body b is in conflict
        #    with a body already present in this region. ...
        if node.child is None:
            new_node = copy.deepcopy(node)
        #    ... Subdivide the region further by creating eight children
            new_node.child = [None for i in range(8)]
        #    ... And to start with, insert the already present body recursively
        #        into the appropriate octant.
            octant = node.into_next_octant()
            new_node.child[octant] = node
        # 2. If node n is an internal node, we don't to modify its child.
        else:
            new_node = node

        # 2. and 3. If node n is or has become an internal node ...
        #           ... update its mass and "center-of-mass times mass".
        new_node.m += body.m
        new_node.m_pos += body.m_pos
        # ... and recursively add the new body into the appropriate octant.
        octant = body.into_next_octant()
        new_node.child[octant] = add(body, new_node.child[octant])
    return new_node


def force_on(body, node, theta):
    '''
    Barnes-Hut algorithm: usage of the quad-tree. This function computes
    the net force on a body exerted by all bodies in node "node".
    Note how the code is shorter and more expressive than the human-language
    description of the algorithm.
    '''
    # 1. If the current node is an external node, 
    #    calculate the force exerted by the current node on b.
    if node.child is None:
        return node.force_on(body)

    # 2. Otherwise, calculate the ratio s/d. If s/d < θ, treat this internal
    #    node as a single body, and calculate the force it exerts on body b.
    if node.s < node.dist(body) * theta:
        return node.force_on(body)

    # 3. Otherwise, run the procedure recursively on each child.
    return sum(force_on(body, c, theta) for c in node.child if c is not None)


def verlet(bodies, root, theta, G, dt):
    '''
    Execute a time iteration according to the Verlet algorithm.
    '''
    for body in bodies:
        force = G * force_on(body, root, theta)
        body.momentum += dt * force
        body.m_pos += dt * body.momentum 


def plot_bodies3(bodies, i):
    '''
    Write an image representing the current position of the bodies.
    To create a movie with avconv or ffmpeg use the following command:
    ffmpeg -r 15 -i bodies3D_%06d.png -q:v 0 bodies3D.avi
    '''
    ax = plt.gcf().add_subplot(111, aspect='equal', projection='3d')
    ax.scatter([b.pos()[0] for b in bodies], \
    [b.pos()[1] for b in bodies], [b.pos()[2] for b in bodies])
    ax.set_xlim([0., 1.0])
    ax.set_ylim([0., 1.0])
    ax.set_zlim([0., 1.0])    
    plt.gcf().savefig('{0}bodies3D_{1:06}.png'.format(images,i))
    

# Theta-criterion of the Barnes-Hut algorithm.
theta = 0.5
# Mass of a body.
mass = 1.0
# Initially, the bodies are distributed inside a circle of radius ini_radius.
ini_radius = 0.1
# Initial maximum velocity of the bodies.
inivel = 0.1
# The "gravitational constant" is chosen so as to get a pleasant output.
G = 4.e-6
# Discrete time step.
dt = 1.e-3
# Number of bodies (the actual number is smaller, because all bodies
# outside the initial radius are removed).
numbodies = 1000
# Number of time-iterations executed by the program.
max_iter = 501#10000
# Frequency at which PNG images are written.
img_iter = 20

if __name__=='__main__':
    # The pseudo-random number generator is initialized at a deterministic # value,
    # for proper validation of the output for the exercise series.  random.seed(1)
    # x-, y-pos, and z-pos are initialized to a cube with side-length 2*ini_radius.
    np.random.seed(1)
    posx = np.random.random(numbodies) *2.*ini_radius + 0.5-ini_radius
    posy = np.random.random(numbodies) *2.*ini_radius + 0.5-ini_radius
    posz = np.random.random(numbodies) *2.*ini_radius + 0.5-ini_radius
    
    # We only keep the bodies inside a sphere of radius ini_radius.
    bodies = [ Node(mass, px, py,pz) for (px,py,pz) in zip(posx, posy,posz) 
                   if (px-0.5)**2 + (py-0.5)**2 + (pz-0.5)**2< ini_radius**2 ]
    
    
    # Initially, the bodies have a radial velocity of an amplitude proportional to
    # the distance from the center. This induces a rotational motion creating a
    # "galaxy-like" impression.
    for body in bodies: 
        r = body.pos() - np.array([0.5, 0.5, body.pos()[2] ])
        body.momentum = np.array([-r[1], r[0], 0.]) *  mass*inivel*np.linalg.norm(r)/ini_radius
    
    # Principal loop over time iterations.
    for i in range(max_iter):
        # The oct-tree is recomputed at each iteration.
        root = None
        for body in bodies:
            body.reset_to_0th_octant()
            root = add(body, root)
        
        verlet(bodies, root, theta, G, dt) # Compute forces and advance bodies.
               
        if i%img_iter==0:
            print("Writing images at iteration {0}".format(i))
            plot_bodies3(bodies, i//img_iter)
        print ('{0},{1}'.format(i,bodies[0].m_pos[2]))
