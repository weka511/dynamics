# -*- coding: utf-8 -*-
# Copyright (C) 2016 Universite de Geneve, Switzerland
# E-mail contact: sebastien.leclaire@etu.unige.ch
#
# The Parity Rule
#

import numpy as np, matplotlib.pyplot as plt,scipy,copy,imageio
from matplotlib import cm

images='./images/'    
# Definition of functions
def readImage(string): # This function works for monochrome BMP only. 
    image =  imageio.imread(string);
    image[image == 255] = 1
    image = image.astype(int) 
    return image # Note that the image output is a numPy array of type "int".

# Main Program

# Program input, i.e. the name of the image "imageName" and the maximum number of iteration "maxIter"
imageName = 'image1.bmp'
maxIter   = 32

# Read the image and store it in the array "image"
image = readImage(imageName) # Note that "image" is a numPy array of type "int".
# Its element are obtained as image[i,j]
# Also, in the array "image" a white pixel correspond to an entry of 1 and a black pixel to an entry of 0.

# Get the shape of the image , i.e. the number of pixels horizontally and vertically. 
# Note that the function shape return a type "tuple" (vertical_size,horizontal_size)
m,n = np.shape(image);

# Print to screen the initial image.
plt.clf()
plt.imshow(image, cmap=cm.gray)
plt.title('Initial image:')
plt.savefig('{0}step_{1}.png'.format(images,0))

# Main loop
for it in range(1,maxIter+1):
    
    imageCopy = copy.copy(image);
    
    for i in range(m):
        for j in range(n):
            image[i][j]=(imageCopy[(i-1)%m][j]+imageCopy[(i+1)%m][j]+imageCopy[i][(j-1)%n]+imageCopy[i][(j+1)%n])%2
    plt.figure()
    plt.clf()
    plt.imshow(image, cmap=cm.gray)
    plt.title('Image after {} iterations:'.format(it))
    plt.savefig('{0}step_{1}.png'.format(images,it))
        
# Print to screen the number of white pixels in the final image
print("The number of white pixels after",it,"iterations is: ", sum(sum(image)))
plt.show()