import sys
import numpy

#basic parameters
grid_size = 4
center = int(grid_size / 2)
num_fes = grid_size * grid_size

#get CLAs
T = sys.argv[1]

#initialize the array
drum_skin = numpy.zeros((grid_size, grid_size), dtype=numpy.float)
drum_skin[center][center] = 1.0





