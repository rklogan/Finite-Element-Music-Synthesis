import sys
import numpy

def process_entry(x,y,grid):
    pass

#basic parameters
grid_size = 4
center = int(grid_size / 2)
num_fes = grid_size * grid_size

#get CLAs
T = sys.argv[1]

#initialize the array
drum_skin = numpy.zeros((grid_size, grid_size), dtype=(numpy.float,3))
drum_skin[center][center] = (1.0, 0.0, 0.0)


for i in range(4):
    for j in range(4):
        for k in range(3):
            print(drum_skin[i][j][k])
            


