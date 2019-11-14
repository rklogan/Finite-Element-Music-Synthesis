import sys
import numpy

#parameters from spec
grid_size = 4
center = int(grid_size / 2)
rho = 0.2
eta = 0.2
G = 0.2

#other useful values
num_fes = grid_size * grid_size
scaler = 1 / (1 + eta)              #optimization to reduce the number of divisions

def process_step(drum_skin):
    output = numpy.zeros_like(drum_skin)
    num_rows = len(drum_skin)
    num_cols = len(drum_skin[0])

    for i in range(num_rows):
        for j in range(num_cols):
            #simulation
            #check if we're on an outer edge
            if i == 0 or j == 0 or i == num_rows-1 or j == num_cols-1:
                output[i][j] = (0.0, 0.0, 0.0)
            else:
                #TODO Optimize these calculations
                output[i][j][0] = drum_skin[i-1][j][1]
                output[i][j][0] += drum_skin[i+1][j][1]
                output[i][j][0] += drum_skin[i][j-1][1]
                output[i][j][0] += drum_skin[i][j+1][1]
                output[i][j][0] -= 4 * drum_skin[i][j][1]
                output[i][j][0] *= rho
                output[i][j][0] += 2 * drum_skin[i][j][1]
                output[i][j][0] -= (1-eta) * drum_skin[i][j][2]
                output[i][j][0] *= scaler

                #propogate values
                output[i][j][1] = drum_skin[i][j][0]
                output[i][j][2] = drum_skin[i][j][1]



    return output


#get CLAs
T = 5     #default to 3 seconds
if(len(sys.argv) >= 2):
    T = int(sys.argv[1])


#initialize the array
drum_skin = numpy.zeros((grid_size, grid_size), dtype=(numpy.float32,3))
drum_skin[center][center] = (1.0, 1.0, 1.0)

for i in range(T):
    drum_skin = process_step(drum_skin)
    print(drum_skin)