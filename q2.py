import sys
import numpy as np
import numba
from numba import cuda
import q1

#define constants
grid_size = 4
eta = 0.0002
rho = 0.5
G = 0.75

@cuda.jit
def iterate(grid, output_grid, width, corner_barrier):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if index >= width * width: return

    row = index // width
    col = index % width

    thread_is_boundary = row == 0 or col == 0 or row == width-1 or col == width-1

    #process interior
    if not thread_is_boundary:
        grid[row][col][0] = grid[row-1][col][1]
        grid[row][col][0] += grid[row+1][col][1]
        grid[row][col][0] += grid[row][col-1][1]
        grid[row][col][0] += grid[row][col+1][1]
        grid[row][col][0] -= 4 * grid[row][col][1]
        grid[row][col][0] *= rho
        grid[row][col][0] += 2 * grid[row][col][1]
        grid[row][col][0] -= (1 - eta) * grid[row][col][2]
        grid[row][col][0] /= (1 + eta)

    #print('interior done')
    #boundary nodes need to wait here for the interior to be written
    cuda.syncthreads()

    thread_is_corner = (row == 0 and col == 0) or (row == 0 and col == width-1) or (row == width-1 and col ==0) or (row == width-1 and col == width-1)

    #boundary nodes can now start to work
    if thread_is_boundary and not thread_is_corner:
        if row == 0:
            grid[0][col][0] = G * grid[1][col][0]
        elif row == width - 1:
            grid[width-1][col][0] = G * grid[width-2][col][0]
        elif col == 0:
            grid[row][0][0] = G * grid[row][1][0]
        elif col == width - 1:
            grid[row][width-1][0] = G * grid[row][width-2][0]

    if (row == 1 and col == 0) or (row == width-2 and col == 0) or (row == 0 and col == width-2) or (row == width-1 and col == width-2):
        cuda.atomic.add(corner_barrier, 0, -1)
    
    #print(corner_barrier[0])
            
    if thread_is_corner:
        #corners will busy-wait until their input data has been written
        i = 0
        while(corner_barrier[0] != 0):
            i = (i + 1) %1024        

        if row == 0 and col ==0:
            grid[0][0][0] = G * grid[1][0][0]
        elif row == width-1 and col == 0:
            grid[width-1][0][0] = G * grid[width-2][0][0]
        elif row == 0 and col == width-1:
            grid[0][width-1][0] = G * grid[0][width-2][0]
        elif row == width-1 and col == width-1:
            grid[width-1][width-1][0] = G * grid[width-1][width-2][0]

    #print('all done')
    #all threads can now propogate their values back
    output_grid[row][col][2] = grid[row][col][1]
    output_grid[row][col][1] = grid[row][col][0]
    output_grid[row][col][0] = grid[row][col][0]


if __name__ == "__main__":
    # initialize the grid
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.float)
    grid[grid_size//2,grid_size//2,1] = 1

    #get CLAs
    num_iterations = 3
    if len(sys.argv) >= 2:
        num_iterations = int(sys.argv[1])

    for i in range(num_iterations):
        barrier = np.array([4])
        barrier_d = cuda.to_device(barrier)
        grid_d = cuda.to_device(grid)
        output_grid_d = cuda.device_array_like(grid)

        iterate[1,grid_size * grid_size](grid_d, output_grid_d, np.int(grid_size), barrier_d)

        grid = output_grid_d.copy_to_host()
        print(grid[grid_size//2][grid_size//2][0])

    
    

