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
def iterate(grid, width, corner_barrier, num_iterations):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if index >= width * width: return

    row = index // width
    col = index % width

    thread_is_boundary = row == 0 or col == 0 or row == width-1 or col == width-1




    iteration = 0
    while iteration < num_iterations:
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
        
        if thread_is_corner:
            #corners will busy-wait until their input data has been written
            while(corner_barrier[0] != 0):
                pass
            cuda.atomic.add(corner_barrier,0,-1)        

            if row == 0 and col ==0:
                grid[0][0][0] = G * grid[1][0][0]
            elif row == width-1 and col == 0:
                grid[width-1][0][0] = G * grid[width-2][0][0]
            elif row == 0 and col == width-1:
                grid[0][width-1][0] = G * grid[0][width-2][0]
            elif row == width-1 and col == width-1:
                grid[width-1][width-1][0] = G * grid[width-1][width-2][0]

        #all threads can now propogate their values back
        grid[row][col][2] = grid[row][col][1]
        grid[row][col][1] = grid[row][col][0]
        grid[row][col][0] = grid[row][col][0]

        if row == width // 2 and col == width // 2:
            print(grid[row][col][0])
            while(cuda.atomic.compare_and_swap(corner_barrier, -4, 4) == 4):
                pass

        cuda.syncthreads()

        iteration += 1


if __name__ == "__main__":
    # initialize the grid
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.float)
    grid[grid_size//2,grid_size//2,1] = 1

    #get CLAs
    num_iterations = 3
    if len(sys.argv) >= 2:
        num_iterations = int(sys.argv[1])

    barrier = np.array([4])
    barrier_d = cuda.to_device(barrier)
    grid_d = cuda.to_device(grid)

    iterate[1,grid_size * grid_size](grid_d, np.int(grid_size), barrier_d, np.int(num_iterations))


    
    

