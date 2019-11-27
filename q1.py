import sys
import numpy as np

#define constants
grid_size = 4
eta = 0.0002
rho = 0.5
G = 0.75

#debug function. Not used in submission
def print_grid(grid, current_only=False):
    for row in grid:
        string = ''
        for col in row:
            if current_only:
                string += str(col[0]) + '\t'
            else:
                string += str(col) + ', '
        print(string)

def iterate(grid):
    row = 1
    while row < grid_size - 1:
        col = 1
        while col < grid_size - 1:
            #print(str(row) + ' ' + str(col))
            grid[row][col][0] = grid[row-1][col][1]
            grid[row][col][0] += grid[row+1][col][1]
            grid[row][col][0] += grid[row][col-1][1]
            grid[row][col][0] += grid[row][col+1][1]
            grid[row][col][0] -= 4 * grid[row][col][1]
            grid[row][col][0] *= rho
            grid[row][col][0] += 2 * grid[row][col][1]
            grid[row][col][0] -= (1 - eta) * grid[row][col][2]
            grid[row][col][0] /= (1 + eta)

            col += 1
        row += 1
    return grid

def apply_boundary_conditions(grid):
    #apply the first 4 boundary conditions
    i = 1
    while i < grid_size - 1:
        grid[0][i][0] = G * grid[1][i][0]
        grid[grid_size-1][i][0] = G * grid[grid_size-2][i][0]
        grid[i][0][0] = G * grid[i][1][0]
        grid[i][grid_size -1][0] = G * grid[i][grid_size-2][0]
        i += 1

    #corner cases
    grid[0][0][0] = G * grid[1][0][0]
    grid[grid_size-1][0][0] = G * grid[grid_size-2][0][0]
    grid[0][grid_size-1][0] = G * grid[0][grid_size-2][0]
    grid[grid_size-1][grid_size-1][0] = G * grid[grid_size-1][grid_size-2][0]

    return grid

def propagate(grid):
    row = 0
    while row < grid_size:
        col = 0
        while col < grid_size:
            grid[row][col][2] = grid[row][col][1]
            grid[row][col][1] = grid[row][col][0]

            col += 1
        row += 1
    return grid

if __name__ == "__main__":
    # initialize the grid
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.float)
    grid[grid_size//2,grid_size//2,1] = 1

    #get CLAs
    num_iterations = 3
    if len(sys.argv) >= 2:
        num_iterations = int(sys.argv[1])

    for i in range(num_iterations):
        grid = iterate(grid)
        grid = apply_boundary_conditions(grid)
        grid = propagate(grid)
        print(grid[grid_size//2][grid_size//2][0])

    
    

