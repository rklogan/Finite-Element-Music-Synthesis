""" Question 2: Cell by cell Parallelization """
import numpy as np
import numba
from numba import cuda
import sys

#define constants
grid_size = 512
eta = 0.0002
rho = 0.5
G = 0.75

@cuda.jit
def process_interior(grid, output, width, subblock_width):
  index = subblock_width * (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)

  i = index
  #print(i)
  while i < subblock_width + index:
    if i >= (width - 2) * (width - 2): return

    row = (i // (width-2)) + 1
    col = (i % (width-2)) + 1

    grid[row][col][0] = grid[row-1][col][1]
    grid[row][col][0] += grid[row+1][col][1]
    grid[row][col][0] += grid[row][col-1][1]
    grid[row][col][0] += grid[row][col+1][1]
    grid[row][col][0] -= 4 * grid[row][col][1]
    grid[row][col][0] *= rho
    grid[row][col][0] += 2 * grid[row][col][1]
    grid[row][col][0] -= (1 - eta) * grid[row][col][2]
    grid[row][col][0] /= (1 + eta)

    output[0] = grid[width // 2][width // 2][0]
    i += 1

@cuda.jit
def process_edges(grid, width):
  index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
  if index >= (width - 2) * 4: return
  
  if index < (width - 2):
    grid[0][index+1][0] = G * grid[1][index+1][0]
  elif index < 2 * (width - 2):
    grid[width-1][(index % (width-2))+1][0] = G * grid[width -2][(index % (width-2))+1][0]
  elif index < 3 * (width - 2):
    grid[(index % (width-2))+1][0][0] = G * grid[(index % (width-2))+1][1][0]
  else:
    grid[(index % (width-2))+1][width-1][0] = G * grid[(index % (width-2))+1][width-2][0]

@cuda.jit
def process_corners(grid, width):
  index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
  if index >= 4: return

  if index == 0:
    grid[0][0][0] = G * grid[1][0][0]
  elif index == 1:
    grid[width-1][0][0] = G * grid[width-2][0][0]
  elif index == 3:
    grid[0][width-1][0] = G * grid[0][width-2][0]
  else:
    grid[width-1][width-1][0] = G * grid[width-1][width-2][0]

@cuda.jit
def propogate(grid, width, subblock_width):
    index = subblock_width * (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
  
    i = index
    while i < subblock_width + index:
        if i >= width * width: return
    
        row = i // width
        col = i % width

        grid[row][col][2] = grid[row][col][1]
        grid[row][col][1] = grid[row][col][0]

        i += 1

""" MAIN """

#decide how to allocate threads/blocks
num_blocks = 16
threads_per_block = 1024
sub_block_width = 16

#make the grid
grid = np.zeros((grid_size, grid_size, 3), dtype=np.float)
grid[grid_size//2][grid_size//2][1] = np.float(1.0)
grid_d = cuda.to_device(grid)

tmp = np.array([0.0], dtype=np.float)
output = cuda.device_array_like(tmp)

num_iterations = 12
if len(sys.argv) >= 2:
    num_iterations = int(sys.argv[1])

for i in range(num_iterations):
  process_interior[num_blocks,threads_per_block](grid_d, output, grid_size, sub_block_width)
  process_edges[num_blocks,threads_per_block](grid_d, grid_size)
  process_corners[num_blocks,threads_per_block](grid_d, grid_size)
  propogate[num_blocks,threads_per_block](grid_d, grid_size, sub_block_width)
  #print(grid_d.copy_to_host())
  result = output.copy_to_host()
  
  print(result[0].round(6))

  