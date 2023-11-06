# This is the solution to problem 1
# Written by Dalim Wahby (T0606-9...) and Valeria Grotto (Person number)
# Last update: 2020-11-05

import numpy as np
import maze as mz 


# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 2, 0]
])
# with the convention 
# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze

# Create an environment maze
mz.draw_maze(maze)
