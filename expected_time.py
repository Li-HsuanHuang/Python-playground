# Compute expected time to get from one state to another.
# Use Lawler's Chatper 3 Example 2 (Introduction to Stochastic Processes)

import numpy as np

# Transition-rate matrix A
# The diagonal entries are negative. 

row1 = np.array([[-1,1,0,0]])
row2 = np.array([[1,-3,1,1]])
row3 = np.array([[0,1,-2,1]])
row4 = np.array([[0,1,1,-2]])
A = np.concatenate((row1,row2,row3,row4),axis=0)

# Compute the expected time to get from state 0, 1, and 2 to state 3.

Ahat = A[:2,:2]
bhat = np.sum(np.linalg.inv(-Ahat),axis=1)


