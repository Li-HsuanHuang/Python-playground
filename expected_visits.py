# Example of computing expected visits before absorption.
# Absorbing state is arranged so it is on the first row 
# of the probability transition matrix.

import numpy as np

# Probability transition matrix P
# First state in the matrix is an absorbing state.
row1 = np.array([[1,0,0,0,0]])
row2 = np.array([[1/2,0,1/2,0,0]])
row3 = np.array([[0,1/2,0,1/2,0]])
row4 = np.array([[0,0,1/2,0,1/2]])
row5 = np.array([[0,0,0,1,0]])
P = np.concatenate((row1,row2,row3,row4,row5),axis=0)


# Obtain substochatstic matrix Q.
Q = P[1:,1:]

# Obtain the matrix of expected visits from one state to another bofore absorption
M = np.linalg.inv(np.eye(Q.shape[0]) - Q)

# The expected number of visits to get from state i to absoprtion
# This is the same as doing np.dot(M,np.ones(Q.shape[0])).
E = np.sum(M,1)
