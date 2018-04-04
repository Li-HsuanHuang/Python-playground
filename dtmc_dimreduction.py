import numpy as np
import scipy.linalg as linalg
r1 = np.repeat(0.25,4)
r2 = np.array([1/3,1/3,0,1/3])
r3 = np.array([0,1/2,1/2,0])
r4 = np.array([0,3/5,0,2/5])

M = np.vstack([r1,r2,r3,r4])
_,vl, _ =linalg.eig(M,left=True)

pivec = np.abs(vl[:,0])/np.sum(np.abs(vl[:,0]))

newvec = pivec.copy()
newvec[0:2] = pivec[0:2]/np.sum(pivec[0:2])
newvec[2:4] = pivec[2:4]/np.sum(pivec[2:4])

M1 =(M.T*newvec).T


R = np.random.randint(5,size=(6,6))
#w = np.zeros((2,2))
n = R.shape[0]
p = 2
newmat = np.zeros((n//p,n//p))
for i in range(0,(n//p)):
    for j in range(0,(n//p)):
        newmat[i,j] = np.sum(R[(p*i):(p*(i+1)),(p*j):(p*(j+1))])



