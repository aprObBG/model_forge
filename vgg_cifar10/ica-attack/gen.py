import numpy as np
import numpy.linalg as la
import numpy.random as rd

std = 1
N = 5
C = 0.5
W = rd.normal(0, std, [N, N])
mask = np.random.choice([0, 1], size=[N, N], p=[1 - C, C]).astype(np.float32)
W = np.multiply(W, mask)
eigs_W, _ = la.eig(W)
sr_W = np.max(abs(eigs_W))
W = W / sr_W
print(W)
print(sr_W)
