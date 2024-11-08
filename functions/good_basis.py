import numpy as np
from functions.gram_schmidt import gram_schmidt


def good_basis(k, d=1):
    bad_vector = np.ones(k)
    bad_vector[0] = -k**0.5

    nor = np.sum(bad_vector**2)**0.5
    v = bad_vector/nor

    if d>1:
        w = np.copy(bad_vector)
        w[0] = -w[0]
        mat = np.hstack((w.reshape(-1,1), v.reshape(-1,1), np.random.normal(size=(k,d-1))))
        return gram_schmidt(mat)[:,1:]
    
    return v.reshape(-1,1)