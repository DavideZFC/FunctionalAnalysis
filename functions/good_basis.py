import numpy as np

def good_basis(k):
    bad_vector = np.ones(k)
    bad_vector[0] = -k**0.5

    nor = np.sum(bad_vector**2)**0.5
    v = bad_vector/nor
    return v.reshape(-1,1)