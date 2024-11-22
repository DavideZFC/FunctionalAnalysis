import numpy as np


def bad_proj_matrix(k):
    bad_vector = np.ones(k)
    bad_vector[0] = k**0.5

    nor = np.sum(bad_vector**2)**0.5
    v = bad_vector/nor
    return np.outer(v,v)