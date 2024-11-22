import numpy as np
from functions.gram_schmidt import gram_schmidt

def get_poly(k):
    mat = np.zeros((k,k))
    x = 2*np.arange(k)/k-1
    for i in range(k):
        mat[:,i] = x**i/(k**0.5)
    return mat

def legendre_polynomials(k, d0=0, d=20):
    basis_poly = gram_schmidt(get_poly(k))
    return basis_poly[:,d0:d]

def legendre_projector(k, d0=0, d=20):
    basis_poly = gram_schmidt(get_poly(k))
    return np.dot(basis_poly[:,d0:d], basis_poly[:,d0:d].T)