import numpy as np

def compute_w_proj(V,W):

    # compute inner matrix that is going to be inverted
    Inner = np.dot(V.T,np.dot(W,V))
    In = np.linalg.inv(Inner)

    # compute actual projection matrix
    P = np.dot(np.dot(V, np.dot(In, V.T)), W)
    return P