from classes.poussin_method import poussin_method
from functions.bad_proj_matrix import bad_proj_matrix
from functions.good_basis import good_basis
from functions.inf import inf
import numpy as np
import matplotlib.pyplot as plt

'''
mat = np.eye(k)
mat[0,0] = 0
basis = np.zeros((k,2))
basis[0] = 1

p = poussin_method(mat, basis)
'''
plt.figure(figsize=(6, 4))
plt.xlabel("Iterations")
plt.ylabel("Infinity norm")
plt.grid(True)

D = [10, 20, 30, 50]

k = 200
mat = bad_proj_matrix(k)

for d in D:
    basis = good_basis(k,d)

    G = d
    R = 1

    p = poussin_method(mat, basis)
    p.optimizer(iter=40, G=G, R=R)
    # p.plot_result()

    plt.plot(p.vals, label='D = {}'.format(d))
plt.legend()
plt.savefig('results/multiplot.pdf')
plt.show()
