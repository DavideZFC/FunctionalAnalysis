from classes.poussin_method import poussin_method
from functions.bad_proj_matrix import bad_proj_matrix
from functions.good_basis import good_basis
from functions.polynomials import *
from functions.inf import inf
import numpy as np
import matplotlib.pyplot as plt

title = 'exp_LEGENDRE_500_5_10_20_40_exp60'
plt.figure(figsize=(6, 4))
plt.xlabel("Iterations")
plt.ylabel("Infinity norm")
plt.grid(True)

D = [5, 10, 20, 40]

k = 200
# mat = bad_proj_matrix(k)
dd = 10
mat = legendre_projector(k, d0=0,d=dd)

for d in D:
    # basis = good_basis(k,d)
    basis =  legendre_polynomials(k, d0=dd, d=dd+d)

    G = d
    R = 1

    p = poussin_method(mat, basis)
    p.optimizer(iter=1000, G=G, R=R)
    # p.plot_result()

    plt.plot(p.vals, label='D = {}'.format(d))
plt.legend()
plt.savefig('results/{}.pdf'.format(title))
plt.show()
