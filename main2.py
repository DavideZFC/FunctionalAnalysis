from classes.poussin_method import poussin_method
from functions.bad_proj_matrix import bad_proj_matrix
from functions.good_basis import good_basis
from functions.inf import inf
import numpy as np

'''
mat = np.eye(k)
mat[0,0] = 0
basis = np.zeros((k,2))
basis[0] = 1

p = poussin_method(mat, basis)
'''


k = 200
d = 30
mat = bad_proj_matrix(k)
print(inf(mat))
basis = good_basis(k,d)

p = poussin_method(mat, basis)
p.optimizer(iter=40)
p.plot_result()
print(np.sum(np.abs(basis)))