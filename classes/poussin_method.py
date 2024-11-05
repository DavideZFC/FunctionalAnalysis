import numpy as np
from functions.inf import inf
import matplotlib.pyplot as plt

eps = 0.00001

class poussin_method:
    def __init__(self, M, basis):
        # here we store the initial projection matrix which we want to improve
        self.M = M

        # orthogonal basis that we use
        self.basis = basis
        self.k = basis.shape[0]
        self.nbasis = basis.shape[1]

        # vector of the coefficients
        self.coef = np.zeros(self.nbasis)

        self.make_checks()

    def get_curr_M(self):
        M = np.copy(self.M)
        for r in range(self.nbasis):
            v = self.basis[:,r]
            v = v[:,np.newaxis]
            M += self.coef[r]*np.dot(v, v.T)
        return M


    def objective_fun(self):
        M = self.get_curr_M()
        return inf(M)
    

    def get_gradient(self, m, j):
        '''
        m = row index of maximal norm

        j = component of the gradient
        '''
        M = self.get_curr_M()

        load = 0
        _, m = inf(M)
        for n in range(self.k):
            if M[n,m] > 0:
                load += self.basis[m,j]*self.basis[n,j]
            else:
                load = self.basis[m,j]*self.basis[n,j]
        return load
    

    def optimizer(self, iter=100, lr=0.005):

        # contain the sequence of values for the objecive function
        self.vals = np.zeros(iter)

        for it in range(iter):
            # current projection matrix
            M = self.get_curr_M()
            value, m = inf(M)
            self.vals[it] = value
            print('current objective: '+str(value))

            grad = np.zeros(self.nbasis)
            for j in range(self.nbasis):
                grad[j] = self.get_gradient(m, j)

            self.coef = self.coef - lr*grad
        
    def plot_result(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.vals)
        plt.xlabel("Iterations")
        plt.ylabel("Infinity norm")
        plt.grid(True)
        plt.savefig('results/plot.pdf')
        plt.show()


    def make_checks(self):
        '''
        make the necessary checks to ensure that the algorithm is implemented in the correct way
        '''
        # check that dimension is correct
        if not (self.M.shape[1] == self.k):
            raise ValueError("wrong dimensions")

        # check that the basis is orthogonal to M
        for j in range(self.nbasis):
            v = np.dot(self.M, self.basis[:,j])
            if np.sum(v**2) > eps:
                raise ValueError("matrix is not orthogonal to v")

        # check that basis is orthonormal
        for i in range(self.nbasis):
            for j in range(self.nbasis):
                delta = np.dot(self.basis[:,i], self.basis[:,j])
                if i==j and (delta-1)**2>eps:
                    raise ValueError("basis is not orthonotmal")
                elif i>j and delta**2>eps:
                    raise ValueError("basis is not orthonotmal")



    