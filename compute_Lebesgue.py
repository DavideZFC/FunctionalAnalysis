import numpy as np
import matplotlib.pyplot as plt

def evaluate_polynomials(d, x):
    """
    Generate a matrix where each row corresponds to the evaluations of the first d standard polynomials on x.

    Parameters:
    d (int): The number of polynomials to evaluate.
    x (np.array): The input array of x values.

    Returns:
    np.array: A matrix of shape (d, len(x)) where each row contains the evaluations of a polynomial.
    """
    return np.array([x**i for i in range(d)])

def plot_polynomials(d, x):
        """
        Plot the first d standard polynomials evaluated at x.

        Parameters:
        d (int): The number of polynomials to plot.
        x (np.array): The input array of x values.
        """
        polynomials = evaluate_polynomials(d, x)
        for i in range(d):
            plt.plot(x, polynomials[i], label=f'x^{i}')
        plt.xlabel('x')
        plt.ylabel('Polynomial value')
        plt.title('Standard Polynomials')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_sequence_of_functions(F, x):
        """
        Plot the first d functions in the sequence defined by f evaluated at x.

        Parameters:
        f (function): The function to evaluate.
        d (int): The number of functions to plot.
        x (np.array): The input array of x values.
        """
        for i in range(d):
            plt.plot(x, F[i], label=f'Function {i}')
        plt.xlabel('x')
        plt.ylabel('Function value')
        plt.title('Sequence of Functions')
        plt.legend()
        plt.grid(True)
        plt.show()

def gram_schmidt(A):
    # Perform QR decomposition on A
    Q, R = np.linalg.qr(A)
    return Q

def inf(matrix):
    """
    Computes the infinity norm of a matrix.

    Parameters:
    matrix (array-like): A 2D list or NumPy array representing the matrix.

    Returns:
    float: The infinity norm of the matrix.
    """
    # Ensure the input is a NumPy array
    matrix = np.array(matrix)
    
    # Compute the infinity norm
    rowsinf = np.sum(np.abs(matrix), axis=1)
    
    infinity_norm = np.max(rowsinf) # np.sort(np.sum(np.abs(matrix), axis=1))[int(k-1)] #
    idx = np.argmax(rowsinf)
    return infinity_norm, idx


k = 50000
d = 100
x = np.linspace(-1, 1, k)
polys = evaluate_polynomials(d, x)
orthogonal_polys = -(k/2)**(1/2)*gram_schmidt(polys.T).T

# plot_sequence_of_functions(orthogonal_polys, x)

k_max = d
projection_matrix = (2/k)*orthogonal_polys.T[:k_max,:] @ orthogonal_polys

print("Projection matrix shape:", projection_matrix.shape)
print(inf(projection_matrix))