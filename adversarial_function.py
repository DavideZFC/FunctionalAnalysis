import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre

# Funzioni fondamentali

def sign(y):
    return np.where(y > 0, 1, -1)

def evaluate_legendre_polynomials(d, x):
    """
    Evaluta i primi d polinomi di Legendre in un array x.
    """
    return np.array([np.sqrt((2*i+1)/2) * eval_legendre(i, x) for i in range(d)])

def plot_sequence_of_functions(F, x, title="Sequence of Functions"):
    d = len(F)
    for i in range(d):
        plt.plot(x, F[i], label=f'Function {i}')
    plt.xlabel('x')
    plt.ylabel('Function value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def inf_norm(matrix):
    """
    Calcola la norma infinito della matrice.
    """
    matrix = np.array(matrix)
    rows_inf = np.sum(np.abs(matrix), axis=1)
    infinity_norm = np.max(rows_inf)
    idx = np.argmax(rows_inf)
    return infinity_norm, idx

# Parametri dell'esperimento
d = 100
k = 20000

# Griglia equispaziata
x = np.linspace(-1, 1, k)


k_max = 1000

# Polinomi di Legendre
polys = evaluate_legendre_polynomials(d, x)

# Operatore proiezione NO-GVP
projection_matrix_nogvp = (2 / k) * polys.T[:k_max,:] @ polys

# Calcolo norma infinito
matinf_nogvp, row = inf_norm(projection_matrix_nogvp)

print('lebesgue constant (infinity norm) NO-GVP:', matinf_nogvp)


plt.plot(x, sign(projection_matrix_nogvp[row,:]), linewidth=1.5, label='adversarial function NO-GVP')

# Finalizzazione grafico
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig('results/adversarial_function.png')
plt.show()