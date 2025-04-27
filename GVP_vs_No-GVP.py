import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre

# Funzioni fondamentali

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
num_d = 6
ds = np.logspace(start=3, stop=7, num=num_d, base=2).astype(int)

result_gvp = np.zeros(num_d)
result_nogvp = np.zeros(num_d)
k = 20000

# Griglia equispaziata
x = np.linspace(-1, 1, k)

# Loop principale
for i in range(len(ds)):
    d = ds[i]
    d_regul = int(d * 2)
    k_max = 6*d_regul

    # Polinomi di Legendre
    polys = evaluate_legendre_polynomials(d_regul, x)

    # Matrice filtro GVP (regolarizzazione)
    regul_identity_matrix_gvp = np.eye(d_regul)
    for m in range(d_regul):
        if m >= d:
            regul_identity_matrix_gvp[m, m] = max(1 - (m - d) / d, 0)

    # Matrice filtro NO-GVP (solo proiezione)
    regul_identity_matrix_nogvp = np.diag([1 if m < d else 0 for m in range(d_regul)])

    # Operatore proiezione GVP
    projection_matrix_gvp = (2 / k) * polys.T[:k_max,:] @ regul_identity_matrix_gvp @ polys

    # Operatore proiezione NO-GVP
    projection_matrix_nogvp = (2 / k) * polys.T[:k_max,:] @ regul_identity_matrix_nogvp @ polys

    # Calcolo norma infinito
    matinf_gvp, _ = inf_norm(projection_matrix_gvp)
    matinf_nogvp, _ = inf_norm(projection_matrix_nogvp)

    result_gvp[i] = matinf_gvp
    result_nogvp[i] = matinf_nogvp

    # Plot parziale per ogni d
plt.plot(ds, result_gvp, marker='o', linestyle='dashed', linewidth=1.5, markersize=5, label='GVP')
plt.plot(ds, result_nogvp, marker='s', linestyle='solid', linewidth=1.5, markersize=5, label='No-GVP')

# Finalizzazione grafico
plt.legend()
plt.xlabel('d (Dimensione del polinomio)')
plt.ylabel('Lebesgue constant (infinity norm)')
plt.title('Lebesgue constant: GVP vs No-GVP')
plt.grid(True)
plt.savefig('results/Regul_Lebesgue_comparison.png')
plt.show()