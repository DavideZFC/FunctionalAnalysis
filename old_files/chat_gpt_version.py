import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre

# Funzioni fondamentali

def evaluate_legendre_polynomials(d, x):
    """
    Evaluta i primi d polinomi di Legendre in un array x.

    Parametri:
    d (int): numero di polinomi.
    x (np.array): array di punti.

    Ritorna:
    np.array: matrice (d, len(x)).
    """
    return np.array([eval_legendre(i, x) for i in range(d)])

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
num_d = 5
num_k = 10
ds = np.logspace(start=3, stop=6, num=num_d, base=2).astype(int)   # gradi d
ks = 10*np.logspace(start=5, stop=11, num=num_k, base=2).astype(int) # numero di punti k

result = np.zeros((num_d, num_k))

# Loop principale
for i in range(len(ds)):
    for j in range(len(ks)):

        k = ks[j]
        d = ds[i]
        d_regul = int(d * 2)

        # Griglia equispaziata
        x = np.linspace(-1, 1, k)

        # Polinomi di Legendre
        polys = evaluate_legendre_polynomials(d_regul, x)

        # Matrice di filtro regolatore
        regul_identity_matrix = np.eye(d_regul)
        for m in range(d_regul):
            if m >= d:
                regul_identity_matrix[m, m] = max(1 - (m - d) / d, 0)

        # Operatore di proiezione filtrato
        # Nota: nessun peso quadratura, quindi normale somma su punti equispaziati
        projection_matrix = (2 / k) * polys.T @ regul_identity_matrix @ polys

        # Calcolo norma infinito
        matinf, _ = inf_norm(projection_matrix)

        result[i, j] = matinf

    # Plot parziale per ogni d
    plt.plot(ks, result[i], marker='o', linestyle='dashed', linewidth=1.5, markersize=5, label='d = {}'.format(ds[i]))

# Finalizzazione grafico
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.xlabel('k (Number of grid points)')
plt.ylabel('Lebesgue constant (infinity norm)')
plt.title('Lebesgue constant vs grid size (continuous case)')
plt.grid(True)
plt.savefig('results/Regul_Lebesgue_continuous.png')
plt.show()
