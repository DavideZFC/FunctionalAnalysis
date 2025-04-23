import numpy as np
import matplotlib.pyplot as plt
from classes.function import function
from functions.functional_gram_schmidt import functional_gram_schmidt
from functions.plot_functions import plot_functions


N = 10000
d = 5

finp = []
for i in range(d):
    fun = function(N, -1, 1)
    f = lambda x: x**i
    fun.forma(f)
    finp.append(fun)

fout = functional_gram_schmidt(finp)
plot_functions(fout)