import numpy as np
import matplotlib.pyplot as plt
from classes.function import function
from functions.functional_gram_schmidt import functional_gram_schmidt
from functions.plot_functions import plot_functions


N = 100
d = 5

finp = []
for i in range(d):
    fun = function(N)
    f = lambda x: x**i
    fun.forma(f)
    finp.append(fun)

fout = functional_gram_schmidt(finp)
plot_functions(fout)

