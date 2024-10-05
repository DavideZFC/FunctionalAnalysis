import numpy as np
import matplotlib.pyplot as plt
from classes.function import function
from functions.gramschmidt import gramschmidt
from functions.plot_functions import plot_functions


N = 100
d = 5

finp = []
for i in range(d):
    fun = function(N)
    f = lambda x: x**i
    fun.forma(f)
    finp.append(fun)

fout = gramschmidt(finp)
plot_functions(fout)

