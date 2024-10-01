import numpy as np
import matplotlib.pyplot as plt
from classes.function import function

f = lambda x: x**2
g = lambda x: x**0
N = 100
fun = function(N)
fun.forma(f)
gun = function(N)
gun.forma(f)

kkk = fun.project(gun)

plt.plot(kkk.x, kkk.y)
plt.show()

