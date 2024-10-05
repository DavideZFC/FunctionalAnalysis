import matplotlib.pyplot as plt

def plot_functions(finp):
    d = len(finp)
    for f in finp:
        plt.plot(f.x, f.y)
    plt.show()