import matplotlib.pyplot as plt
import numpy as np


def plot(x):
    color = ['red', 'yellow', 'blue']
    ls = ['-', ':', '--', '-.', '']
    mk = ['.', 'x', '+','o', '*']
    for i in range(len(x)):
        plt.plot(x[i, :].data.cpu().numpy(),
                 marker=mk[i], color=color[i], linewidth=2, linestyle=ls[i], label="x" + str(i))
    plt.legend()
    plt.show()

