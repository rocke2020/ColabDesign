# Import libraries
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))


def plot_plddts(plddts, Ls=None, dpi=100, fig=True):
    if fig:
        plt.figure(figsize=(8, 5), dpi=dpi)
    plt.title("Predicted lDDT per position")
    for n, plddt in enumerate(plddts):
        plt.plot(plddt, label=f"model_{n+1}")
    if Ls is not None:
        L_prev = 0
        for L_i in Ls[:-1]:
            L = L_prev + L_i
            L_prev += L_i
            plt.plot([L, L], [0, 100], color="black")
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Predicted lDDT")
    plt.xlabel("Positions")
    return plt


def save_plddts_img(img_file, plddts, Ls=None, dpi=100, fig=True):
    plt = plot_plddts(plddts, Ls, dpi, fig)
    plt.savefig(str(img_file), bbox_inches="tight")
    plt.close()
