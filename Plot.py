from __future__ import print_function
import pylab as plt
from  matplotlib.patches import  Rectangle
from matplotlib.collections import PatchCollection


def plot_grid(grid, title):
    # Various preparations
    n = len(grid)
    m = len(grid[0])
    plt.rcParams['figure.figsize'] = int((m * 5) / n), 5
    plt.subplots_adjust(bottom=0.02, top=0.92, left=0.02, right=0.98)
    graph = plt.subplot(1, 1, 1)
    graph.get_xaxis().set_visible(False)
    graph.get_yaxis().set_visible(False)
    plt.axis([0, len(grid[0]), n, 0])
    plt.suptitle(title, fontsize=24)

    # Black squares in prohibited boxes
    CN = []
    CB = []
    CG = []
    for i in range(n):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                CN.append(Rectangle((j, i), 1, 1, color="black"))
            elif grid[i][j] == -1:
                CB.append(Rectangle((j, i), 1, 1, color="white"))
            elif grid[i][j] == 0:
                CG.append(Rectangle((j, i), 1, 1, color="grey"))
    if CN != []:
        graph.add_collection(PatchCollection(CN, match_original=True))
    if CB != []:
        graph.add_collection(PatchCollection(CB, match_original=True))
    if CG != []:
        graph.add_collection(PatchCollection(CG, match_original=True))
    plt.show()


if __name__ == '__main__':
    pass