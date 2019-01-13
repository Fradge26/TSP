import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_path(path, ids2x, ids2y, annotate):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [ids2x[node] for node in path]
    y = [ids2y[node] for node in path]
    line = Line2D(x, y, linewidth=0.2)
    ax.add_line(line)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    if annotate:
        for i, label in enumerate(path):
            ax.annotate(label, (ids2x[label], ids2y[label]), size=5)
    plt.show()
    #fig.savefig(f'Figures/{sample_size}_{int(score)}.png', dpi=1000)


def initial_path_from_nearest(ids, xs, ys):
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))
    kdtree = cKDTree(np.column_stack((xs, ys)))
    path = [0]
    unvisited_nodes = set(ids)
    unvisited_nodes.remove(0)
    while unvisited_nodes:
        for a in [4, 16, 100, 1000, 10000, 50000, 197000]:
            closest_nodes = kdtree.query((ids2x[path[-1]], ids2y[path[-1]]), k=a)
            overlap = set(closest_nodes[1]) & unvisited_nodes
            if len(overlap) > 0:
                break
        for node in closest_nodes[1]:
            if node in unvisited_nodes:
                path.append(node)
                unvisited_nodes.remove(node)
                break
    path.append(0)
    return path


def get_score(path, ids2x, ids2y):
    score = 0
    for i, a in enumerate(path[1:]):
        x1 = ids2x[path[i + 1]]
        y1 = ids2y[path[i + 1]]
        x2 = ids2x[path[i]]
        y2 = ids2y[path[i]]
        score += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return score
