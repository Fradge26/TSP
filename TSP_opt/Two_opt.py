import numpy as np
from scipy.spatial import cKDTree
from random import sample
from numba import jit
import copy


def create_neighbours_dict(ids, xs, ys, neighbour_limit):
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))
    neighbour_dict = {}
    kdtree = cKDTree(np.column_stack((xs, ys)))
    for i in ids:
        neighbour_dict[i] = {}
        closest_nodes = kdtree.query([ids2x[i], ids2y[i]], k=neighbour_limit)
        for dist, node in zip(closest_nodes[0][1:], closest_nodes[1][1:]):
            neighbour_dict[i][node] = dist
    return neighbour_dict


def close_neighbour_generator(i, best_path, n_dict, id2index):
    for j_node, _ in n_dict[best_path[i]].items():
        j = id2index[j_node]
        if j <= i:
            continue
        yield j


def all_neighbour_generator(i, best_path, n_dict=None, id2index=None):
    for j in range(i + 2, len(best_path) - 4):
        yield j


@jit(nopython=True)
def hypot(x1, x2, y1, y2):
    return np.hypot((x2-x1), (y2-y1))
    #return ((x1-x2)**2 + (y1-y2)**2)**0.5


def cache_edge_distances(adj_dict, ids2x, ids2y, path, edges):
    for e in edges:
        if not (e[0], e[1]) in adj_dict:
            adj_dict[(e[0], e[1])] = hypot(ids2x[e[0]], ids2x[e[1]], ids2y[e[0]], ids2y[e[1]])
            #adj_dict[(e[0], e[1])] = ((ids2x[e[0]] - ids2x[e[1]]) ** 2 + (ids2y[e[0]] - ids2y[e[1]]) ** 2) ** 0.5


def cost_k_edges(adj_dict, path, edges):
    cost = 0
    for e in edges:
        cost += adj_dict[(path[e[0]], path[e[1]])]
    return cost


def update_score_from_dict(n1, n2, n3, n4, adj_dict):
    return - adj_dict[(n1, n2)] - adj_dict[(n3, n4)] + adj_dict[(n1, n3)] + adj_dict[(n2, n4)]


def run_2opt(ids, xs, ys, neighbour_limit=None, path=None):
    path = copy.copy(path)
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))
    if path is None:
        path = sample(ids[1:], len(ids) - 1)
        path.insert(0, ids[0])
        path.append(ids[0])
    if neighbour_limit is not None:
        n_dict = create_neighbours_dict(ids, xs, ys, neighbour_limit)
        neighbour_generator = close_neighbour_generator
    else:
        n_dict = None
        neighbour_generator = all_neighbour_generator
    id2index = dict(zip(path, range(len(path))))
    roll_i = 0
    adj_dict = {}
    improved = True
    local_best_score = 0
    len_path = len(path)
    dd=0
    while improved:
        improved = False
        for i in list(range(roll_i, len_path - 1)) + list(range(1, roll_i)):
            print(i)
            for j in neighbour_generator(i, path, n_dict, id2index):
                n1, n2, n3, n4 = path[i - 1], path[i], path[j - 1], path[j]
                edges = [(n1, n2), (n3, n4), (n1, n3), (n2, n4)]
                cache_edge_distances(adj_dict, ids2x, ids2y, path, edges)
                change = update_score_from_dict(n1, n2, n3, n4, adj_dict)
                if change < local_best_score - 0.001:

                    local_best_score = change
                    best_i = i
                    best_j = j
                    improved = True
            if improved:
                dd -= local_best_score
                print(dd)

                local_best_score = 0
                path[best_i:best_j] = path[best_j - 1:best_i - 1:-1]
                id2index = dict(zip(path, range(len(path))))
                roll_i = i + 1
                break
    assert len(path) == len(path)
    return path



