import numpy as np
import os
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
import itertools
import datetime


def create_neighbours_dict(ids, xs, ys, neighbour_limit):
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))
    neighbour_dict = {}
    kdtree = spatial.cKDTree([xs, ys])
    for i in ids:
        neighbour_dict[i] = {}
        closest_nodes = kdtree.query((ids2x[i], ids2y[i]), k=neighbour_limit)
        for dist, node in zip(closest_nodes[0][1:], closest_nodes[1][1:]):
            neighbour_dict[i][node] = dist
    return neighbour_dict


def close_neighbour_generator(i, best_path, n_dict, id2index):
    for j_node, _ in n_dict[best_path[i]].items():
        j = id2index[j_node]
        for k_node, _ in n_dict[best_path[j]].items():
            k = id2index[k_node]
            i, j, k = sorted([i, j, k])
            if j - i == 1 or k - j == 1 or len({i, j, k}) != 3:
                continue
            yield j, k


def all_neighbour_generator(i, best_path, n_dict=None, id2index=None):
    for j in range(i + 2, len(best_path) - 4):
        for k in range(j + 2, len(best_path) - 2):
            yield j, k


def cache_edge_distances(adj_dict, ids2x, ids2y, path, edges):
    for e in edges:
        if not (path[e[0]], path[e[1]]) in adj_dict:
            adj_dict[(path[e[0]], path[e[1]])] = ((ids2x[path[e[0]]] - ids2x[path[e[1]]]) ** 2 +
                                                  (ids2y[path[e[0]]] - ids2y[path[e[1]]]) ** 2) ** 0.5


def cost_k_edges(adj_dict, path, edges):
    cost = 0
    for e in edges:
        cost += adj_dict[(path[e[0]], path[e[1]])]
    return cost


def run_3opt(ids, xs, ys, best_path=None, neighbour_limit=None):
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))
    if best_path is None:
        best_path = np.random.shuffle(ids[1:])
        best_path.insert(0, ids[0])
        best_path.append(ids[0])
    if neighbour_limit is not None:
        n_dict = create_neighbours_dict(ids, xs, ys, neighbour_limit)
        neighbour_generator = close_neighbour_generator
    else:
        neighbour_generator = all_neighbour_generator
    id2index = dict(zip(best_path, range(len(best_path))))
    i_indices = range(1, len(best_path) - 6)
    roll_i = 0
    adj_dict = {}
    improvement = True
    while improvement:
        improvement = False
        for i in list(np.roll(i_indices, -roll_i)):
            for j, k in neighbour_generator(i,best_path,n_dict,id2index):
                improvement = False
                n1, n2, n3, n4, n5, n6 = i - 1, i, j - 1, j, k - 1, k
                best_swap = [(n1, n2), (n3, n4), (n5, n6)]
                cache_edge_distances(adj_dict, ids2x, ids2y, best_path, best_swap)
                current_cost = cost_k_edges(adj_dict, best_path, best_swap)
                for sol in [[(n1, n2), (n3, n5), (n4, n6)],
                            [(n1, n3), (n2, n4), (n5, n6)],
                            [(n1, n3), (n2, n5), (n4, n6)],
                            [(n1, n4), (n5, n2), (n3, n6)],
                            [(n1, n4), (n5, n3), (n2, n6)],
                            [(n1, n5), (n4, n2), (n3, n6)],
                            [(n1, n5), (n4, n3), (n2, n6)]]:
                    cache_edge_distances(adj_dict, ids2x, ids2y, best_path, sol)
                    sol_cost = cost_k_edges(adj_dict, best_path, sol)
                    if sol_cost < current_cost:
                        best_swap = sol
                        current_cost = sol_cost
                        improvement = True
                if improvement:
                    if best_swap[1][0] > best_swap[0][1]:
                        s2 = 1
                    else:
                        s2 = -1
                    if best_swap[2][0] > best_swap[1][1]:
                        s3 = 1
                    else:
                        s3 = -1
                    best_path = best_path[:best_swap[0][0] + 1] + \
                                best_path[best_swap[0][1]:best_swap[1][0] + s2:s2] + \
                                best_path[best_swap[1][1]:best_swap[2][0] + s3:s3] + \
                                best_path[best_swap[2][1]:]
                    roll_i = i
                    break
            else:
                continue
            break
    assert len(best_path) == len(best_path)
    return best_path
