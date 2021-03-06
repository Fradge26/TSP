import numpy as np
from scipy.spatial import cKDTree
from random import sample


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
        for k_node, _ in n_dict[best_path[j]].items():
            k = id2index[k_node]
            if k <= j:
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


def run_3opt(ids, xs, ys, neighbour_limit=None, path=None):
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
    len_path = len(path)
    shorter = 0
    while improved:
        improved = False
        for i in list(range(roll_i, len_path - 1)) + list(range(1, roll_i)):
            for j, k in neighbour_generator(i, path, n_dict, id2index):
                improved_i = False
                n1, n2, n3, n4, n5, n6 = i - 1, i, j - 1, j, k - 1, k
                best_swap = [(n1, n2), (n3, n4), (n5, n6)]
                cache_edge_distances(adj_dict, ids2x, ids2y, path, best_swap)
                current_cost = cost_k_edges(adj_dict, path, best_swap)
                for sol in [[(n1, n2), (n3, n5), (n4, n6)],
                            [(n1, n3), (n2, n4), (n5, n6)],
                            [(n1, n3), (n2, n5), (n4, n6)],
                            [(n1, n4), (n5, n2), (n3, n6)],
                            [(n1, n4), (n5, n3), (n2, n6)],
                            [(n1, n5), (n4, n2), (n3, n6)],
                            [(n1, n5), (n4, n3), (n2, n6)]]:
                    cache_edge_distances(adj_dict, ids2x, ids2y, path, sol)
                    sol_cost = cost_k_edges(adj_dict, path, sol)
                    if sol_cost < current_cost - 0.001:
                        shorter += current_cost - sol_cost
                        print(shorter)
                        best_swap = sol
                        current_cost = sol_cost
                        improved_i = True
                        improved = True
                if improved_i:
                    if best_swap[1][0] > best_swap[0][1]:
                        s2 = 1
                    else:
                        s2 = -1
                    if best_swap[2][0] > best_swap[1][1]:
                        s3 = 1
                    else:
                        s3 = -1
                    path = path[:best_swap[0][0] + 1] + \
                           path[best_swap[0][1]:best_swap[1][0] + s2:s2] + \
                           path[best_swap[1][1]:best_swap[2][0] + s3:s3] + \
                           path[best_swap[2][1]:]
                    id2index = dict(zip(path, range(len(path))))
                    roll_i = i + 1
                    break
    assert len(path) == len(path)
    return path

