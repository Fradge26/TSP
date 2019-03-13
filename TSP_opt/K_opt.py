import numpy as np
from scipy.spatial import cKDTree
from random import sample
from itertools import permutations
import TSP_opt.helper_functions


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


def yield_neighbours(i, best_path, n_dict, id2index, k):
    for j_node, _ in n_dict[best_path[i]].items():
        j = id2index[j_node]
        k.append(j - 1)
        k.append(j)
        yield k


def close_neighbour_generator(i, best_path, n_dict, id2index, depth, k):
    if depth == 0:
        yield tuple(k)
    else:
        for j_node, _ in n_dict[best_path[i]].items():
            j = id2index[j_node]
            if j > i:
                yield from close_neighbour_generator(j, best_path, n_dict, id2index, depth - 1, k + [j - 1] + [j])


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


def cache_swaps():
    def chunker(seq, size):
        return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

    swap_dict = {}
    for d in range(2, 6):
        indices = list(range(d * 2))
        swaps = []
        for p in permutations(indices[1:-1]):
            for p2 in chunker(p, 2):
                if abs(p2[0] - p2[1]) != 1:
                    break
            else:
                swaps.append((0,) + p + (d*2-1,))
        swap_dict[d] = swaps
    return swap_dict


def swap_generator(k, swap_set):
    for s in swap_set:
        output = []
        for i in range(0, len(k), 2):
            output.append((k[s[i]], k[s[i+1]]))
        yield output


def k_swap(swap, path):
    signs = []
    for i in range(len(swap) - 1):
        signs.append(int(swap[i + 1][0] > swap[i][1]) * 2 - 1)
    new_path = path[:swap[0][0] + 1]
    for i in range(len(swap) - 1):
        new_path += path[swap[i][1]:swap[i+1][0] + signs[i]:signs[i]]
    new_path += path[swap[len(swap)-1][1]:]
    return new_path


def run_kopt(ids, xs, ys, depth, neighbour_limit=None, path=None):
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
    swap_dict = cache_swaps()
    len_path = len(path)
    shorter = 0
    prev = 0
    while improved:
        improved = False
        for i in list(range(roll_i, len_path - 1)) + list(range(1, roll_i)):
            for k in neighbour_generator(i, path, n_dict, id2index, depth - 1, [i - 1, i]):
                improved_i = False
                for idx, swap in enumerate(swap_generator(k, swap_dict[int(len(k) / 2)])):
                    cache_edge_distances(adj_dict, ids2x, ids2y, path, swap)
                    if idx == 0:
                        current_cost = cost_k_edges(adj_dict, path, swap)
                        continue
                    else:
                        swap_cost = cost_k_edges(adj_dict, path, swap)
                    if swap_cost < current_cost - 0.001:
                        shorter += current_cost - swap_cost
                        best_swap = swap
                        current_cost = swap_cost
                        improved_i = True
                        improved = True
                if improved_i:
                    # print(best_swap)
                    path = k_swap(best_swap, path)
                    id2index = dict(zip(path, range(len(path))))
                    roll_i = i + 1
                    if int(shorter / 100) > prev:
                        prev = int(shorter / 100)
                        score = TSP_opt.get_score(path,ids2x,ids2y)
                        TSP_opt.write_submission(path, score, 197769)
                    break
    assert len(path) == len(path)
    return path
