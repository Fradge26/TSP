import numpy as np
from scipy.spatial import cKDTree
from random import sample
import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import permutations


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
                # continue
                yield from close_neighbour_generator(j, best_path, n_dict, id2index, depth - 1, k + [j - 1] + [j])


def gen(indeces, level):
    if level == 0:
        yield tuple(indeces)
    else:
        if indeces:
            init = indeces[-1]
        else:
            init = x0
        for i in range(init, init+delta):
            yield from gen(indeces+[i], level-1)


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
    #def chunker(seq, size):
    #    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

    for s in swap_set:
        output = []
        for i in range(0, len(k), 2):
            output.append((k[s[i]], k[s[i+1]]))
        yield output
        # yield [(k[x[0]], k[x[1]]) for x in chunker([0] + list(s) + [len(k) - 1], 2)]


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
    i_indices = range(1, len(path) - 6)
    roll_i = 0
    adj_dict = {}
    improved = True
    swap_dict = cache_swaps()
    while improved:
        improved = False
        for i in list(np.roll(i_indices, -roll_i)):
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
                        best_swap = swap
                        current_cost = swap_cost
                        improved_i = True
                        improved = True
                if improved_i:
                    # print(best_swap)
                    path = k_swap(best_swap, path)
                    id2index = dict(zip(path, range(len(path))))
                    roll_i = i
                    break
    assert len(path) == len(path)
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


if __name__ == '__main__':
    nodes = 100
    ids = list(range(nodes))
    xs = np.random.randint(0, 10000, nodes, int)
    ys = np.random.randint(0, 10000, nodes, int)
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))

    path = sample(ids[1:], len(ids) - 1)
    path.insert(0, ids[0])
    path.append(ids[0])
    print(get_score(path, ids2x, ids2y))

    path2 = initial_path_from_nearest(ids, xs, ys)
    print(get_score(path2, ids2x, ids2y))

    #start = datetime.datetime.now()
    #best_path = run_3opt(ids, xs, ys, path=path)
    #print(get_score(best_path, ids2x, ids2y), datetime.datetime.now() - start)

    start = datetime.datetime.now()
    best_path = run_kopt(ids, xs, ys, depth=3, neighbour_limit=20, path=path)
    print(get_score(best_path, ids2x, ids2y), datetime.datetime.now() - start)
    plot_path(best_path, True)

    start = datetime.datetime.now()
    best_path = run_kopt(ids, xs, ys, neighbour_limit=20, path=path)
    print(get_score(best_path, ids2x, ids2y), datetime.datetime.now() - start)
    plot_path(best_path, True)
    pass