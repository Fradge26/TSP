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


def get_best_submission(sample_size):
    all_files = [f for f in os.listdir("./Submissions/") if os.path.isfile(os.path.join("./Submissions/", f))]
    valid_files = sorted([int(f.split("_")[1].split(".")[0]) for f in all_files if f.split("_")[0] == str(sample_size)])
    if len(valid_files) == 0:
        return False
    else:
        best_file = f"./Submissions/{sample_size}_{valid_files[0]}.csv"
        path = list(np.genfromtxt(best_file, delimiter=',', skip_header=True, dtype=int))
        return path


def write_submission(path, score):
    path2 = deepcopy(path)
    path2.insert(0, 'path')
    np.savetxt(f'Submissions/{sample_size}_{int(score)}.csv', path2, fmt='%s')


def plot_path(sample_size, path, score, annotate):
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
            ax.annotate(label, (ids2x[label], ids2y[label]), size=1)
    # plt.show()
    fig.savefig(f'Figures/{sample_size}_{int(score)}.png', dpi=1000)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def is_prime(n):
    if n == 2 or n == 3: return True
    if n < 2 or n % 2 == 0: return False
    if n < 9: return True
    if n % 3 == 0: return False
    r = int(n ** 0.5)
    f = 5
    while f <= r:
        if n % f == 0: return False
        if n % (f + 2) == 0: return False
        f += 6
    return True


def get_score(path, ids2x, ids2y):
    score = 0
    for i, a in enumerate(path[1:]):
        x1 = ids2x[path[i + 1]]
        y1 = ids2y[path[i + 1]]
        x2 = ids2x[path[i]]
        y2 = ids2y[path[i]]
        score += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return score


def get_score_prime(path, ids2x, ids2y):
    score = 0
    for i, a in enumerate(path[1:]):
        if ((i + 1) % 10 == 0) and not is_prime(a):
            penalty = 1.1
        else:
            penalty = 1.0
        x1 = ids2x[path[i + 1]]
        y1 = ids2y[path[i + 1]]
        x2 = ids2x[path[i]]
        y2 = ids2y[path[i]]
        score += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * penalty
    return score


def update_score(score, n1, n2, n3, n4, ids2x, ids2y):
    score -= ((ids2x[n1] - ids2x[n2]) ** 2 + (ids2y[n1] - ids2y[n2]) ** 2) ** 0.5
    score -= ((ids2x[n3] - ids2x[n4]) ** 2 + (ids2y[n3] - ids2y[n4]) ** 2) ** 0.5
    score += ((ids2x[n1] - ids2x[n3]) ** 2 + (ids2y[n1] - ids2y[n3]) ** 2) ** 0.5
    score += ((ids2x[n2] - ids2x[n4]) ** 2 + (ids2y[n2] - ids2y[n4]) ** 2) ** 0.5
    return score


def update_score_from_dict(n1, n2, n3, n4, adj_dict):
    return - adj_dict[(n1, n2)] - adj_dict[(n3, n4)] + adj_dict[(n1, n3)] + adj_dict[(n2, n4)]


def cache_dist(n1, n2, n3, n4, adj_dict, ids2x, ids2y):
    if not (n1, n2) in adj_dict:
        adj_dict[(n1, n2)] = ((ids2x[n1] - ids2x[n2]) ** 2 + (ids2y[n1] - ids2y[n2]) ** 2) ** 0.5
    if not (n3, n4) in adj_dict:
        adj_dict[(n3, n4)] = ((ids2x[n3] - ids2x[n4]) ** 2 + (ids2y[n3] - ids2y[n4]) ** 2) ** 0.5
    if not (n1, n3) in adj_dict:
        adj_dict[(n1, n3)] = ((ids2x[n1] - ids2x[n3]) ** 2 + (ids2y[n1] - ids2y[n3]) ** 2) ** 0.5
    if not (n2, n4) in adj_dict:
        adj_dict[(n2, n4)] = ((ids2x[n2] - ids2x[n4]) ** 2 + (ids2y[n2] - ids2y[n4]) ** 2) ** 0.5


def cache_edge_distances(adj_dict, ids2x, ids2y, path, edges):
    for e in edges:
        if not (e[0], e[1]) in adj_dict:
            adj_dict[(path[e[0]], path[e[1]])] = ((ids2x[path[e[0]]] - ids2x[path[e[1]]]) ** 2 +
                                                  (ids2y[path[e[0]]] - ids2y[path[e[1]]]) ** 2) ** 0.5


def cost_k_edges(adj_dict, route, edges):
    cost = 0
    for e in edges:
        cost += adj_dict[(route[e[0]], route[e[1]])]
    return cost


def initial_path_from_nearest(ids, coords):
    kdtree = spatial.KDTree(coords)
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


def find_crosses(path, score_func):
    '''
    obselete
    '''

    def ccw(a, b, c):
        return (ids2y[c] - ids2y[a]) * (ids2x[b] - ids2x[a]) > (ids2y[b] - ids2y[a]) * (ids2x[c] - ids2x[a])
        # return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)

    # Return true if line segments AB and CD intersect
    def intersect(a, b, c, d):
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    improvement = True
    best_path = path
    best_score = score_func(path)
    while improvement:
        improvement = False
        for i in range(1, len(best_path)):
            line1_node1 = best_path[i - 1]
            line1_node2 = best_path[i]
            for k in range(i + 1, len(best_path)):
                line2_node1 = best_path[k - 1]
                line2_node2 = best_path[k]
                if len(set((line1_node1, line1_node2, line2_node1, line2_node2))) == 4 and \
                        intersect(line1_node1, line1_node2, line2_node1, line2_node2):
                    # print(line1_node1, line1_node2, line2_node1, line2_node2)
                    new_score = update_score(best_score, best_path[i - 1], best_path[i], best_path[k - 1], best_path[k])
                    if new_score < best_score:
                        best_score = new_score
                        best_path = swap_2opt(best_path, i, k)
                        improvement = True
                        print(new_score, line1_node1, line1_node2, line2_node1, line2_node2)
                        if plots:
                            plot_path(best_path, best_score)
                        write_submission(best_path, best_score)
                        break
                    else:
                        print("Err", line1_node1, line1_node2, line2_node1, line2_node2)
            if improvement:
                break
    return best_path


def run_2opt(path, score_func):
    """
    improves an existing path using the 2-opt swap until no improved path is found
    best path found will differ depending of the start node of the list of nodes
        representing the input tour
    returns the best path found
    path - path to improve
    """
    improvement = True
    best_path = path
    best_score = score_func(path)
    while improvement:
        improvement = False
        for i in range(1, len(best_path) - 1):
            # print(i)
            for k in range(i + 1, len(best_path) - 1):
                new_score = update_score(best_score, best_path[i - 1], best_path[i], best_path[k - 1], best_path[k])
                if new_score < best_score - 0.001:
                    best_score = new_score
                    best_path = swap_2opt(best_path, i, k)
                    improvement = True
                    print(new_score)
                    if plots:
                        plot_path(best_path, best_score)
                    # write_submission(best_path, best_score)
                    break
            if improvement:
                break
    assert len(best_path) == len(path)
    return best_path


def run_2opt_chunks(score_func, ids2x, ids2y, time_limit, path):
    '''
    obselete
    '''
    now = datetime.datetime.now()
    adjacency_dict = calc_adjacency_dict(path, [ids2x[x] for x in path], [ids2y[y] for y in path])
    improvement = True
    best_path = path
    best_score = score_func(path, ids2x, ids2y)
    while improvement and datetime.datetime.now() < (now + datetime.timedelta(seconds=time_limit)):
        improvement = False
        for i in range(1, len(best_path) - 1):
            for k in range(i + 1, len(best_path) - 1):
                # new_score = update_score(best_score, best_path[i-1], best_path[i], best_path[k-1], best_path[k], ids2x, ids2y)
                new_score = update_score_from_dict(best_score, adjacency_dict, best_path[i - 1], best_path[i],
                                                   best_path[k - 1], best_path[k])
                if new_score < best_score - 0.001:
                    best_score = new_score
                    best_path = swap_2opt(best_path, i, k)
                    improvement = True
                    break
            if improvement:
                break
    return best_path


def swap_2opt_fast(path, i, k):
    path[i:k] = path[k - 1:i - 1:-1]
    return path


def swap_2opt(route, i, k):
    """
    swaps the endpoints of two edges by reversing a section of nodes,
        ideally to eliminate crossovers
    returns the new route created with a the 2-opt swap
    route - route to apply 2-opt
    i - start index of the portion of the route to be reversed
    k - index of last node in portion of route to be reversed
    pre: 0 <= i < (len(route) - 1) and i < k < len(route)
    post: length of the new route must match length of the given route
    """
    assert i >= 0 and i < (len(route) - 1)
    assert k > i and k < len(route)
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k]))
    new_route.extend(route[k:])
    assert len(new_route) == len(route)
    return new_route


def create_neighbours_dict(ids, coords, num):
    neighbour_dict = {}
    kdtree = spatial.KDTree(coords)
    for i in ids:
        neighbour_dict[i] = {}
        closest_nodes = kdtree.query((ids2x[i], ids2y[i]), k=num)
        for dist, node in zip(closest_nodes[0][1:], closest_nodes[1][1:]):
            neighbour_dict[i][node] = dist
    return neighbour_dict


def run_2opt_n_dict(path, score_function, n_dict, ids2x, ids2y):
    start = datetime.datetime.now()
    improvement = True
    best_path = path
    local_best_score = 0
    id2index = dict(zip(best_path, range(len(best_path))))
    i_indices = range(1, len(best_path) - 1)
    roll_i = 0
    adj_dict = {}
    while improvement:
        improvement = False
        for i in list(np.roll(i_indices, -roll_i)):
            for node, dist in n_dict[best_path[i]].items():
                k = id2index[node]
                n1, n2, n3, n4 = best_path[i - 1], best_path[i], best_path[k - 1], best_path[k]
                cache_dist(n1, n2, n3, n4, adj_dict, ids2x, ids2y)
                change = score_function(n1, n2, n3, n4, adj_dict)
                if change < local_best_score - 0.0001:
                    local_best_score = change
                    best_i = i
                    best_k = k
                    improvement = True
            if improvement:
                best_path = swap_2opt_fast(best_path, min(best_i, best_k), max(best_i, best_k))
                id2index = dict(zip(best_path, range(len(best_path))))
                local_best_score = 0
                roll_i = best_i
                break
    assert len(best_path) == len(path)
    return best_path, (datetime.datetime.now() - start)


def run_3opt_n_dict(path, score_function, n_dict, ids2x, ids2y):
    start = datetime.datetime.now()
    improvement = True
    best_path = path
    local_best_score = 0
    id2index = dict(zip(best_path, range(len(best_path))))
    i_indices = range(1, len(best_path) - 1)
    roll_i = 0
    adj_dict = {}
    while improvement:
        improvement = False
        for i in list(np.roll(i_indices, -roll_i)):
            for j_node, _ in n_dict[best_path[i]].items():
                j = id2index[j_node]
                for k_node, _ in n_dict[best_path[j]].items():
                    k = id2index[k_node]
                    i, j, k = sorted([i, j, k])
                    if j - i == 1 or k - j == 1 or len(set((i, j, k))) != 3: continue
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

    assert len(best_path) == len(path)
    return best_path, (datetime.datetime.now() - start)


if __name__ == '__main__':
    plots = False
    annotate = False
    sample_size = int(197769 / 197769 * 499)  # 197769 total

    cities = np.genfromtxt('Inputs/cities.csv', delimiter=',', skip_header=True)
    sub_cities = cities[:sample_size, :]
    coords = sub_cities[:, [1, 2]]
    x = sub_cities[:, 1]
    y = sub_cities[:, 2]
    ids = list(map(int, sub_cities[:, 0]))
    ids2x = dict(zip(ids, x))
    ids2y = dict(zip(ids, y))
    init_path = get_best_submission(sample_size)
    init_path = False
    if not init_path:
        print("Creating initial path from nearest neighbours")
        init_path = initial_path_from_nearest(ids, coords)
    init_score = get_score_prime(init_path, ids2x, ids2y)
    if plots:
        plot_path(sample_size, init_path, init_score, annotate)
    print(f"Creating dictionary of close neighbours")
    neighbours_dict = create_neighbours_dict(ids, coords, 100)
    print(f"Initial score {init_score}")
    print("Running 3-opt")
    best_path, rt = run_3opt_n_dict(init_path, update_score_from_dict, neighbours_dict, ids2x, ids2y)
    best_score = get_score_prime(best_path, ids2x, ids2y)
    write_submission(best_path, best_score)
    print(f"3-opt complete! Score:{best_score}, Time:{rt}")
    if plots:
        plot_path(sample_size, best_path, best_score, annotate)

'''
    print(f"Initial score {init_score}")
    pool = Pool()
    score_func = get_score
    path = split(init_path, threads)
    partial_run_2opt_chunks = partial(run_2opt_chunks, score_func, ids2x, ids2y, time_limit)
    best_path = list(itertools.chain(*pool.map(partial_run_2opt_chunks, path)))
    best_score = get_score_prime(best_path)
    write_submission(best_path, best_score)
    print(f"Best score {best_score}")
'''
