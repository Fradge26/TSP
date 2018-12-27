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


def calc_adjacency_dict(ids, x, y):
    adj_dict = {}
    for ind_i, i in enumerate(ids):
        for ind_j, j in enumerate(ids):
            adj_dict[(i, j)] = ((x[ind_i] - x[ind_j])**2 + (y[ind_i] - y[ind_j])**2) ** 0.5
    return adj_dict


def get_best_submission(sample_size):
    all_files = [f for f in os.listdir("./Submissions/") if os.path.isfile(os.path.join("./Submissions/", f))]
    valid_files = [f.split("_") for f in all_files if int(f.split("_")[0]) == sample_size]
    if len(valid_files) == 0:
        return False
    else:
        best_file = os.path.join("./Submissions/", "_".join(sorted(valid_files)[0]))
        path = list(np.genfromtxt(best_file, delimiter=',', skip_header=True, dtype=int))
        return path


def write_submission(path, score):
    path2 = deepcopy(path)
    path2.insert(0, 'path')
    np.savetxt(f'Submissions/{sample_size}_{int(score)}.csv', path2, fmt='%s')


def plot_path(path, score):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [ids2x[node] for node in path]
    y = [ids2y[node] for node in path]
    line = Line2D(x, y)
    ax.add_line(line)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    # for i, label in enumerate(path):
    #    ax.annotate(label, (ids2x[label], ids2y[label]), size=10)
    plt.show()
    fig.savefig(f'Figures/{score}.png', dpi=1000)


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


def update_score(score, n1, n2, n3, n4, ids2x, ids2y):
    score -= ((ids2x[n1] - ids2x[n2]) ** 2 + (ids2y[n1] - ids2y[n2]) ** 2) ** 0.5
    score -= ((ids2x[n3] - ids2x[n4]) ** 2 + (ids2y[n3] - ids2y[n4]) ** 2) ** 0.5
    score += ((ids2x[n1] - ids2x[n3]) ** 2 + (ids2y[n1] - ids2y[n3]) ** 2) ** 0.5
    score += ((ids2x[n2] - ids2x[n4]) ** 2 + (ids2y[n2] - ids2y[n4]) ** 2) ** 0.5
    return score


def update_score_from_dict(score, adj_dict, n1, n2, n3, n4):
    return score - adj_dict[(n1, n2)] - adj_dict[(n3, n4)] + adj_dict[(n1, n3)] + adj_dict[(n2, n4)]


def get_score_prime(path):
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
        if len(path) % 1000 == 0:
            print(len(path))
    path.append(0)
    return path


def find_crosses(path, score_func):
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
                    new_score = update_score(best_score, best_path[i-1], best_path[i], best_path[k-1], best_path[k])
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
                new_score = update_score(best_score, best_path[i-1], best_path[i], best_path[k-1], best_path[k])
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
                new_score = update_score_from_dict(best_score, adjacency_dict, best_path[i-1], best_path[i], best_path[k-1], best_path[k])
                if new_score < best_score - 0.001:
                    best_score = new_score
                    best_path = swap_2opt(best_path, i, k)
                    improvement = True
                    break
            if improvement:
                break
    return best_path


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


if __name__ == '__main__':
    plots = False
    sample_size = int(197769/200)
    threads = 1
    time_limit = 60

    cities = np.genfromtxt('Inputs/cities.csv', delimiter=',', skip_header=True)
    sub_cities = cities[:sample_size, :]
    coords = sub_cities[:, [1, 2]]
    x = sub_cities[:, 1]
    y = sub_cities[:, 2]
    ids = list(map(int, sub_cities[:, 0]))
    # adj_dict = calc_adjacency_dict(ids, x, y)
    ids2x = dict(zip(ids, x))
    ids2y = dict(zip(ids, y))
    init_path = get_best_submission(sample_size)
    if not init_path:
        init_path = initial_path_from_nearest(ids, coords)
    init_score = get_score_prime(init_path)
    write_submission(init_path, init_score)
    if plots:
        plot_path(init_path, init_score)
    print(f"Initial score {init_score}")
    pool = Pool()
    score_func = get_score
    path = split(init_path, threads)
    partial_run_2opt_chunks = partial(run_2opt_chunks, score_func, ids2x, ids2y, time_limit)
    best_path = list(itertools.chain(*pool.map(partial_run_2opt_chunks, path)))
    best_score = get_score_prime(best_path)
    write_submission(best_path, best_score)
    print(f"Best score {best_score}")
