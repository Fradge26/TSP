import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_path(path, score):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [ids2x[node] for node in path]
    y = [ids2y[node] for node in path]
    line = Line2D(x, y)
    ax.add_line(line)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    for i, label in enumerate(path):
        ax.annotate(label, (ids2x[label], ids2y[label]), size=10)
    plt.show()
    fig.savefig(f'Figures/{score}.png', dpi=fig.dpi)


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


def get_score(path):
    score = 0
    for i, a in enumerate(path[1:]):
        x1, y1 = ids2coords[path[i + 1]]
        x2, y2 = ids2coords[path[i]]
        score += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return score


def get_score_prime(path):
    score = 0
    for i, a in enumerate(path[1:]):
        if ((i + 1) % 10 == 0) and not is_prime(a):
            penalty = 1.1
        else:
            penalty = 1.0
        x1, y1 = ids2coords[path[i + 1]]
        x2, y2 = ids2coords[path[i]]
        score += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * penalty
    return score


def initial_path_from_nearest(ids, coords):
    kdtree = spatial.KDTree(coords)
    path = [0]
    unvisited_nodes = set(ids)
    unvisited_nodes.remove(0)
    while unvisited_nodes:
        closest_nodes = kdtree.query(ids2coords[path[-1]], k=100)
        for cn in closest_nodes[1]:
            if cn in unvisited_nodes:
                path.append(cn)
                unvisited_nodes.remove(cn)
                break
        else:
            assert False
        if len(path) % 100 == 0:
            print(len(path))
    path.append(0)
    return path


def run_2opt(route):
    """
    improves an existing route using the 2-opt swap until no improved route is found
    best path found will differ depending of the start node of the list of nodes
        representing the input tour
    returns the best path found
    route - route to improve
    """
    improvement = True
    best_route = route
    best_distance = get_score(route)
    while improvement:
        improvement = False
        for i in range(1, len(best_route) - 1):
            for k in range(i + 1, len(best_route) - 1):
                new_route = swap_2opt(best_route, i, k)
                new_distance = get_score(new_route)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_route = new_route
                    improvement = True
                    print(new_distance)
                    plot_path(best_route, best_distance)
                    break
            if improvement:
                break
    assert len(best_route) == len(route)
    return best_route


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
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k + 1:])
    assert len(new_route) == len(route)
    return new_route


if __name__ == '__main__':
    cities = np.genfromtxt('Inputs/cities.csv', delimiter=',', skip_header=True)
    sub_cities = cities[:100, :]
    coords = sub_cities[:, [1, 2]]
    x = sub_cities[:, 1]
    y = sub_cities[:, 2]
    ids = list(map(int, sub_cities[:, 0]))
    ids2coords = dict(zip(ids, coords))
    ids2x = dict(zip(ids, x))
    ids2y = dict(zip(ids, y))
    init_path = initial_path_from_nearest(ids, coords)
    init_score = get_score_prime(init_path)
    plot_path(init_path, init_score)
    print(f"Initial score {get_score_prime(init_path)}")
    best_path = run_2opt(init_path)
    print(f"Best score {get_score_prime(best_path)}")
