import K_opt
import Three_opt
import numpy as np
from random import sample
import datetime
import cProfile

if __name__ == '__main__':
    nodes = 150
    ids = list(range(nodes))
    xs = np.random.randint(0, 10000, nodes, int)
    ys = np.random.randint(0, 10000, nodes, int)
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))

    path = sample(ids[1:], len(ids) - 1)
    path.insert(0, ids[0])
    path.append(ids[0])
    print(K_opt.get_score(path, ids2x, ids2y))

    path2 = K_opt.initial_path_from_nearest(ids, xs, ys)
    print(K_opt.get_score(path2, ids2x, ids2y))

    # cProfile.run('Three_opt.run_3opt(ids, xs, ys, neighbour_limit=20, path=path)')
    #best_path = Three_opt.run_3opt(ids, xs, ys, neighbour_limit=20, path=path)
    #print(Three_opt.get_score(best_path, ids2x, ids2y))
    #Three_opt.plot_path(best_path, ids2x, ids2y, True)

    # cProfile.run('K_opt.run_kopt(ids, xs, ys, depth=3, neighbour_limit=20, path=path)')
    best_path = K_opt.run_kopt(ids, xs, ys, depth=2, neighbour_limit=20, path=path)
    print(K_opt.get_score(best_path, ids2x, ids2y))
    K_opt.plot_path(best_path, ids2x, ids2y, True)

    best_path = K_opt.run_kopt(ids, xs, ys, depth=3, neighbour_limit=20, path=path)
    print(K_opt.get_score(best_path, ids2x, ids2y))
    K_opt.plot_path(best_path, ids2x, ids2y, True)

    best_path = K_opt.run_kopt(ids, xs, ys, depth=4, neighbour_limit=20, path=path)
    print(K_opt.get_score(best_path, ids2x, ids2y))
    K_opt.plot_path(best_path, ids2x, ids2y, True)

'''
    start = datetime.datetime.now()
    best_path = K_opt.run_kopt(ids, xs, ys, depth=4, neighbour_limit=20, path=path)
    print(K_opt.get_score(best_path, ids2x, ids2y), datetime.datetime.now() - start)
    K_opt.plot_path(best_path, ids2x, ids2y, True)

'''



