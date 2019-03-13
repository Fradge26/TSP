import TSP_opt
import numpy as np
from random import sample
import datetime
import TSP
import cProfile

if __name__ == '__main__':
    sample_size = int(197769)  # 197769 total
    cities = np.genfromtxt('Inputs/cities.csv', delimiter=',', skip_header=True)
    sub_cities = cities[:sample_size, :]
    coords = sub_cities[:, [1, 2]]
    xs = sub_cities[:, 1]
    ys = sub_cities[:, 2]
    ids = list(map(int, sub_cities[:, 0]))
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))


    p = TSP.get_best_submission(197769)
    s = TSP.get_score_prime(p, ids2x, ids2y)
    #TSP_opt.save_path_plot(p, f'Figures/{sample_size}_{int(s)}.png', ids2x, ids2y, True)

    #TSP.write_submission(p,s,197769)

    bp = TSP_opt.run_2opt(ids, xs, ys, 100, p)
    s = TSP.get_score_prime(p, ids2x, ids2y)
    TSP.write_submission(bp, s, 197769)

    '''
    nodes = 1000
    ids = list(range(nodes))
    xs = np.random.randint(0, 10000, nodes, int)
    ys = np.random.randint(0, 10000, nodes, int)
    ids2x = dict(zip(ids, xs))
    ids2y = dict(zip(ids, ys))
 

    path = sample(ids[1:], len(ids) - 1)
    path.insert(0, ids[0])
    path.append(ids[0])
    print(TSP_opt.get_score(path, ids2x, ids2y))
    path2 = TSP_opt.initial_path_from_nearest(ids, xs, ys)
    print(TSP_opt.get_score(path2, ids2x, ids2y))

    #start = datetime.datetime.now()
    #best_path = TSP_opt.run_2opt(ids, xs, ys, neighbour_limit=20, path=path2)
    #print(TSP_opt.get_score(best_path, ids2x, ids2y), datetime.datetime.now() - start)

    start = datetime.datetime.now()
    best_path = TSP_opt.run_2opt2(ids, xs, ys, neighbour_limit=20, path=path2)
    best_score = TSP_opt.get_score(best_path, ids2x, ids2y)
    TSP.write_submission(best_path, best_score, sample_size)

    #best_path = cProfile.run('TSP_opt.run_2opt2(ids, xs, ys, neighbour_limit=20, path=path)')

  
    # cProfile.run('K_opt.run_kopt(ids, xs, ys, depth=3, neighbour_limit=20, path=path)')
    best_path = TSP_opt.run_kopt(ids, xs, ys, depth=2, neighbour_limit=20, path=path)
    print(TSP_opt.get_score(best_path, ids2x, ids2y))
    TSP_opt.plot_path(best_path, ids2x, ids2y, True)

    best_path = TSP_opt.run_kopt(ids, xs, ys, depth=3, neighbour_limit=20, path=path)
    print(TSP_opt.get_score(best_path, ids2x, ids2y))
    TSP_opt.plot_path(best_path, ids2x, ids2y, True)

    best_path = TSP_opt.run_kopt(ids, xs, ys, depth=4, neighbour_limit=20, path=path)
    print(TSP_opt.get_score(best_path, ids2x, ids2y))
    TSP_opt.plot_path(best_path, ids2x, ids2y, True)


    start = datetime.datetime.now()
    best_path = K_opt.run_kopt(ids, xs, ys, depth=4, neighbour_limit=20, path=path)
    print(K_opt.get_score(best_path, ids2x, ids2y), datetime.datetime.now() - start)
    K_opt.plot_path(best_path, ids2x, ids2y, True)

'''



