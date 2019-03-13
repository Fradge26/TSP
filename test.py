import numpy as np
import pickle
import datetime


def get_score(route, cost_mat):
    score = 0
    for i in range(len(route) - 1):
        score += cost_mat[route[i]][route[i + 1]]
    return score


def cost_k_edges(cost_mat, route, edges):
    cost = 0
    for e in edges:
        cost += cost_mat[route[e[0]]][route[e[1]]]
    return cost


def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]


def three_opt(route, cost_mat):
    start = datetime.datetime.now()
    best_route = route
    improved = True
    while improved:
        for i in range(1, len(route) - 6):
            for j in range(i + 2, len(route) - 4):
                for k in range(j + 2, len(route) - 2):
                    improved = False
                    n1, n2, n3, n4, n5, n6 = i - 1, i, j - 1, j, k - 1, k
                    best_swap = [(n1, n2), (n3, n4), (n5, n6)]
                    current_cost = cost_k_edges(cost_mat, best_route, best_swap)
                    for sol in [[(n1, n2), (n3, n5), (n4, n6)],
                                [(n1, n3), (n2, n4), (n5, n6)],
                                [(n1, n3), (n2, n5), (n4, n6)],
                                [(n1, n4), (n5, n2), (n3, n6)],
                                [(n1, n4), (n5, n3), (n2, n6)],
                                [(n1, n5), (n4, n2), (n3, n6)],
                                [(n1, n5), (n4, n3), (n2, n6)]]:
                        sol_cost = cost_k_edges(cost_mat, best_route, sol)
                        if sol_cost < current_cost:
                            best_swap = sol
                            current_cost = sol_cost
                            improved = True
                    if improved:
                        if best_swap[1][0] > best_swap[0][1]:
                            s2 = 1
                        else:
                            s2 = -1
                        if best_swap[2][0] > best_swap[1][1]:
                            s3 = 1
                        else:
                            s3 = -1
                        best_route = best_route[:best_swap[0][0] + 1] + \
                                     best_route[best_swap[0][1]:best_swap[1][0] + s2:s2] + \
                                     best_route[best_swap[1][1]:best_swap[2][0] + s3:s3] + \
                                     best_route[best_swap[2][1]:]
                        break
            route = best_route
    return best_route, datetime.datetime.now() - start


def three_opt_roll(route, cost_mat):
    start = datetime.datetime.now()
    best_route = route
    improved = True
    i_indices = range(1, len(route) - 6)
    roll_i = 0
    while improved:
        for i in list(np.roll(i_indices, -roll_i-1)):
            for j in range(i + 2, len(route) - 4):
                for k in range(j + 2, len(route) - 2):
                    improved = False
                    n1, n2, n3, n4, n5, n6 = i - 1, i, j - 1, j, k - 1, k
                    best_swap = [(n1, n2), (n3, n4), (n5, n6)]
                    current_cost = cost_k_edges(cost_mat, best_route, best_swap)
                    for sol in [[(n1, n2), (n3, n5), (n4, n6)],
                                [(n1, n3), (n2, n4), (n5, n6)],
                                [(n1, n3), (n2, n5), (n4, n6)],
                                [(n1, n4), (n5, n2), (n3, n6)],
                                [(n1, n4), (n5, n3), (n2, n6)],
                                [(n1, n5), (n4, n2), (n3, n6)],
                                [(n1, n5), (n4, n3), (n2, n6)]]:
                        sol_cost = cost_k_edges(cost_mat, best_route, sol)
                        if sol_cost < current_cost:
                            best_swap = sol
                            current_cost = sol_cost
                            improved = True
                    if improved:
                        if best_swap[1][0] > best_swap[0][1]:
                            s2 = 1
                        else:
                            s2 = -1
                        if best_swap[2][0] > best_swap[1][1]:
                            s3 = 1
                        else:
                            s3 = -1
                        best_route = best_route[:best_swap[0][0] + 1] + \
                                     best_route[best_swap[0][1]:best_swap[1][0] + s2:s2] + \
                                     best_route[best_swap[1][1]:best_swap[2][0] + s3:s3] + \
                                     best_route[best_swap[2][1]:]
                        roll_i = i
                        break
                else:
                    continue
                break
            route = best_route
    return best_route, datetime.datetime.now() - start


def two_opt(route, cost_mat):
    start = datetime.datetime.now()
    best = route
    improved = True
    local_best_score = 0
    i_indices = range(1, len(route) - 2)
    roll_i = 0
    while improved:
        improved = False
        for i in list(np.roll(i_indices, -roll_i)):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < local_best_score:
                    best_i, best_j = i, j
                    # best[best_i:best_j] = best[best_j - 1:best_i - 1:-1]
                    improved = True
            if improved:
                best[best_i:best_j] = best[best_j - 1:best_i - 1:-1]
                local_best_score = 0
                roll_i = best_i
                break
        route = best
    return best, datetime.datetime.now() - start


if __name__ == '__main__':
    nodes = 100
    init_route = list(range(nodes))

    for _ in range(10):
        cost_mat = np.random.randint(100, size=(nodes, nodes))
        cost_mat += cost_mat.T
        np.fill_diagonal(cost_mat, 0)
        cost_mat = list(cost_mat)
        pickle.dump(cost_mat, open("./Pickle/cost_mat.p", "wb"))

        cost_mat = pickle.load(open("./Pickle/cost_mat.p", "rb"))
        print(f"inital route score {get_score(init_route, cost_mat)}")

        print("Running 2-opt")
        best_route, run_time = two_opt(init_route, cost_mat)
        print(f"2-opt route score {get_score(best_route, cost_mat)}, time:{run_time}")

        print("Running 3-opt")
        best_route, run_time = three_opt(init_route, cost_mat)
        print(f"3-opt route score {get_score(best_route, cost_mat)}, time:{run_time}")

        print("Running 3-opt roll")
        best_route, run_time = three_opt_roll(init_route, cost_mat)
        print(f"3-opt route score {get_score(best_route, cost_mat)}, time:{run_time}")
