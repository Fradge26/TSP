import dwave_networkx as dnx
import networkx as nx
import numpy as np
from scipy import spatial
import dimod

cities = np.genfromtxt('cities_mini.csv', delimiter=',', skip_header=True)
coords = cities[:, [1, 2]]
G = nx.Graph([(row[1], row[2]) for row in cities])
kdtree = spatial.KDTree(coords)

for idx, c in enumerate(coords):
    neighbours = kdtree.query(c, k=10)
    for neighbour in neighbours:
        G.add_edge(idx, neighbour[1], weight=neighbour[0])

G2 = nx.complete_graph(G.nodes)
sp = dnx.traveling_salesman(G2, dimod.ExactSolver())
print(sp)

