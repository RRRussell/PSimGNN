import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_adj_mat(path):
    csvfile = open(path, 'r')
    reader = csv.reader(csvfile, delimiter=' ')
    num_nodes = int(next(reader)[0])
    data = np.zeros((num_nodes, num_nodes))
    for i, row in enumerate(reader):
        for node in row[1:-1]:
            data[i, int(node)] = 1
    return data


def get_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def show_graph(g):
    nx.draw(g, node_size=5)
    plt.show()


'''
def main():
    adj_mat = get_adj_mat(sys.argv[1])
    g = get_graph(adj_mat)
    nx.write_gexf(g, sys.argv[2])
    #show_graph(g)

if __name__ == '__main__':
    main()
'''
