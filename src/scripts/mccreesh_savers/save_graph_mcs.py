import sys
import time
import subprocess 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import get_model_path, get_data_path
from os.path import join

def clean_directories():
    pipe = subprocess.Popen(['rm', '-r', 'temp/'])
    pipe.wait()
    pipe = subprocess.Popen(['mkdir', 'temp'])
    pipe.wait()

def decompress_graph(path):
    pipe = subprocess.Popen([join(get_model_path(), '..', 'src', 'scripts', 'mccreesh_savers', 'save_graph_mcs.out'), path])
    pipe.wait()

def get_nodes(num_nodes):
    g = nx.Graph()
    g.add_nodes_from([0,int(num_nodes)-1])
    '''
    for i in range(int(num_nodes)):
        g.add_node(str(i))
    '''
    return g

def get_edges(g, adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    g.add_edges_from(edges)
    return g

def get_node_features(g, features):
    for i, feature in enumerate(features):
        g.nodes[i]['feature'] = int(feature)
    return g

def get_edge_features(g, features, adj_matrix):
    n1, n2 = np.where(adj_matrix == 1)
    for i in range(len(n1)):
        g.edges[(n1[i],n2[i])]['feature'] = int(features[n1[i],n2[i]])
    return g

def show_graph(g):
    nx.draw(g, node_size=500)
    plt.show()

'''
def main():
    clean_directories()
    decompress_graph(sys.argv[1])

    adj_mat = np.genfromtxt('temp/adj_matrix.txt', delimiter=',')
    node_features = np.genfromtxt('temp/node_features.txt', delimiter=',')
    edge_features = np.genfromtxt('temp/edge_features.txt', delimiter=',')

    g = get_graph(adj_mat)
    g = get_node_features(g, node_features)
    g = get_edge_features(g, edge_features, adj_mat)

    #show_graph(g)
    nx.write_gexf(g, argv[2])

if __name__ == '__main__':
    main()
'''
