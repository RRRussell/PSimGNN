import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ot
import scipy as sp

def gromov_wasserstein(g1, g2):
    g1 = g1.get_nxgraph()
    g2 = g2.get_nxgraph()
    g1nodes = g1.node
    g2nodes = g2.node
    print(g1nodes)
    print(g2nodes)
    pos1 = nx.kamada_kawai_layout(g1)
    pos2 = nx.kamada_kawai_layout(g2)
    plt.figure()
    plt.subplot(1, 2, 1)
    nx.draw_networkx(g1, with_labels=True, pos=pos1)
    plt.subplot(1, 2, 2)
    nx.draw_networkx(g2, with_labels=True, pos=pos2)
    plt.show()


    n1 = g1.number_of_nodes()
    n2 = g2.number_of_nodes()
    degree1 = np.array([x[1] for x in g1.degree])
    degree1 = degree1 / np.sum(degree1)
    #degree1 = np.tile(degree1, (5, 1))

    degree2 = np.array([x[1] for x in g2.degree])
    degree2 = degree2 / np.sum(degree2)
    #degree2 = np.tile(degree2, (5, 1))


    #vfunc = np.vectorize(ot.emd)
    #M = np.dot(degree1.T, degree2)
    #w = vfunc((degree1, degree2), M)
    #print(M.shape)

    # C1 = sp.spatial.distance.cdist(degree1.reshape((len(degree1), 1)), degree1.reshape((len(degree1), 1)))  # default = euclidean
    # C2 = sp.spatial.distance.cdist(degree2.reshape((len(degree2), 1)), degree2.reshape((len(degree2), 1)))  # default = euclidean
    C1 = ot.dist(np.arange(len(degree1)).reshape((len(degree1), 1)), np.arange(len(degree1)).reshape((len(degree1), 1)))
    C2 = ot.dist(np.arange(len(degree2)).reshape((len(degree2), 1)), np.arange(len(degree2)).reshape((len(degree2), 1)))
    C1 = np.ones((len(degree1),len(degree1)))
    C2 = np.ones((len(degree2),len(degree2)))
    print(C1)
    print(C2)
    adj1 = nx.adjacency_matrix(g1).todense()
    adj2 = nx.adjacency_matrix(g2).todense()
    adj1 = np.asarray(adj1)
    adj2 = np.asarray(adj2)
    #print(type(adj1))
    for i in range(len(degree1)):
        for j in range(len(degree1)):
            if adj1[i][j] == 1:
                C1[i][j] = 0.5
            # else:
            #     C1[i][j] = 0.00001
    for i in range(len(degree2)):
        for j in range(len(degree2)):
            if adj2[i][j] == 1:
                C2[i][j] = 0.5
            # else:
            #     C2[i][j] = 0.00001


    gw, log = ot.gromov.gromov_wasserstein(C1, C2, degree1, degree2, 'square_loss', verbose=True, log=True)
    ngw, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, degree1, degree2, 'square_loss', max_iter=100, epsilon=0.001, verbose=True, log=True)

    #test = np.vstack([g.get_nxgraph().degree for g in g1.degree)]
    #degree1 = np.vstack([x[1] for x in g.get_nxgraph().degrees for g in g1])
    #print(test)

    # degree1 /= np.sum(degree1)
    # degree2 = [x[1] for x in g2.degree]
    # degree2 /= np.sum(degree2)

    # M_1d /= M_1d.max()
    # print(M_1d.shape)
    #

    rtn = np.zeros((n1, n2))
    return gw, ngw


if __name__ == '__main__':
    from load_data import load_dataset

    dataset = 'aids700nef'
    align_metric = 'mcs'
    node_ordering = 'bfs'

    dataset = load_dataset(dataset, 'all', align_metric, node_ordering)
    dataset.print_stats()
    gs = dataset.gs
    g1 = gs[0]
    g2 = gs[0]
    gw, ngw = gromov_wasserstein(g1, g2)
    g1nodes = list(g1.get_nxgraph().nodes)
    g2nodes = list(g2.get_nxgraph().nodes)
    print(g1nodes)
    print(g2nodes)
    idx_map = np.argmax(gw, axis=1)
    node_mapping = [g2nodes[i] for i in idx_map]
    print(node_mapping)

    # for n1, idx in zip(g1nodes, idx_map):
    #     print(n1, g2nodes[idx])
    print("------------------")
    print(g1nodes)
    print(g2nodes)
    idx_map = np.argmax(ngw, axis=1)
    node_mapping = [g2nodes[i] for i in idx_map]
    print(node_mapping)

    print(gw)
    print(ngw)