import numpy as np
from utils import get_sparse_mat
import networkx as nx
from os.path import join, exists
from os import makedirs

def get_degree_dist(nx_graph):
    degrees = np.asarray([nx_graph.degree(n) for n in nx_graph.nodes()])
    return degrees / degrees.sum()


def create_hyperlevel_nxgraphs(d_name, gids, edge_types_edge_list, data_dir=None, write_graphs=False):
    assert((data_dir is not None and write_graphs) or (data_dir is None and not write_graphs))
    gid_to_adj_idx = {gid: i for i, gid in enumerate(sorted(gids))}
    adj_idx_to_gid = {adj_ind: gid for gid, adj_ind in gid_to_adj_idx.items()}

    adj_mats = {edge_type: get_sparse_mat(edge_list, gid_to_adj_idx, gid_to_adj_idx)
                for edge_type, edge_list in edge_types_edge_list.items()}
    hyperlevel_nxgraphs = {}

    for edge_type, adj_mat in adj_mats.items():
        nx_graph = nx.from_scipy_sparse_matrix(adj_mat)
        nx.set_node_attributes(nx_graph, adj_idx_to_gid, 'gid')
        # assert (len(list(nx.isolates(nx_graph))) == 0)
        if 'snap' in d_name:
            hyperlevel_nxgraphs[edge_type] = max(nx.connected_component_subgraphs(nx_graph), key=len)
        elif 'decagon' in d_name:
            nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
            hyperlevel_nxgraphs[edge_type] = nx_graph
        elif 'small' in d_name:
            hyperlevel_nxgraphs[edge_type] = nx_graph
        else:
            raise NotImplementedError
    if write_graphs:
        if not exists(join(data_dir, "ddi_graphs")):
            makedirs(join(data_dir, "ddi_graphs"))
        for edge_type, nx_graph in hyperlevel_nxgraphs.items():
            nx.readwrite.write_gexf(nx_graph, join(data_dir, "ddi_graphs", edge_type + "_ddi.gexf"))
    return hyperlevel_nxgraphs