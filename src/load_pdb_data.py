from dataset import OurDataset
from graph import HierarchicalGraph
from graph_pair import GraphPair
from graph_label import add_glabel_to_each_graph
import copy
import re
from utils import sorted_nicely, get_data_path, assert_valid_nid, load
from os.path import join, basename
from glob import glob
import networkx as nx


def load_pdb_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel):
    name = _check_list_element_allowed([name], ['pdb'])
    natts = _check_list_element_allowed(
        natts, ['aa_type', 'aa_x', 'aa_y', 'aa_z', 'ss_type', 'ss_range',
                'c_range'])
    eatts = _check_list_element_allowed(eatts, [])
    tvt = _check_list_element_allowed([tvt], ['all'])
    align_metric = _check_list_element_allowed([align_metric], ['interaction'])
    glabel = _check_list_element_allowed([glabel], ['discrete'])
    graphs, graph_pairs = _iterate_get_graphs_pairs()
    return OurDataset(name, graphs, natts, eatts, graph_pairs, tvt, align_metric,
                      node_ordering, glabel, None)


def _iterate_get_graphs_pairs():
    # graphs: [HierarchicalGraph(), HierarchicalGraph(), ...]
    # graph_pairs: {(('1g14', 'xxxx'): GraphPair())}
    return None, None # TODO: iterate the folder join(get_data_path(), 'PPI_Datasets', 'PDB') using glob


def _check_list_element_allowed(li1, li2):
    assert type(li1) is list and type(li2) is list
    for ele1 in li1:
        assert ele1 in li2
    return li2
