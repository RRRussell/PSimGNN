from config import FLAGS
from merged_graph import MergedGraphData
from node_feat import encode_node_features
from pair_processor import preproc_graph_pair, \
    postproc_graph_pairs_assign_node_embeds
from torch_geometric.data import Data as PyGSingleGraphData
from torch_geometric.utils import to_undirected
import torch
import networkx as nx


class BatchData(object):
    """Mini-batch.

    We assume the following sequential model architecture: Merge --> Split.

        Merge: For efficiency, first merge graphs in a batch into a large graph.
            This is only done for the first several `NodeEmbedding` layers.

        Split: For flexibility, split the merged graph into individual pairs.
            The `gen_list_view_by_split` function should be called immediately
            after the last `NodeEmbedding` layer.
    """

    def __init__(self, batch_gids, dataset, input_gids = None):
        self.dataset = dataset
        # print("??????",batch_gids)
        self.merge_data, self.pair_list = self._merge_into_one_graph(
            batch_gids, input_gids)
        # print("important",self.merge_data)
        # self.merge = mgd['merge']
        # self.ind_list = mgd['ind_list']  # for split later

    def _merge_into_one_graph(self, batch_gids, input_gids = None):
        single_graph_list = []
        metadata_list = []
        pair_list = []
        # assert len(batch_gids) == 2
        # gids1 = batch_gids[0]
        # gids2 = batch_gids[1]
        # print("batch",batch_gids)
        gids1 = batch_gids[:, 0]
        gids2 = batch_gids[:, 1]
        assert gids1.shape == gids2.shape
        
        if input_gids:
            for (gid1, gid2) in input_gids:
                # print(gid1,gid2)
                self._preproc_gid_pair(gid1, gid2, single_graph_list, metadata_list, pair_list)
        else:
            for (gid1, gid2) in zip(gids1, gids2):
                # print("?",gid1,gid2)
                self._preproc_gid_pair(gid1, gid2, single_graph_list, metadata_list, pair_list)
        # assert len(pair_list) == gids1.shape[0] == gids2.shape[0]
        # print("single",single_graph_list)
        # print("meta",metadata_list)
        return MergedGraphData.from_data_list(single_graph_list, metadata_list), pair_list

    def _preproc_gid_pair(self, gid1, gid2, single_graph_list, metadata_list, pair_list):
        if type(gid1) != int:
            gid1 = gid1.item()
            gid2 = gid2.item()
        # print(type(gid1))
        # print("in preproc",gid1,gid2)
        assert gid1 - int(gid1) == 0
        assert gid2 - int(gid2) == 0
        gid1 = int(gid1)
        gid2 = int(gid2)
        g1 = self.dataset.look_up_graph_by_gid(gid1)
        g2 = self.dataset.look_up_graph_by_gid(gid2)
        pair = self.dataset.look_up_pair_by_gids(g1.gid(), g2.gid())
        # print("??????????????",pair)
        if pair==None:
            return
        # print(g1,g2,pair)
        preproc_g_list = preproc_graph_pair(g1, g2, pair)  # possibly combine
        # print(preproc_g_list)
        this_single_graph_list = [self._convert_nx_to_pyg_graph(g.get_nxgraph())
                                  for g in preproc_g_list]
        # this_metadata_list = [(g.nxgraph.graph['dists_max'], g.nxgraph.graph['dists_argmax'])
        #                       for g in preproc_g_list]
        single_graph_list.extend(this_single_graph_list)
        # metadata_list.extend(this_metadata_list)
        pair.assign_g1_g2(g1, g2)
        pair_list.append(pair)

    def _convert_nx_to_pyg_graph(self, g):  # g is a networkx graph object
        """converts_a networkx graph to a PyGSingleGraphData."""
        # Reference: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/ppi.py
        if type(g) is not nx.Graph:
            raise ValueError('Input graphs must be undirected nx.Graph,'
                             ' NOT {}'.format(type(g)))
        edge_index = create_edge_index(g)
        # print("in convert")
        # print(edge_index)
        # print(g.nodes())
        # print(g.init_x)
        # print("con",g.nodes())
        # print("g",type(g.init_x))
        data = PyGSingleGraphData(
            # x=torch.tensor(g.init_x,
            #                dtype=torch.float32,
            #                # required by concat with LocalDegreeProfile()
            #                device=FLAGS.device),
            x=(g.init_x).clone().detach().requires_grad_(True),
            edge_index=edge_index,
            edge_attr=None,
            y=None)  # TODO: add one-hot
        # print('before', data)

        data, nf_dim = encode_node_features(pyg_single_g=data)
        # print("fea",data)
        assert data.is_undirected()
        assert data.x.shape[1] == nf_dim
        # print('after', data.x.shape)
        return data

    def split_into_pair_list(self, node_embed_merge, node_embed_name):
        node_embed_list = MergedGraphData.to_data_list(
            self.merge_data, node_embed_merge)
        assert len(node_embed_list) == self.merge_data['merge'].num_graphs
        # print("node_embed_merge",node_embed_merge.shape)
        # print("node_embed_name",node_embed_name)
        postproc_graph_pairs_assign_node_embeds(
            node_embed_list, node_embed_name, self.pair_list)
        return self.pair_list


def create_edge_index(g):
    # print("-------------")
    # print(g.edges)
    # print("number of nodes %d"%(len(g)))
    # print(g.number_of_nodes())
    # print(len(g.edges))
    e = []
    for i in g.edges:
        e.append(tuple(map(int,i)))
    # print(e)
    edge_index = torch.tensor(e,
                              device=FLAGS.device).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=g.number_of_nodes())
    return edge_index