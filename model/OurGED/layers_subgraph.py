from config import FLAGS
from layers import create_act, NodeEmbedding, get_prev_layer
from utils_our import debug_tensor, pad_extra_rows
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Timer
from torch_scatter import scatter_add
from torch_geometric.nn import EdgeConv
from graph import RegularGraph
from dataset import OurDataset
from graph_pair import GraphPair
from batch import BatchData
import networkx as nx
import matplotlib.pyplot as plt

class ANPM_FAST(nn.Module):
    """ Attention_NTN_Padding_MNE layer. """

    def __init__(self, input_dim, feature_map_dim, reduce_factor, criterion):

        super(ANPM_FAST, self).__init__()

        self.emb_dim = input_dim
        self.W_0 = glorot([self.emb_dim, self.emb_dim])

        self.feature_map_dim = feature_map_dim
        self.reduce_factor = reduce_factor
        self.num_partitions = FLAGS.num_partitions
        self.num_select = FLAGS.num_select
        self.k2 = self.num_partitions*self.num_partitions

        self.linear_layer_graph_level = nn.Linear(self.k2, 8, bias=False)
        
        nn.init.xavier_normal_(self.linear_layer_graph_level.weight)

        proj_layers = []

        if FLAGS.which_branch=="up_down":
            self.linear_layer_sub_graph_level = nn.Linear(self.num_select, 8, bias=False)
            nn.init.xavier_normal_(self.linear_layer_sub_graph_level.weight)

            D = self.feature_map_dim
            self.propagator_num = 3
            propagator_layers = []
            for i in range(self.propagator_num):
                propagator_layer = GMNPropagator_FAST(input_dim=D,output_dim=D,distance_metric='cosine',more_nn='None')
                propagator_layers.append(propagator_layer)
            self.propagator_layers = nn.ModuleList(propagator_layers)
            self.aggregator_layers = GMNAggregator_FAST(input_dim=D,output_dim=D)
            self.gmn_loss = GMNLoss_FAST(ds_metric='cosine')

            D = 16

        else:
            D = 8

        while D > 1:
            next_D = D // reduce_factor
            if next_D < 1:
                next_D = 1
            linear_layer = nn.Linear(D, next_D, bias=True)
            nn.init.xavier_normal_(linear_layer.weight)
            proj_layers.append(linear_layer)
            if next_D != 1:
                proj_layers.append(nn.PReLU(1))
            D = next_D

        proj_layers.append(nn.Sigmoid())
        
        self.proj_layers = nn.ModuleList(proj_layers)

        if criterion == 'MSELoss':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError()

    def forward(self, ins, batch_data, model, pair_times):
        # ins is the embedding for single subgraph
        pair_list = batch_data.split_into_pair_list(ins, 'x')
       
        pairwise_scores = []

        sub_pair_list = self.get_sub_list(pair_list,pair_times)

        for i in range(pair_times):
            #coarse-grained comparision
            pairwise_embeddings = []
            pairwise_embeddings_sub = []
            sub_pairwise_ranks = []
            G1 = []
            G2 = []
            sub_batch_gids = []
            sub_pairs = {}
            sub_graphs = []
            used_sub_graphs = {}
            #-----------------------------------------------------
            # up
            # G=nx.grid_2d_graph(4,4)

            for j, pair in enumerate(sub_pair_list[i]):
                x1, x2 = pair.g1.x, pair.g2.x
                # print("x1_sub", pair.g1.gid())
                # print("x2_sub", pair.g2.gid())
                # x_1_gemb = self.att_layer(x1)
                # x_2_gemb = self.att_layer(x2)
                x_1_gemb = get_att(x1,self.W_0,self.emb_dim)
                x_2_gemb = get_att(x2,self.W_0,self.emb_dim)
                
                # plt.show()
                # x_1_gemb = sub_graph_att[i*self.k2+2*j,].reshape(1,-1)
                # x_2_gemb = sub_graph_att[i*self.k2+2*j+1,].reshape(1,-1)
                # print("x_1_gemb", x_1_gemb)
                # print("x_2_gemb", x_2_gemb)

                x = torch.cosine_similarity(x_1_gemb,x_2_gemb)
                # x = self.ntn_layer([x_1_gemb, x_2_gemb])
                # print("x after ntn",x)
                # for layer in self.graph_mlp_layers:
                #     x = layer(x)
                #     print("x after ntn and mlp",x)

                # DotProductSimilarity()
                # x = torch.dist(x_1_gemb, x_2_gemb, p=2)

                # x = torch.exp(-x)
                # print("x",x)
                pairwise_embeddings.append(x)

                sub_pairwise_ranks.append(x.clone().detach())

            embeddings = torch.cat(pairwise_embeddings)
            embeddings = embeddings.to(FLAGS.device)
            # print("up",embeddings)
            embeddings = self.linear_layer_graph_level(embeddings)
            # print("embeddings",embeddings)
            # print("ranks",sub_pairwise_ranks)
            # embeddings = torch.tensor(pairwise_embeddings)
            # embeddings = self.linear_layer_graph_level(embeddings)
            #-----------------------------------------------------------------------------------
            # below
            
            if FLAGS.which_branch == "up_down":
                l = np.squeeze(sub_pairwise_ranks)
                ranks = np.argsort(l,axis=0)

                if FLAGS.draw_sub_graph:
                    plt.subplots(2,3,figsize=(18,6)) 

                M = self.num_select
                for j in range(M):
                    pair = sub_pair_list[i][ranks[::-1][j]]
                    x1, x2 = pair.g1.x, pair.g2.x

                    if FLAGS.draw_sub_graph:
                        print("x1_sub", pair.g1.gid())
                        print(x1.shape)
                        plt.subplot(23*10+j+1)
                        nx.draw(pair.g1.get_nxgraph(),node_size=30)
                        # plt.show(plt.show())
                        print("x2_sub", pair.g2.gid())
                        print(x2.shape)
                        plt.subplot(23*10+j+4)
                        nx.draw(pair.g2.get_nxgraph(),node_size=30)
                        print("cos",x)

                    temp = torch.cat((x1,x2),0)
                    if j == 0:
                        sub_pairwise_score = temp
                    else:
                        sub_pairwise_score = torch.cat((sub_pairwise_score,temp),0)
                    sub_batch_gids.append([pair.g1.gid(),pair.g2.gid()])
                    sub_pairs[(pair.g1.gid(),pair.g2.gid())] = (GraphPair(g1=pair.g1,g2=pair.g2))
                    if pair.g1.gid() not in used_sub_graphs:
                        sub_graphs.append(pair.g1)
                        used_sub_graphs[pair.g1.gid()] = pair.g1.gid()
                    if pair.g2.gid() not in used_sub_graphs:
                        sub_graphs.append(pair.g2)
                        used_sub_graphs[pair.g2.gid()] = pair.g2.gid()
                if FLAGS.draw_sub_graph:
                    plt.show()

                # print(ranks)

                pairwise_score = (embeddings.reshape(1,-1))

                sub_batch_gids = torch.tensor(sub_batch_gids)
                sub_data = OurDataset(name="aids700nef",gs1=[],gs2=[],graphs=sub_graphs,natts=['type'],eatts=[],pairs=sub_pairs,tvt="train",align_metric="mcs",node_ordering="bfs",glabel=None,loaded_dict=None,mini="mini")
                sub_batch_data = BatchData(sub_batch_gids, sub_data)

                # print("================================")
                # print("begin",sub_pairwise_score.shape)
                for propagator_layer in self.propagator_layers:
                    sub_pairwise_score = propagator_layer(sub_pairwise_score,sub_batch_data,model)

                # print("after pro", sub_pairwise_score)
                sub_pairwise_score = self.aggregator_layers(sub_pairwise_score,sub_batch_data,model)
                # print("after agg", sub_pairwise_score)

                sub_pairwise_score = self.gmn_loss(sub_pairwise_score,sub_batch_data,model,pair_times)
                # print("after loss",sub_pairwise_score)
                # sub_pairwise_score = self.linear_layer_sub_graph_level(sub_pairwise_score)
                #-----------------------------------------------------------------------------------
                # concact

                # embeddings, _ = (torch.sort(embeddings, descending=True))
                
                # sub_pairwise_score, _ = (torch.sort(sub_pairwise_score, descending=True))
                # print("below",sub_pairwise_score)
                sub_pairwise_score = self.linear_layer_sub_graph_level(sub_pairwise_score)

                pairwise_score = torch.cat((embeddings,sub_pairwise_score),0).to(FLAGS.device)
            # elif FLAGS.which_branch == "down":
            #     pairwise_score = sub_pairwise_score
            elif FLAGS.which_branch == "up":
                pairwise_score = embeddings.reshape(1,-1)
            if i == 127:
                print("ppp",pairwise_score)

            # print("begin mlp", pairwise_score)
            for proj_layer in self.proj_layers:
                pairwise_score = proj_layer(pairwise_score)

                # print("w",proj_layer.weight)
                if i == 127:
                    print("in mlp",pairwise_score)

            pairwise_scores.append(pairwise_score)
            
        pairwise_scores = torch.cat(pairwise_scores,0)

        return pairwise_scores

    def get_sub_list(self, pair_list, pair_times):
        sub_pair_list = []
        one_pair = []
        for i, pair in enumerate(pair_list):
            gid1 = pair.g1.gid()//10
            gid2 = pair.g2.gid()//10
            if i == 0:
                last_gid1 = gid1
                last_gid2 = gid2
            if gid1 == last_gid1 and gid2 == last_gid2:
                one_pair.append(pair)
                if i == len(pair_list)-1:
                    sub_pair_list.append(one_pair)
                    #print("last one",len(one_pair))
            else:
                sub_pair_list.append(one_pair)
                #print(len(one_pair))
                one_pair=[]
                one_pair.append(pair)
            last_gid1 = gid1
            last_gid2 = gid2
        #print(len(sub_pair_list),'  ',pair_times)

        return sub_pair_list

def get_att(x,W,emb_dim):
    temp = torch.mean(x, 0).view((1, -1))  # (1, D)
    h_avg = torch.tanh(torch.mm(temp, W))
    att = interact_two_sets_of_vectors(x,h_avg,1,W=[torch.eye(emb_dim, device=FLAGS.device)],act=torch.sigmoid)
    output = torch.mm(att.view(1, -1), x)  # (1, D)
    return output

def interact_two_sets_of_vectors(x_1, x_2, interaction_dim, V=None,
                                 W=None, b=None, act=None, U=None):
    """
    Calculates the interaction scores between each row of x_1 (a marix)
        and x_2 ( a vector).
    :param x_1: an (N, D) matrix (if N == 1, a (1, D) vector)
    :param x_2: a (1, D) vector
    :param interaction_dim: number of interactions
    :param V:
    :param W:
    :param b:
    :param act:
    :param U:
    :return: if U is not None, interaction results of (N, interaction_dim)
             if U is None, interaction results of (N, 1)
    """
    feature_map = []
    for i in range(interaction_dim):
        middle = 0.0
        if V is not None:
            # In case x_2 is a vector but x_1 is a matrix, tile x_2.
            tiled_x_2 = torch.mul(torch.ones_like(x_1, device=FLAGS.device), x_2)
            concat = torch.cat((x_1, tiled_x_2), 1)
            v_weight = V[i].view(-1, 1)
            V_out = torch.mm(concat, v_weight)
            middle += V_out
        if W is not None:
            temp = torch.mm(x_1, W[i])
            h = torch.mm(temp, x_2.t())  # h = K.sum(temp*e2,axis=1)
            middle += h
        if b is not None:
            middle += b[i]
        feature_map.append(middle)
    # print("--------------")
    # print(feature_map)
    output = torch.cat(feature_map, 1)
    # print(output)
    # output = F.normalize(output, p=1, dim=1)  # TODO: check why need this
    # print(output)
    if act is not None:
        output = act(output)
    if U is not None:
        output = torch.mm(output, U)
        # print(output.shape)

    return output


class MNE(nn.Module):
    """ MNE layer. """

    def __init__(self, input_dim, inneract):
        super(MNE, self).__init__()

        self.inneract = create_act(inneract)
        self.input_dim = input_dim

    def forward(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            # Double list.
            rtn = []
            for input in inputs:
                assert (len(input) == 2)
                rtn.append(self._call_one_pair(input))
            return rtn
        else:
            assert (len(inputs) == 2)
            return self._call_one_pair(inputs)

    def _call_one_pair(self, input):
        # Assume x_1 & x_2 are of dimension N * D
        x_1 = input[0]
        x_2 = input[1]

        # print("in call",x_1.shape)
        # print("in call",x_2.shape)
        # one pair comparison
        t = Timer()
        output = torch.mm(x_1, x_2.t())
        # print("in call pm",t.time_msec_and_clear())
        return self.inneract(output)


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    rtn = nn.Parameter(torch.Tensor(*shape).to(FLAGS.device))
    nn.init.xavier_normal_(rtn)
    return rtn

class MLP_FAST(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        # self.activation = create_act(activation_type)
        self.activation = nn.PReLU(1)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1], bias=True)
                                   for i in range(len(self.layer_channels) - 1)])))

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x, batch_data, model):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            layer_inputs.append(self.activation(layer(input)))
        model.store_layer_output(self, layer_inputs[-1])
        return layer_inputs[-1]


class GMNPropagator_FAST(nn.Module):
    def __init__(self, input_dim, output_dim, more_nn, distance_metric='cosine', f_node='MLP'):
        super().__init__()
        self.out_dim = output_dim
        if distance_metric == 'cosine':
            self.distance_metric = nn.CosineSimilarity()
        elif distance_metric == 'euclidean':
            self.distance_metric = nn.PairwiseDistance()
        self.softmax = nn.Softmax(dim=1)
        self.f_messasge = MLP_FAST(2 * input_dim, 2 * input_dim, num_hidden_lyr=1, hidden_channels=[
            2 * input_dim])  # 2*input_dim because in_dim = dim(g1) + dim(g2)
        self.f_node_name = f_node
        if f_node == 'MLP':
            self.f_node = MLP_FAST(4 * input_dim, output_dim,
                              num_hidden_lyr=1)  # 2*input_dim for m_sum, 1 * input_dim for u_sum and 1*input_dim for x
        elif f_node == 'GRU':
            self.f_node = nn.GRUCell(3 * input_dim,
                                     input_dim)  # 2*input_dim for m_sum, 1 * input_dim for u_sum
        else:
            raise ValueError("{} for f_node has not been implemented".format(f_node))
        self.more_nn = more_nn
        if more_nn == 'None':
            pass
        elif more_nn == 'EdgeConv':
            nnl = nn.Sequential(nn.Linear(2 * input_dim, output_dim), nn.ReLU(),
                                nn.Linear(output_dim, output_dim))
            self.more_conv = EdgeConv(nnl, aggr='max')
            self.proj_back = nn.Sequential(nn.Linear(2 * output_dim, output_dim), nn.ReLU(),
                                nn.Linear(output_dim, output_dim))
        else:
            raise ValueError("{} has not been implemented".format(more_nn))

    def forward(self, ins, batch_data, model):
        x = ins  # x has shape N(gs) by D

        edge_index = batch_data.merge_data['merge'].edge_index  # edges of each graph
        row, col = edge_index
        # print("row",row.shape)
        # print("row",row)
        # print("col",col.shape)
        # print("col",col)
        m = torch.cat((x[row], x[col]), dim=1)  # E by (2 * D)
        m = self.f_messasge(m, batch_data, model)
        # print("m",m.shape)
        m_sum = scatter_add(m, row, dim=0, dim_size=x.size(0))  # N(gs) by (2 * D)
        # print("m sum",m_sum.shape)
        u_sum = self.f_match(x, batch_data)  # u_sum has shape N(gs) by D

        if self.f_node_name == 'MLP':
            in_f_node = torch.cat((x, m_sum, u_sum), dim=1)
            out = self.f_node(in_f_node, batch_data, model)
        elif self.f_node_name == 'GRU':
            in_f_node = torch.cat((m_sum, u_sum), dim=1)  # N by 3*D
            out = self.f_node(in_f_node, x)

        if self.more_nn != 'None':
            more_out = self.more_conv(x, edge_index)
            # Concat the GMN output with the additional output.
            out = torch.cat((out, more_out), dim=1)
            out = self.proj_back(out) # back to output_dim

        model.store_layer_output(self, out)
        return out

    def f_match(self, x, batch_data):
        '''from the paper https://openreview.net/pdf?id=S1xiOjC9F7'''
        ind_list = batch_data.merge_data['ind_list']
        # print("ind_list",ind_list)
        u_all_l = []

        for i in range(0, len(ind_list), 2):
            g1_ind = i
            g2_ind = i + 1
            g1x = x[ind_list[g1_ind][0]: ind_list[g1_ind][1]]
            g2x = x[ind_list[g2_ind][0]: ind_list[g2_ind][1]]

            u1 = self._f_match_helper(g1x, g2x)  # N(g1) by D tensor
            u2 = self._f_match_helper(g2x, g1x)  # N(g2) by D tensor

            u_all_l.append(u1)
            u_all_l.append(u2)

        return torch.cat(u_all_l, dim=0).view(x.size(0), -1)

    def _f_match_helper(self, g1x, g2x):

        g1_norm = torch.nn.functional.normalize(g1x, p=2, dim=1)
        g2_norm = torch.nn.functional.normalize(g2x, p=2, dim=1)
        # print("g1_norm",g1_norm.shape)
        # print("g2_norm",g2_norm.shape)
        g1_sim = torch.matmul(g1_norm, torch.t(g2_norm))
        # print("g1_sim",g1_sim.shape)
        # N_1 by N_2 tensor where a1[x][y] is the softmaxed a_ij of the yth node of g2 to the xth node of g1
        a1 = self.softmax(g1_sim)
        # print("a1",a1)
        sum_a1_h = torch.sum(g2x * a1[:, :, None],
                             dim=1)  # N1 by D tensor where each row is sum_j(a_j * h_j)
        # print("sum_a1_h",sum_a1_h.shape)
        return g1x - sum_a1_h


class GMNAggregator_FAST(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.out_dim = output_dim
        self.sigmoid = nn.Sigmoid()
        self.weight_func = MLP_FAST(input_dim, output_dim, num_hidden_lyr=1,
                               hidden_channels=[output_dim])
        self.gate_func = MLP_FAST(input_dim, output_dim, num_hidden_lyr=1, hidden_channels=[output_dim])
        self.mlp_graph = MLP_FAST(output_dim, output_dim, num_hidden_lyr=1, hidden_channels=[output_dim])

    def forward(self, x, batch_data, model):

        weighted_x = self.weight_func(x, batch_data, model)  # shape N by input_dim
        # print("weight",weighted_x.shape)
        gated_x = self.sigmoid(self.gate_func(x, batch_data, model))  # shape N by input_dim
        # print("gatedx",gated_x.shape) 
        hammard_prod = gated_x * weighted_x
        merge_data = batch_data.merge_data['merge']
        batch = merge_data.batch
        num_graphs = merge_data.num_graphs
        # print("batch",batch)
        # print("ham",hammard_prod.shape)
        graph_embeddings = scatter_add(hammard_prod, batch, dim=0,
                                       dim_size=num_graphs)  # shape G by output_dim
        return self.mlp_graph(graph_embeddings, batch_data, model)


class GMNLoss_FAST(nn.Module):
    def __init__(self, ds_metric='cosine'):
        super().__init__()
        if ds_metric == 'cosine':
            self.ds_metric = nn.CosineSimilarity()
            if FLAGS.dos_pred != 'sim':
                raise ValueError('cosine must use dos_pred == sim')
        elif ds_metric == 'euclidean':
            self.ds_metric = nn.PairwiseDistance()
            if FLAGS.dos_pred != 'euclidean':
                raise ValueError('euclidean must use dos_pred == dist')
        elif ds_metric == 'scalar':
            self.ds_metric = None

        self.linear_layer = nn.Linear(FLAGS.num_partitions*FLAGS.num_partitions, 2, bias=True)
        nn.init.xavier_normal_(self.linear_layer.weight)

        self.linear_layer_2 = nn.Linear(2, 1, bias=True)
        nn.init.xavier_normal_(self.linear_layer_2.weight)
        # self.act_layer = nn.()

        self.loss = nn.MSELoss()

    def forward(self, x, batch_data, model, pair_times):

        if self.ds_metric:
            g1s = x[0::2, ]  # num_graphs/2 by D
            g2s = x[1::2, ]
            pred_ds = self.ds_metric(g1s, g2s)
        else:
            pred_ds = x.squeeze()
            sum_preds = sum(pred_ds)
            if sum_preds / len(x) < 0.05:
                print("Weights poorly initialized please try running again")
                exit(1)

        return pred_ds

    def get_sub_list(self, pair_list, pair_times):
        sub_pair_list = []
        one_pair = []
        for i, pair in enumerate(pair_list):
            gid1 = pair.g1.gid()//10
            gid2 = pair.g2.gid()//10
            if i == 0:
                last_gid1 = gid1
                last_gid2 = gid2
            if gid1 == last_gid1 and gid2 == last_gid2:
                one_pair.append(pair)
                if i == len(pair_list)-1:
                    sub_pair_list.append(one_pair)
                    #print("last one",len(one_pair))
            else:
                sub_pair_list.append(one_pair)
                #print(len(one_pair))
                one_pair=[]
                one_pair.append(pair)
            last_gid1 = gid1
            last_gid2 = gid2
        #print(len(sub_pair_list),'  ',pair_times)

        return sub_pair_list