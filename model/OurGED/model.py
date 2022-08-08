from config import FLAGS
from layers_factory import create_layers
from layers import NodeEmbedding, create_act
from layers_subgraph import interact_two_sets_of_vectors, glorot, get_att
import torch.nn as nn
from time import time
from utils import Timer
from utils_our import get_branch_names, get_flag
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import numpy as np
import community
import networkx as nx
import matplotlib.pyplot as plt
import torch
from graph import RegularGraph
from dataset import OurDataset
from graph_pair import GraphPair
from batch import BatchData
import datetime
# import metis
import pickle
#from more_itertools import chunked

class Model(nn.Module):
    def __init__(self, data):
        super(Model, self).__init__()

        self.train_data = data
        self.num_node_feat = data.num_node_feat

        self.layers = create_layers(self, 'layer', FLAGS.layer_num)
        assert (len(self.layers) > 0)
        self._print_layers(None, self.layers)

        # Create layers for branches
        # (except for the main branch which has been created above).
        bnames = get_branch_names()
        
        print(bnames)#NONE
        if bnames:
            for bname in bnames:
                blayers = create_layers(
                    self, bname,
                    get_flag('{}_layer_num'.format(bname), check=True))
                setattr(self, bname, blayers)  # so that print(model) can print
                self._print_layers(bname, getattr(self, bname))

        self.layer_output = {}

        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, batch_data):
        t = time()
        total_loss = 0.0
        # Go through each layer except the last one.
        md = batch_data.merge_data['merge']
        acts = [md.x]
        if FLAGS.model != "simgnn_fast":
            pair_times = len(batch_data.pair_list) 
            test_timer = Timer()
            for k, layer in enumerate(self.layers):
                ln = layer.__class__.__name__
                # print('\t{}'.format(ln))
                # print(ln)
                if ln == "GraphConvolutionCollector":
                    gcn_num = layer.gcn_num
                    ins = acts[-gcn_num:]
                else:
                    ins = acts[-1]
                # print("ins",ins)
                outs = layer(ins, batch_data, self)
                acts.append(outs)
                # print(outs.shape)
            total_loss = acts[-1]
            
            return total_loss
        else:
            # so how to combine the mini batches and how to generate the ids? 
            pair_times = len(batch_data.pair_list)            
            true = torch.zeros(len(batch_data.pair_list), 1, device=FLAGS.device)
            pairwise_scores = torch.zeros(len(batch_data.pair_list), 1, device=FLAGS.device)
            pointer = 0
            #graph ids for GCN
            batch_gids = []
            pairs = {}
            NodeEmbedding_pairs = {}
            graphs = []
            used_graphs = {} # the graphs which appeared once in former pairs
            gid = 1 #far big graph ids
            sub_id = 0 #subid stands for the start id numbers for a group of subgraphs from the same graph
            subids = {}
            sub_node_num = []
            sub_node_fea_pos = []
            sub_node_fea_pos_iter = 0
            num_partitions = 2
            for i in range(pair_times):
                # print("Partitioning...")
                # print("--------------------------")
                # test_timer = Timer()
                pair = batch_data.pair_list[i]
                true[i] = pair.get_ds_true(
                    FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)
                # print("Ture",true)
                g1 = pair.g1.nxgraph
                g2 = pair.g2.nxgraph
                g1_node_num = len(g1.nodes())
                g2_node_num = len(g2.nodes())
                # print("g1_nodes",g1_node_num)
                # print("g1:",g1.nodes())
                # print("g2_nodes",g2_node_num)
                # print("g2:",g2.nodes())
                # print("x1", pair.g1.gid())
                # print(x1)
                # print("x2", pair.g2.gid())
                # print(x2)
                
                # print("acts",acts[0].shape)
                g1_feature = acts[0][pointer:pointer+g1_node_num]
                pointer = pointer+g1_node_num
                g2_feature = acts[0][pointer:pointer+g2_node_num]
                pointer = pointer+g2_node_num
                # g1_feature = acts[0][0:(g1_node_num)]
                # g2_feature = acts[0][0:(g2_node_num)]

                if FLAGS.load_sub_graph or FLAGS.save_sub_graph:
                    sub_graph_path = FLAGS.sub_graph_path
                    g1_sub_set_file = open(sub_graph_path+str(g1.graph['gid'])+'.pkl','rb')
                    g1_sub_set = pickle.load(g1_sub_set_file)
                    g2_sub_set_file = open(sub_graph_path+str(g2.graph['gid'])+'.pkl','rb')
                    g2_sub_set = pickle.load(g2_sub_set_file)
                else:
                    # print("ppp")
                    g1_sub_set = partitioning(g=g1,k=FLAGS.num_partitions,f=g1_feature,gid_start=pair.g1.gid()*10)    
                    g2_sub_set = partitioning(g=g2,k=FLAGS.num_partitions,f=g2_feature,gid_start=pair.g2.gid()*10)
                      
                # g1_sub_set_file = open(sub_graph_path+str(g1.graph['gid'])+'.pkl','rb')
                # g1_sub_set = pickle.load(g1_sub_set_file)
                # for sub in g1_sub_set:
                #     # print("sub g1",sub.get_nxgraph().nodes())
                #     print("sub g1",sub.gid())
                # print("g1_sub_set",g1_sub_set)
                #print("partition done!")
                g1_sub_num = len(g1_sub_set)
                for kk in range(g1_sub_num):
                    sub_node_num.append((g1_sub_set[kk].get_nodes_num()))
                
                # g2_sub_set_file = open(sub_graph_path+str(g2.graph['gid'])+'.pkl','rb')
                # g2_sub_set = pickle.load(g2_sub_set_file)
                # for sub in g2_sub_set:
                #     # print("sub g2",sub.get_nxgraph().nodes())
                #     print("sub g2",sub.gid())
                g2_sub_num = len(g2_sub_set)
                for kk in range(g2_sub_num):
                    sub_node_num.append((g2_sub_set[kk].get_nodes_num()))
                # print(sub_node_num)
                # print("partitioning g2",test_timer.time_msec_and_clear())
                
                for sg1 in range(g1_sub_num):
                    for sg2 in range(g2_sub_num):
                        # print("app",g1_sub_set[sg1].gid(),g2_sub_set[sg2].gid())
                        batch_gids.append([g1_sub_set[sg1].gid(),g2_sub_set[sg2].gid()])
                        pairs[(g1_sub_set[sg1].gid(),g2_sub_set[sg2].gid())] = GraphPair(g1=g1_sub_set[sg1],g2=g2_sub_set[sg2])
                        

                for sg1 in range(g1_sub_num):
                    if g1_sub_set[sg1].gid() not in used_graphs:
                        used_graphs[g1_sub_set[sg1].gid()] = g1_sub_set[sg1].gid()
                        graphs.append(g1_sub_set[sg1])
                        # print("append graph1",g1_sub_set[sg1].gid())

                for sg2 in range(g2_sub_num):
                    if g2_sub_set[sg2].gid() not in used_graphs:
                        used_graphs[g2_sub_set[sg2].gid()] = g2_sub_set[sg2].gid()
                        graphs.append(g2_sub_set[sg2])
                        # print("append graph2",g2_sub_set[sg2].gid())

                if i == 0:
                    flag = 0               
                for sub_g1 in g1_sub_set:
                    for sub_g2 in g2_sub_set:
                        if flag==0:
                            acts_mini = np.array(sub_g1.nxgraph.init_x)
                            acts_mini = np.vstack((acts_mini,sub_g2.nxgraph.init_x))
                            flag=1
                        else:
                            acts_mini = np.vstack((acts_mini,sub_g1.nxgraph.init_x))
                            acts_mini = np.vstack((acts_mini,sub_g2.nxgraph.init_x))

            batch_gids = torch.tensor(batch_gids)

            acts_mini = [torch.from_numpy(acts_mini).to(FLAGS.device)]

            tvt="train"
            align_metric="ged"
            node_ordering="bfs"
            name = "aids700nef"

            data = OurDataset(name=name,gs1=[],gs2=[],graphs=graphs,natts=['type'],eatts=[],pairs=pairs,tvt=tvt,align_metric=align_metric,node_ordering=node_ordering,glabel=None,loaded_dict=None,mini="mini")
            # NodeEmbedding_data = OurDataset(name=name,graphs=graphs,natts=['type'],eatts=[],pairs=NodeEmbedding_pairs,tvt=tvt,align_metric=align_metric,node_ordering=node_ordering,glabel=None,loaded_dict=None,mini="mini")

            batch_data_mini = BatchData(batch_gids, data)
            # NodeEmbedding_batch_data_mini = BatchData(batch_gids, NodeEmbedding_data, NodeEmbedding_pairs)
            # print("make feature",test_timer.time_msec_and_clear())

            flag = 0
            for k, layer in enumerate(self.layers):
                ln = layer.__class__.__name__
                # print(ln)
                if ln == "GraphConvolutionCollector":
                    gcn_num = layer.gcn_num
                    ins = acts_mini[-gcn_num:]
                else:
                    ins = acts_mini[-1]
                # print("ins",ins)

                if ln != "ANPM_FAST":
                    outs = layer(ins, batch_data_mini, self)
                else:
                    outs = layer(ins, batch_data_mini, self, pair_times)

                # print("outs",outs)
                # print("each layer",test_timer.time_msec_and_clear())
                acts_mini.append(outs)

            
            pairwise_scores = (acts_mini[-1]).reshape(-1,1)
        
            print("true",true.reshape(-1))
            print("pred",pairwise_scores.reshape(-1))
            # print("before sigmoid",acts_mini[-1])
            for i, pair in enumerate(batch_data.pair_list):
                pair.assign_ds_pred(pairwise_scores[i])
            
            loss = self.criterion(pairwise_scores, true)

            return loss
            
        #------------------------------------------------------------------------------
        
    def _forward_for_branches(self, acts, total_loss, batch_data):
        bnames = get_branch_names()
        if not bnames:  # no other branch besides the main branch (i.e. layers)
            return total_loss
        for bname in bnames:
            blayers = getattr(self, bname)
            ins = acts[get_flag('{}_start'.format(bname))]
            outs = None
            assert len(blayers) >= 1
            for layer in blayers:
                outs = layer(ins, batch_data, self)
                ins = outs
            total_loss += get_flag('{}_loss_alpha'.format(bname)) * outs
        return total_loss

    def store_layer_output(self, layer, output):
        self.layer_output[layer] = output

    def get_layer_output(self, layer):
        return self.layer_output[layer]  # may get KeyError/ValueError

    def _print_layers(self, branch_name, layers):
        print('Created {} layers{}: {}'.format(
            len(layers),
            '' if branch_name is None else ' for branch {}'.format(branch_name),
            ', '.join(l.__class__.__name__ for l in layers)))

def graph_node_resort(graph):
    g = nx.Graph()
    g_nodes = list(graph.nodes())
    g_edges = list(graph.edges())
    add_edges_list = []
    g.add_nodes_from(list(range(0,len(g_nodes))))
    for u,v,_ in (graph.edges.data()):
        # print(u,v)
        add_edges_list.append((g_nodes.index(u),g_nodes.index(v)))
    g.add_edges_from(add_edges_list)
    return g

# g -> origin graph
# k -> partition k graphs
# m -> select m graphs
# f -> origin graph's feature

def partitioning(g, k, f, gid_start):
    i=0
    while(1):
        i=i+1
        # print("begin partitioning")
        g_num_nodes = len(g.nodes())
        if i>30:
            #print("just origin graph")
            x = g.copy()
            x.graph['gid']=gid_start
            x.init_x = f
            return [RegularGraph(x)]
        if 3*k <= g_num_nodes:
            # part_time = Timer()
            comp = asyn_fluidc(g,k=k)#.to_undirected(), k=k)
            # community.best_partition
            # print("--------------------------")
            # print("comp",comp)
            # print("in partitioning",part_time.time_msec_and_clear())
        else:
            # print("just origin graph")
            x = g.copy()
            x.graph['gid']=gid_start
            x.init_x = f
            return [RegularGraph(x)]
        
        
        # # plt.show()
        # print("gid",g.graph['gid'])
        list_nodes = [frozenset(c) for c in comp]
        # # # print("list_nodes",list_nodes)
        # color = ['r','g','b','k','#ffb6c1']
        # pos = nx.spring_layout(g)
        # for i in range(k):
        #     # print(list_nodes[i])
        #     sub_nodes = [nodes for nodes in list_nodes[i]]
        #     # print(li_nodes)
        #     nx.draw_networkx_nodes(g,pos,sub_nodes,node_size=30,node_color=color[i])
        # nx.draw_networkx_edges(g,pos)
        # plt.savefig(str(g.graph['gid'])+".pdf")
        # plt.show()



        sub_graph_set = []
        break_flag = 1
        # extract feature from origin graph / select m subgraphs
        for i in range(k):
            j = 0
            for sub_node in list_nodes[i]:
                # temp = np.array(f[int(sub_node)])
                temp = (f[int(sub_node)].cpu().detach().numpy())
                if j==0:
                    sub_graph_feature = temp
                else:
                    sub_graph_feature = np.vstack((sub_graph_feature,temp))
                j=j+1
            sub_graph = g.subgraph(list_nodes[i]).copy()
            sub_graph = graph_node_resort((sub_graph))
            sub_graph.graph['gid']=gid_start+i
            sub_graph.init_x = torch.from_numpy(sub_graph_feature)
            # print("sub_graph ",sub_graph.graph['gid'])
            # print("sub fea:",sub_graph_feature)
            if len(sub_graph.nodes())!=1 and nx.is_connected(sub_graph):
                sub_graph_set.append(RegularGraph(sub_graph))
            else:
                 break_flag = 0
                 break
        if break_flag:
            return sub_graph_set
            break

def partitioning1(g, k, f, gid_start):
    begin = datetime.datetime.now()
    i=0
    while(1):
        i = i+1
        g_num_nodes = len(g.nodes())
        if i > 1:
            x = g.copy()
            x.graph['gid']=gid_start
            x.init_x = f
            # print("Original graph")
            return [RegularGraph(x)]
        if 3*k <= g_num_nodes:
            # _, parts = metis.part_graph(g, nparts=k, contig = True)
            _, parts = metis.part_graph(g, nparts=k)
        # else:
        #     x = g.copy()
        #     x.graph['gid']=gid_start
        #     x.init_x = f
        #     print("Original graph")
        #     return [RegularGraph(x)], [RegularGraph(x)]
        list_nodes = dict()
        idx = 0
        for subgraphi in parts:
            if subgraphi not in list_nodes.keys():
                list_nodes[subgraphi] = []
                list_nodes[subgraphi].append((list(g.nodes())[idx]))
            else:
                list_nodes[subgraphi].append((list(g.nodes())[idx]))
            idx = idx+1      

        color = ['r','g','b','k','#ffb6c1']
        pos = nx.spring_layout(g)
        for i in range(k):
            # print(list_nodes[i])
            sub_nodes = [nodes for nodes in list_nodes[i]]
            # print(li_nodes)
            nx.draw_networkx_nodes(g,pos,sub_nodes,node_size=30,node_color=color[i])
        nx.draw_networkx_edges(g,pos,with_labels=True)
        plt.show()

        sub_graph_set = []
        sub_graph_subset = []
        break_flag = 1
        # extract feature from origin graph / select m subgraphs
        for i in range(k):
            j = 0
            for sub_node in list_nodes[i]:
                # temp = np.array(f[int(sub_node)])
                temp = (f[int(sub_node)].detach().numpy())
                if j==0:
                    sub_graph_feature = temp
                else:
                    sub_graph_feature = np.vstack((sub_graph_feature,temp))
                j=j+1
            sub_graph = g.subgraph(list_nodes[i]).copy()
            sub_graph = graph_node_resort(sub_graph)
            sub_graph.graph['gid']=gid_start+i
            sub_graph.init_x = sub_graph_feature
            #if len(sub_graph.nodes())!=1 and nx.is_connected(sub_graph):
            if len(sub_graph.nodes())!=1:
                sub_graph_set.append(RegularGraph(sub_graph))
            else:
                break_flag = 0
                break
        
        if break_flag:
            # end = datetime.datetime.now()
            # k = end - begin
            # print(k.total_seconds())
            return sub_graph_set
            break

def partitioning2(g, k, f, gid_start):
    i=0
    while(1):
        i=i+1
        # print("begin partitioning")
        g_num_nodes = len(g.nodes())
        if i>30:
            #print("just origin graph")
            x = g.copy()
            x.graph['gid']=gid_start
            x.init_x = f
            return [RegularGraph(x)]
        if 3*k <= g_num_nodes:
            # part_time = Timer()
            # comp = asyn_fluidc(g,k=k)#.to_undirected(), k=k)
            # community.best_partition
            partition = community.best_partition(g)
            # print("--------------------------")
            # print("comp",comp)
            # print("in partitioning",part_time.time_msec_and_clear())
        else:
            # print("just origin graph")
            x = g.copy()
            x.graph['gid']=gid_start
            x.init_x = f
            return [RegularGraph(x)]
        
        
        size = float(len(set(partition.values())))
        pos = nx.spring_layout(g)
        color = ['r','g','b','k','#ffb6c1']
        count = 0.
        for com in set(partition.values()) :
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                                        if partition[nodes] == com]
            nx.draw_networkx_nodes(g, pos, list_nodes, node_size = 20,
                                        node_color = str(count / size))
            print("list_nodes",list_nodes)

        nx.draw_networkx_edges(g, pos, alpha=0.5)
        plt.show()

        # plt.show()
        # list_nodes = [frozenset(c) for c in comp]
        print("list_nodes",list_nodes)
        # color = ['r','g','b','k','#ffb6c1']
        # pos = nx.spring_layout(g)
        # for i in range(k):
        #     # print(list_nodes[i])
        #     sub_nodes = [nodes for nodes in list_nodes[i]]
        #     # print(li_nodes)
        #     nx.draw_networkx_nodes(g,pos,sub_nodes,node_size=30,node_color=color[i])
        # nx.draw_networkx_edges(g,pos,with_labels=True)
        # plt.show()


        sub_graph_set = []
        break_flag = 1
        # extract feature from origin graph / select m subgraphs
        for i in range(k):
            j = 0
            for sub_node in list_nodes[i]:
                # temp = np.array(f[int(sub_node)])
                temp = (f[int(sub_node)].cpu().detach().numpy())
                if j==0:
                    sub_graph_feature = temp
                else:
                    sub_graph_feature = np.vstack((sub_graph_feature,temp))
                j=j+1
            sub_graph = g.subgraph(list_nodes[i]).copy()
            sub_graph = graph_node_resort((sub_graph))
            sub_graph.graph['gid']=gid_start+i
            sub_graph.init_x = torch.from_numpy(sub_graph_feature)
            # print("sub_graph ",sub_graph.graph['gid'])
            # print("sub fea:",sub_graph_feature)
            if len(sub_graph.nodes())!=1 and nx.is_connected(sub_graph):
                sub_graph_set.append(RegularGraph(sub_graph))
            else:
                 break_flag = 0
                 break
        if break_flag:
            return sub_graph_set
            break