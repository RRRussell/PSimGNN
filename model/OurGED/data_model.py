from load_data import load_dataset
from node_feat import encode_node_features
from config import FLAGS
from torch.utils.data import Dataset as TorchDataset
import torch
from utils_our import get_flags_with_prefix_as_list
from utils import get_save_path, save, load
from os.path import join
from warnings import warn
import os
import networkx as nx
from graph import RegularGraph
import random
from graph_pair import GraphPair
from dataset import OurDataset, OurOldDataset
import csv
import shutil
import numpy as np
# import metis
import pickle
from networkx.algorithms.community.asyn_fluid import asyn_fluidc

class OurModelData(TorchDataset):
    """Stores a list of graph id pairs with known pairwise results."""

    def __init__(self, dataset, num_node_feat):
        self.dataset, self.num_node_feat = dataset, num_node_feat

        gid_pairs = list(self.dataset.pairs.keys())
        self.gid1gid2_list = torch.tensor(
            sorted(gid_pairs),
            device=FLAGS.device)  # takes a while to move to GPU
    

    def __len__(self):
        return len(self.gid1gid2_list)

    def __getitem__(self, idx):
        return self.gid1gid2_list[idx]

    def get_pairs_as_list(self):
        return [self.dataset.look_up_pair_by_gids(gid1.item(), gid2.item())
                for (gid1, gid2) in self.gid1gid2_list]

    def truncate_large_graphs(self):
        gid_pairs = list(self.dataset.pairs.keys())
        if FLAGS.filter_large_size < 1:
            raise ValueError('Cannot filter graphs of size {} < 1'.format(
                FLAGS.filter_large_size))
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1)
            g2 = self.dataset.look_up_graph_by_gid(gid2)
            if g1.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size and \
                    g2.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU

    def select_specific_for_debugging(self):
        gid_pairs = list(self.dataset.pairs.keys())
        gids_selected = FLAGS.select_node_pair.split('_')
        assert(len(gids_selected) == 2)
        gid1_selected, gid2_selected = int(gids_selected[0]), int(gids_selected[1])
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1).get_nxgraph()
            g2 = self.dataset.look_up_graph_by_gid(gid2).get_nxgraph()
            if g1.graph['gid'] == gid1_selected and g2.graph['gid'] == gid2_selected:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        FLAGS.select_node_pair = None # for test
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU

def load_train_test_data():
    # tvt = 'train'
    dir = join(get_save_path(), 'OurModelData')
    sfn = '{}_train_test_{}_{}_{}'.format(
        FLAGS.dataset, FLAGS.align_metric, FLAGS.node_ordering,
        '_'.join(get_flags_with_prefix_as_list('node_fe')))
    '''
    sfn = '{}_train_test_{}_{}_{}{}{}'.format(
        FLAGS.dataset, FLAGS.align_metric, FLAGS.node_ordering,
        '_'.join(get_flags_with_prefix_as_list('node_fe')),
        _none_empty_else_underscore(FLAGS.filter_large_size),
        _none_empty_else_underscore(FLAGS.select_node_pair))
    '''
    tp = join(dir, sfn)
    # version option
    print(FLAGS.dataset)
    if FLAGS.dataset in ['aids700nef', 'linux', 'imdbmulti', 'alchemy']:
        if FLAGS.dataset_version != None:
            tp += '_{}'.format(FLAGS.dataset_version)
    print("tp",tp)
    rtn = load(tp)
    if rtn:
        train_data, test_data = rtn['train_data'], rtn['test_data']
        #train_data, test_data = rtn[0:sp], rtn[sp:]
    else:
        train_data, test_data = _load_train_test_data_helper()
        save({'train_data': train_data, 'test_data': test_data}, tp)
    # if FLAGS.validation:
    #     all_spare_ratio = 1 - FLAGS.throw_away
    #     train_val_ratio = 0.6 * all_spare_ratio
    #     dataset = train_data.dataset
    #     dataset.tvt = 'all'
    #     if all_spare_ratio != 1:
    #         dataset_train, dataset_test, _ = dataset.tvt_split(
    #             [train_val_ratio, all_spare_ratio], ['train', 'validation', 'spare'])
    #     else:
    #         dataset_train, dataset_test = dataset.tvt_split(
    #             [train_val_ratio], ['train', 'validation'])
    #     assert train_data.num_node_feat == test_data.num_node_feat
    #     train_data = OurModelData(dataset_train, train_data.num_node_feat)
    #     test_data = OurModelData(dataset_test, test_data.num_node_feat)

    # if FLAGS.filter_large_size is not None:
    #     print('truncating graphs...')
    #     train_data.truncate_large_graphs()
    #     test_data.truncate_large_graphs()

    # if FLAGS.select_node_pair is not None:
    #     print('selecting node pair...')
    #     train_data.select_specific_for_debugging()
    #     test_data.select_specific_for_debugging()

    train_data.dataset.print_stats()
    test_data.dataset.print_stats()

    # dir = join(get_save_path(), 'anchor_data')
    # sfn = '{}_{}_{}_{}'.format(
    #     FLAGS.dataset, FLAGS.align_metric, FLAGS.node_ordering,
    #     '_'.join(get_flags_with_prefix_as_list('node_fe')))
    # tp = join(dir, sfn)
    # rtn = load(tp)
    # if rtn:
    #     train_anchor, test_anchor = rtn['train_anchor'], rtn['test_anchor']
    #     train_data.dataset.generate_anchors(train_anchor)
    #     test_data.dataset.generate_anchors(test_anchor)
    # else:
    #     train_anchor = train_data.dataset.generate_anchors(None)
    #     test_anchor = test_data.dataset.generate_anchors(None)
    #     save({'train_anchor': train_anchor, 'test_anchor': test_anchor}, tp)
    #
    # # load to device
    # def load_to_device(dataset, device = FLAGS.device):
    #     for i, g in enumerate(dataset.dataset.gs):
    #         dataset.dataset.gs[i].nxgraph.graph['dists_max'] = g.nxgraph.graph['dists_max'].to(device)
    #         dataset.dataset.gs[i].nxgraph.graph['dists_argmax'] = g.nxgraph.graph['dists_argmax'].to(
    #             device)
    # load_to_device(train_data)
    # load_to_device(test_data)

    return train_data, test_data


def _none_empty_else_underscore(v):
    if v is None:
        return ''
    return '_{}'.format(v)


def _load_train_test_data_helper():
    if FLAGS.tvt_options == 'all':
        dataset = load_dataset(FLAGS.dataset, 'all', FLAGS.align_metric,
                               FLAGS.node_ordering)
        dataset.print_stats()
        # Node feature encoding must be done at the entire dataset level.
        print('Encoding node features')
        dataset, num_node_feat = encode_node_features(dataset=dataset)
        print('Splitting dataset into train test')
        dataset_train, dataset_test = dataset.tvt_split(
            [FLAGS.train_test_ratio], ['train', 'test'])
    elif FLAGS.tvt_options == 'train,test':
        dataset_test = load_dataset(FLAGS.dataset, 'test', FLAGS.align_metric,
                                    FLAGS.node_ordering)
        dataset_train = load_dataset(FLAGS.dataset, 'train', FLAGS.align_metric,
                                     FLAGS.node_ordering)
        dataset_train, num_node_feat_train = \
            encode_node_features(dataset=dataset_train)
        dataset_test, num_node_feat_test = \
            encode_node_features(dataset=dataset_test)
        if num_node_feat_train != num_node_feat_test:
            raise ValueError('num_node_feat_train != num_node_feat_test '
                             '{] != {}'.
                             format(num_node_feat_train, num_node_feat_test))
        num_node_feat = num_node_feat_train
    else:
        print(FLAGS.tvt_options)
        raise NotImplementedError()
    dataset_train.print_stats()
    dataset_test.print_stats()
    train_data = OurModelData(dataset_train, num_node_feat)
    test_data = OurModelData(dataset_test, num_node_feat)
    return train_data, test_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    from torch.utils.data.sampler import SubsetRandomSampler
    from batch import BatchData
    import random

    # print(len(load_dataset(FLAGS.dataset).gs))
    data = OurModelData()
    print(len(data))
    # print('model_data.num_features', data.num_features)
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(dataset_size * 0.2)
    random.Random(123).shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader = DataLoader(data, batch_size=3, shuffle=True)
    print(len(loader.dataset))
    for i, batch_gids in enumerate(loader):
        print(i, batch_gids)
        batch_data = BatchData(batch_gids, data.dataset)
        print(batch_data)
        # print(i, batch_data, batch_data.num_graphs, len(loader.dataset))
        # print(batch_data.sp)

def cal_ged(g1,g2):
    g1_gid = g1.graph['gid']
    g2_gid = g2.graph['gid']
    # print(g1_gid,g2_gid)
    if g1_gid == g2_gid:
        ged = 0
    elif g1_gid<100 & g2_gid<100:
        ged = len(g1.nodes())+len(g2.nodes())
    elif g1_gid < 100:
        if g1_gid != g2_gid//100:
            ged = len(g1.nodes())+len(g2.nodes())
        else:
            ged = g2_gid%100
    elif g2_gid < 100:
        if g2_gid != g1_gid//100:
            ged = len(g1.nodes())+len(g2.nodes())
        else:
            ged = g1_gid%100
    else:
        if g1_gid//100 != g2_gid//100:
            ged = len(g1.nodes())+len(g2.nodes())
        else:
            ged = g1_gid%100 + g2_gid%100
    # print(ged)
    return ged

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
            # print("--------------------------")
            # print("comp",comp)
            # print("in partitioning",part_time.time_msec_and_clear())
        else:
            # print("just origin graph")
            x = g.copy()
            x.graph['gid']=gid_start
            x.init_x = f
            return [RegularGraph(x)]
        
        
        # plt.show()
        list_nodes = [frozenset(c) for c in comp]
        # # print("list_nodes",list_nodes)
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

def partitioning1(g, k, f, gid_start):
    # begin = datetime.datetime.now()
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
            # comp
            # parts = asyn_fluidc(g,k=k)
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
        sub_graph_subset = []
        break_flag = 1
        # extract feature from origin graph
        # print("in part f",f.shape)
        # print("in part s",list_nodes)
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
            sub_graph = graph_node_resort(sub_graph.copy())
            sub_graph.graph['gid']=gid_start+(i)
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

def load_our_data(train_path,csv_path):
    
    train_graph_set = []
    gs1 = []
    gs2 = []
    
    num_node_feat = FLAGS.num_node_feat
    num_partitions = FLAGS.num_partitions

    sub_graph_path = FLAGS.sub_graph_path

    query = FLAGS.rank

    if not query:
        for g_file in os.listdir(train_path):
            g = nx.read_gexf(train_path+g_file)
            g.graph['gid'] = int(g_file.replace(".gexf",""))
            g_feature = torch.ones(len(g.nodes()), num_node_feat, device=FLAGS.device)
            # g_feature = torch.zeros(len(g.nodes()), num_node_feat, device=FLAGS.device)
            # for i in g.nodes():
            #     g_feature[int(i),random.randint(0,num_node_feat-1)] = 1
                # g_feature[int(i),num_node_feat//2] = 1
            g.init_x = g_feature
            # print(g.init_x)
            train_graph_set.append(RegularGraph(g))
        random.shuffle(train_graph_set)
        pairs = {}
        length = len(train_graph_set)
        for sg1 in range(length):
            for sg2 in range(length):
                # print(sg1,sg2)
                gid1,gid2 = train_graph_set[sg1].gid(),train_graph_set[sg2].gid()
                # pairs[gid1,gid2] = GraphPair(ds_true=cal_ged(g1=train_graph_set[sg1].get_nxgraph(),g2=train_graph_set[sg2].get_nxgraph()),g1=train_graph_set[sg1],g2=train_graph_set[sg2])
                pairs[gid1,gid2] = GraphPair(g1=train_graph_set[sg1],g2=train_graph_set[sg2])
    else:        
        for g_file in os.listdir(train_path+"query/"):
            g = nx.read_gexf(train_path+"query/"+g_file)
            g.graph['gid'] = int(g_file.replace(".gexf",""))
            g_feature = torch.ones(len(g.nodes()), num_node_feat, device=FLAGS.device)
            # g_feature = torch.zeros(len(g.nodes()), num_node_feat, device=FLAGS.device)
            # for i in g.nodes():
            #     g_feature[int(i),random.randint(0,num_node_feat-1)] = 1
                # g_feature[int(i),num_node_feat//2] = 1
            g.init_x = g_feature
            # print(g.init_x)
            gs1.append(RegularGraph(g))
            train_graph_set.append(RegularGraph(g))
        random.shuffle(gs1)

        for g_file in os.listdir(train_path+"database/"):
            g = nx.read_gexf(train_path+"database/"+g_file)
            g.graph['gid'] = int(g_file.replace(".gexf",""))
            g_feature = torch.ones(len(g.nodes()), num_node_feat, device=FLAGS.device)
            # g_feature = torch.zeros(len(g.nodes()), num_node_feat, device=FLAGS.device)
            # for i in g.nodes():
            #     g_feature[int(i),random.randint(0,num_node_feat-1)] = 1
                # g_feature[int(i),num_node_feat//2] = 1
            g.init_x = g_feature
            # print(g.init_x)
            gs2.append(RegularGraph(g))
            train_graph_set.append(RegularGraph(g))
        random.shuffle(gs2)
        random.shuffle(train_graph_set)
        pairs = {}
        
        for sg1 in range(len(gs1)):
            for sg2 in range(len(gs2)):
                gid1,gid2 = gs1[sg1].gid(),gs2[sg2].gid()
                pairs[gid1,gid2] = GraphPair(g1=gs1[sg1],g2=gs2[sg2])
                # pairs[gid2,gid1] = GraphPair(g1=gs2[sg2],g2=gs1[sg1])

    if FLAGS.model == "simgnn_fast" and FLAGS.save_sub_graph and not FLAGS.load_sub_graph:
        if not os.path.exists(sub_graph_path):
            os.makedirs(sub_graph_path)
        for i in range(len(train_graph_set)):
            if not os.path.exists(sub_graph_path+str(train_graph_set[i].gid())+'.pkl'):
                g_sub_set = partitioning(g=train_graph_set[i].get_nxgraph(),k=num_partitions,f=train_graph_set[i].get_nxgraph().init_x,gid_start=(train_graph_set[i].gid()*10))
                # print("----------------")
                # print(train_graph_set[i].gid())
                pickle.dump(g_sub_set, open(sub_graph_path+str(train_graph_set[i].gid())+'.pkl','wb'))

    # 读取csv至字典
    csvFile = open(csv_path,"r")
    reader = csv.reader(csvFile)
    
    next(reader)
    # print(reader)
    count = 1
    for item in reader:
    #     # 忽略第一行

        g1 = int(item[0])
        g2 = int(item[1])
        true = int(item[2])

        # pairs[(g1,g2)] = GraphPair(ds_true=true,g1=train_graph_set[sg1],g2=train_graph_set[sg2])
        if (g1,g2) in pairs:
            # print("g1g2",g1,g2)
            (pairs[(g1,g2)]).input_ds_true(true)
            # print("count",count)
            # count += 1
 
    name = "our_data"
    tvt="train"
    align_metric="ged"
    node_ordering="bfs"
    data = OurDataset(name=name,gs1=gs1,gs2=gs2,graphs=train_graph_set,natts=['type'],eatts=[],pairs=pairs,tvt=tvt,align_metric=align_metric,node_ordering=node_ordering,glabel=None,loaded_dict=None,mini="mini",my="my")
    # data = OurOldDataset(name=name,gs1=gs1,gs2=gs2,all_gs=train_graph_set,natts=['type'],eatts=[],pairs=pairs,tvt=tvt,align_metric=align_metric,node_ordering=node_ordering,glabel=None,loaded_dict=None,mini="mini",my="my")
    train_data = OurModelData(data, num_node_feat)
    return train_data

def load_dzh_data():
    # rank_data = load_our_data(FLAGS.test_dir_path,FLAGS.csv_path)
    train_data = load_our_data(FLAGS.train_dir_path,FLAGS.csv_path)
    test_data = load_our_data(FLAGS.test_dir_path,FLAGS.csv_path)
#
    return train_data, test_data
    # return train_data, train_data
    # return test_data, test_data
    # return rank_data, rank_data
