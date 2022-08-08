from dataset_config import get_dataset_conf, check_tvt, check_align, check_node_ordering
from dataset import OurDataset, OurOldDataset
# from bignn_dataset import BiLevelDataset
from utils import get_save_path, load, save
from os.path import join



def load_dataset(name, tvt, align_metric, node_ordering):
    name_list = [name]
    if not name or type(name) is not str:
        raise ValueError('name must be a non-empty string')
    check_tvt(tvt)
    name_list.append(tvt)
    align_metric_str = check_align(align_metric)
    name_list.append(align_metric_str)
    node_ordering = check_node_ordering(node_ordering)
    name_list.append(node_ordering)
    full_name = '_'.join(name_list)
    p = join(get_save_path(), 'dataset', full_name)
    ld = load(p)
    '''
    ######### this is solely for running locally lol #########
    ld['pairs'] = {(1022,1023):ld['pairs'][(1022,1023)],\
                   (1036,1037):ld['pairs'][(1036,1037)], \
                   (104,105):ld['pairs'][(104,105)],\
                   (1042,1043):ld['pairs'][(1042,1043)],\
                   (1048,1049):ld['pairs'][(1048,1049)],\
                   }
    '''
    if ld:
        _, _, _, _, _, dataset_type, _ = get_dataset_conf(name)
        if dataset_type == 'OurDataset':
            rtn = OurDataset(None, None, None, None, None, None, None, None,
                             None, ld)
        elif dataset_type == 'OurOldDataset':
            rtn = OurOldDataset(None, None, None, None, None, None, None, None,
                                None, None, None, ld)
        # elif dataset_type == "BiLevelDataset":
        #     rtn = BiLevelDataset(None, None, None, None, None, None, None, None,
        #                          None, None, None, None, ld)
        else:
            raise NotImplementedError()
    else:

        rtn = _load_dataset_helper(name, tvt, align_metric, node_ordering)
        save(rtn.__dict__, p)
    if rtn.num_graphs() == 0:
        raise ValueError('{} has 0 graphs'.format(name))
    return rtn


def _load_dataset_helper(name, tvt, align_metric, node_ordering):
    natts, eatts, tvt_options, align_metric_options, loader, _, glabel = \
        get_dataset_conf(name)
    if tvt not in tvt_options:
        raise ValueError('Dataset {} only allows tvt options '
                         '{} but requesting {}'.
                         format(name, tvt_options, tvt))
    if align_metric not in align_metric_options:
        raise ValueError('Dataset {} only allows alignment metrics '
                         '{} but requesting {}'.
                         format(name, align_metric_options, align_metric))
    assert loader
    return loader(name, natts, eatts, tvt, align_metric, node_ordering, glabel)


if __name__ == '__main__':
    # import networkx.algorithms.isomorphism as iso
    # import networkx as nx
    # import numpy as np
    # from tqdm import tqdm
    # import torch
    #
    #
    # def _gen_nx_subgraph(y_mat, pair):
    #     # y_mat[0][1] = 1
    #     indices_left = _gen_nids(y_mat, 1)
    #     indices_right = _gen_nids(y_mat, 0)
    #     g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
    #     g1_subgraph = g1.subgraph(indices_left)  # g1 is left
    #     g2_subgraph = g2.subgraph(indices_right)  # g2 is right
    #     # print(indices_left, '@@@\n', indices_right)
    #     return g1_subgraph, g2_subgraph, g1.graph['gid'], g2.graph['gid']
    #
    #
    # def _gen_nids(y_mat, axis):
    #     if axis == 0:
    #         rtn = np.where(y_mat == 1)[1]
    #     elif axis == 1:
    #         rtn = np.where(y_mat.T == 1)[1]
    #     else:
    #         assert False
    #     return list(rtn)


    '''
    p = '/home/yba/Documents/GraphMatching/save/OurModelData/alchemy_train_test_mcs_bfs_one_hot_v2.klepto'
    rtn = load(p)
    train_data, test_data = rtn['train_data'], rtn['test_data']

    dataset = train_data.dataset
    pairs_pop = []
    for pair in tqdm(dataset.pairs):
        gid1, gid2 = pair
        if gid1 > gid2:
            pairs_pop.append(pair)
    for pair in pairs_pop:
        train_data.dataset.pairs.pop(pair)
    gid_pairs = list(dataset.pairs.keys())
    train_data.gid1gid2_list = torch.tensor(
        sorted(gid_pairs),
        device='cpu')

    dataset = test_data.dataset
    pairs_pop = []
    for pair in tqdm(dataset.pairs):
        gid1, gid2 = pair
        if gid1 > gid2:
            pairs_pop.append(pair)
    for pair in pairs_pop:
        test_data.dataset.pairs.pop(pair)
    gid_pairs = list(dataset.pairs.keys())
    test_data.gid1gid2_list = torch.tensor(
        sorted(gid_pairs),
        device='cpu')

    p = '/home/yba/Documents/GraphMatching/save/OurModelData/alchemy_train_test_mcs_bfs_one_hot_v2.klepto'
    save({'train_data': train_data, 'test_data': test_data}, p)
    '''

    '''
    name = 'alchemy'
    dataset = load_dataset(name, 'all', 'mcs', 'bfs')
    dataset.print_stats()
    
    tp,fp,fn = [],[],[]
    i = 0
    for pair in tqdm(dataset.pairs):
        i+=1
        if i == 1000:
            break
        gid1, gid2 = pair
        pair = dataset.look_up_pair_by_gids(gid1, gid2)
        pair.assign_g1_g2(dataset.look_up_graph_by_gid(gid1), dataset.look_up_graph_by_gid(gid2))
        y_true_mat_list = pair.get_y_true_list_mat_view(format='numpy')
        true_left, true_right, true_gid1, true_gid2 = _gen_nx_subgraph(y_true_mat_list[0], pair)
        natts, eatts, *_ = get_dataset_conf(name)
        nm = iso.categorical_node_match(natts, [''] * len(
            natts))  # TODO: check the meaning of default value
        em = iso.categorical_edge_match(eatts, [''] * len(
            eatts))  # TODO: check the meaning of default value
        is_iso = nx.is_isomorphic(true_left, true_right, node_match=nm, edge_match=em)
        if not is_iso and gid1 > gid2:
            tp.append((gid1,gid2))
            print('pn: {}'.format((gid1,gid2)))
        elif not is_iso and gid1 <= gid2:
            fp.append((gid1,gid2))
            print('fp: {}'.format((gid1,gid2)))
        elif is_iso and gid1 > gid2:
            fn.append((gid1,gid2))
    print(tp)
    print(fp)
    print(fn)
    print('tp:{}\nfp:{}\nfn:{}'.format(len(tp),len(fp),len(fn)))
    '''
    # print(dataset)
    # print(dataset.gs)
    # pair = dataset.look_up_pair_by_gids(165, 20679)
    # print(pair)
    # g1 = dataset.look_up_graph_by_gid(165)
    # g2 = dataset.look_up_graph_by_gid(20679)

    # name = 'redditmulti10k'
    # dataset = load_dataset(name, 'all', 'mcs', 'bfs')
    # print(dataset)
    # dataset.print_stats()
    # dataset.save_graphs_as_gexf()

    name = 'pdb'
    dataset = load_dataset(name, 'all', 'interaction', 'bfs')
    print(dataset)
    dataset.print_stats()
    dataset.save_graphs_as_gexf()

