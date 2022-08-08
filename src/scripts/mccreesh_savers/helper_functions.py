import re
from scripts.mccreesh_savers.save_graph_mcs import *
from scripts.mccreesh_savers.save_graph_sip import *
from os.path import join

def get_graph_pairs(file_name, path, exp_name, ignore_list, timeout_time):
    csv_file = open(join(path, 'experiments', 'gpgnode-results', exp_name,
                         file_name), 'r', newline='')
    reader = csv.reader(csv_file, delimiter=' ')
    names = next(reader)[1:]
    graph_pairs = []
    for row in reader:
        for i, model_runtime in enumerate(row[1:]):
            model_runtime = float(model_runtime)
            if (model_runtime < timeout_time) and (i not in ignore_list):
                graph_pairs.append((row[0], names[i]))
                break
    csv_file.close()
    return graph_pairs


def get_mappings(graph_pairs, exp_name, path):
    mappings = []
    algorithms = []
    for file_name, directory_name in graph_pairs:
        metadata_file = open(join(path, 'experiments', 'gpgnode-results', exp_name, directory_name, '{}.out'.format(file_name)), 'r')
        line = metadata_file.readline()
        while line:
            # if (directory_name == 'clique' and line[0:4] == 'true') \
            # or (directory_name == 'kdown' and line[0:4] == 'true') \
            # or (directory_name == 'kdown-par/t32' and line[0:4] == 'true') \
            if (directory_name == 'james-cpp-max' and line[0:8] == 'Solution') \
                    or (directory_name == 'james-cpp-max-down' and line[0:8] == 'Solution'):
                mappings.append(metadata_file.readline()[:-1])
                algorithms.append(directory_name)
                metadata_file.close()
                break
            line = metadata_file.readline()
        if not line:
            print('Could not find mapping for ' + directory_name + '/' + file_name + '.out')
            mappings.append(None)
            algorithms.append(None)
    return mappings, algorithms


def get_ignore_list(experiment):
    if experiment == 'mcs33v':
        ignore = [0, 1, 2]
    elif experiment == 'mcs33ve':
        ignore = [0, 1]
    elif experiment == 'mcs33ve-connected':
        ignore = [0, 2, 3, 4, 5, 6]
    elif experiment == 'mcs33ved':
        ignore = [0, 1, 2, 3, 4, 6, 7, 8]
    elif experiment == 'mcsplain':
        ignore = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    elif experiment == 'mcsplain-connected':
        ignore = [0, 2, 3, 4, 5, 6]
    elif experiment == 'sip':
        ignore = [0, 1, 2, 5]
    else:
        ignore = []
    return ignore


def get_statistics(total):
    statistics = {
        'mcs': 0,
        'sip/si': 0,
        'sip/scalefree': 0,
        'sip/images-CVIU11': 0,
        'sip/meshes-CVIU11': 0,
        'sip/PR15': 0,
        'sip/LV': 0
    }

    for x in total:
        if x[0] in ['g']:
            statistics['sip/LV'] += 1
        elif x[:2] in ['A.', 'B.', 'C.', 'D.', 'E.', 'F.']:
            statistics['sip/scalefree'] += 1
        elif x[:3] in ['mcs']:
            statistics['mcs'] += 1
        elif x[:2] in ['si']:
            statistics['sip/si'] += 1
        elif x[:4] in ['PR15']:
            statistics['sip/PR15'] += 1
        elif x[:13] in ['images-CVIU11']:
            statistics['sip/images-CVIU11'] += 1
        elif x[:13] in ['meshes-CVIU11']:
            statistics['sip/meshes-CVIU11'] += 1
        else:
            print('ERROR: could not find ' + x)
    return statistics


def get_graph_filenames(x, path):
    file_1 = path
    file_2 = path
    if x[0] in ['g']:
        file_1 += 'sip-instances/LV/' + x.split('-')[0]
        file_2 += 'sip-instances/LV/' + '-'.join(x.split('-')[1:])
        graph_dataset = 'LV'
    elif x[:2] in ['A.', 'B.', 'C.', 'D.', 'E.', 'F.']:
        file_1 += 'sip-instances/scalefree/' + x + '/pattern'
        file_2 += 'sip-instances/scalefree/' + x + '/target'
        graph_dataset = 'scalefree'
    elif x[:3] in ['mcs']:
        x_split = x.split('_')
        mcs_size, graph_type = x_split[0:2]
        graph_id = x_split[-1]
        if graph_type[0] == 'b':
            graph_type_file_name = 'bvg/' + graph_type + '/'
        elif graph_type[0] == 'r':
            graph_type_file_name = 'rand/' + graph_type + '/'
        elif graph_type[0:3] == 'm2D':
            graph_type_file_name = 'm2D/' + graph_type + '/'
        elif graph_type[0:3] == 'm3D':
            graph_type_file_name = 'm3D/' + graph_type + '/'
        elif graph_type[0:3] == 'm4D':
            graph_type_file_name = 'm4D/' + graph_type + '/'
        file_1 += 'mcs-instances/' + mcs_size + '/' + graph_type_file_name + \
                  '_'.join(x_split[:-1]) + '.A' + graph_id[2:]
        file_2 += 'mcs-instances/' + mcs_size + '/' + graph_type_file_name + \
                  '_'.join(x_split[:-1]) + '.B' + graph_id[2:]
        graph_dataset = 'mcs'
    elif x[:2] in ['si']:
        x_split = x.split('.')[0].split('_')
        graph_type_file_name = x_split[0] + '_'
        x_split[-1] = x_split[-1].replace('m', '')

        graph_type = x_split[1]
        if graph_type[0] == 'b':
            graph_type_file_name += 'bvg_'
        elif graph_type[0] == 'r':
            graph_type_file_name += 'rand_'
        elif graph_type[0:3] == 'm2D':
            graph_type_file_name += 'm2D_'
        elif graph_type[0:3] == 'm3D':
            graph_type_file_name += 'm3D_'
        elif graph_type[0:3] == 'm4D':
            graph_type_file_name += 'm4D_'

        graph_type_file_name += '_'.join(x_split[1:])

        file_1 += 'sip-instances/si/' + graph_type_file_name + '/' + x + '/pattern'
        file_2 += 'sip-instances/si/' + graph_type_file_name + '/' + x + '/target'
        graph_dataset = 'si'
    elif x[:4] in ['PR15']:
        file_1 += 'sip-instances/images-PR15/pattern' + x.split('-')[1]
        file_2 += 'sip-instances/images-PR15/target'
        graph_dataset = 'PR15'
    elif x[:13] in ['images-CVIU11']:
        x_split = x.split('-')
        file_1 += 'sip-instances/images-CVIU11/patterns/pattern' + x_split[2]
        file_2 += 'sip-instances/images-CVIU11/targets/target' + x_split[3]
        graph_dataset = 'images-CVIU11'
    elif x[:13] in ['meshes-CVIU11']:
        x_split = x.split('-')
        file_1 += 'sip-instances/meshes-CVIU11/patterns/pattern' + x_split[2]
        file_2 += 'sip-instances/meshes-CVIU11/targets/target' + x_split[3]
        graph_dataset = 'meshes-CVIU11'
    else:
        print('ERROR: could not find ' + x)
        graph_dataset = 'unknown'

    return file_1, file_2, graph_dataset


def get_processed_graph(graph_filename, graph_dataset, ignore_features):
    if 'mcs' == graph_dataset:
        # these functions can be found in save_graph_mcs.py
        clean_directories()
        decompress_graph(graph_filename)
        num_nodes = np.genfromtxt('temp/n.txt', delimiter=',')
        adj_mat = np.genfromtxt('temp/adj_matrix.txt', delimiter=',')
        if not ignore_features:
            node_features = np.genfromtxt('temp/node_features.txt', delimiter=',')
            edge_features = np.genfromtxt('temp/edge_features.txt', delimiter=',')
        g = get_nodes(num_nodes)
        g = get_edges(g, adj_mat)
        if not ignore_features:
            g = get_node_features(g, node_features)
            g = get_edge_features(g, edge_features, adj_mat)
        '''
        clean_directories()
        decompress_graph()

        adj_mat = np.genfromtxt('temp/adj_matrix.txt', delimiter=',')
        node_features = np.genfromtxt('temp/node_features.txt', delimiter=',')
        edge_features = np.genfromtxt('temp/edge_features.txt', delimiter=',')

        g = get_graph(adj_mat)
        g = get_node_features(g, node_features)
        g = get_edge_features(g, edge_features, adj_mat)
        '''
    else:
        # these functions can be found in save_graph_sip.py
        adj_mat = get_adj_mat(graph_filename)
        g = get_graph(adj_mat)

    print(graph_filename + ' processed.')
    return g


def get_processed_mapping(mapping):
    raw_mapping = [[int(node) for node in nodes[1:-1].split(',')] for nodes in
                   re.sub(' -> ', ',', mapping).split(' ')[:-1]]
    mapping = {}
    for node_pair in raw_mapping:
        mapping[node_pair[0]] = node_pair[1]
    return [mapping]


'''
def get_training_sample(graph_filenames, graph_dataset, mapping):
    graphs = []
    for i, graph_filename in enumerate(graph_filenames):
        if 'mcs' == graph_dataset:
            # these functions can be found in save_graph_mcs.py
            clean_directories()
            decompress_graph(graph_filename)

            adj_mat = np.genfromtxt('temp/adj_matrix.txt', delimiter=',')
            node_features = np.genfromtxt('temp/node_features.txt', delimiter=',')
            edge_features = np.genfromtxt('temp/edge_features.txt', delimiter=',')

            g = get_graph(adj_mat)
            g = get_node_features(g, node_features)
            g = get_edge_features(g, edge_features, adj_mat)
        else:
            # these functions can be found in save_graph_sip.py
            adj_mat = get_adj_mat(graph_filename)
            g = get_graph(adj_mat)

        print(graph_filename + ' processed.')
        graphs.append(g)

    mapping = [[int(node) for node in nodes[1:-1].split(',')] for nodes in re.sub(' -> ', ',', mapping).split(' ')[:-1]]
    
    return graphs[0], graphs[1], mapping
'''
