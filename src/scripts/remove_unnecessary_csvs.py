import os
import sys
from glob import glob
from os.path import join
from pandas import read_csv


def remove_duplicate_csvs(folder_path):
    search_str_preprocessed = '*' + '*'.join(["mccreesh2017", "preprocessed_node_mapping", ".csv"])
    search_str_unprocessed = '*' + '*'.join(["mccreesh2017", ".csv"])
    search_str_bad_mappings = '*' + '*'.join(["mccreesh2017", "_bad_mappings.csv"])

    all_files =  set(glob(join(folder_path, search_str_unprocessed)))
    processed_files = set(glob(join(folder_path, search_str_preprocessed)))
    bad_mapping_files = set(glob(join(folder_path, search_str_bad_mappings)))
    files_to_process = list(all_files - processed_files - bad_mapping_files)

    # zero_time_files_to_remove = set()
    # for file in files_to_process:
    #     file_size = os.stat(file).st_size
    #     res_file_name = os.path.basename(os.path.splitext(file)[0])
    #     res_path = os.path.join(folder_path, res_file_name + "_preprocessed_node_mapping.csv")
    #     # if not os.path.exists(res_path) and not (file_size < 100):
    #     #     first_line = read_csv(file, sep=',', nrows=1)
    #     #     # if first_line["time(msec)"][0] == 0.0:
    #     #     #     zero_time_files_to_remove.add(file)

    # files_to_process = list(set(files_to_process))


    files_to_remove = set()
    for i, file in enumerate(files_to_process):
        if file in files_to_remove:
            continue
        else:
            file_size = os.stat(file).st_size
            res_file_name = os.path.basename(os.path.splitext(file)[0])
            res_path = os.path.join(folder_path, res_file_name + "_preprocessed_node_mapping.csv")
            if file_size < 100 or os.path.exists(res_path):
                files_to_remove.add(file)
                continue
            first_line = read_csv(file, sep=',', nrows=1)
            graph_pair = set([first_line["i_gid"][0], first_line["j_gid"][0]])
            zeros = first_line["time(msec)"][0] == 0.0
            for remain_file in files_to_process[i+1:]:
                remain_file_size = os.stat(remain_file).st_size
                if remain_file_size < 100:
                    files_to_remove.add(remain_file)
                    continue
                first_line_remain = read_csv(remain_file, sep=',', nrows=1)
                graph_pair_remain = set([first_line_remain["i_gid"][0], first_line_remain["j_gid"][0]])
                zeros_remain = first_line_remain["time(msec)"][0] == 0.0
                if graph_pair == graph_pair_remain:
                    if file_size > remain_file_size:
                        if not zeros or zeros_remain:
                            files_to_remove.add(remain_file)
                    else:
                        if zeros or not zeros_remain:
                            files_to_remove.add(file)

    for file in files_to_remove:
        os.remove(file)
    # for file in zero_time_files_to_remove:
    #     os.remove(file)

if __name__ == '__main__':
    if 'python' in sys.argv[0]:
        dataset_csv_path = sys.argv[2]
    else:
        dataset_csv_path = sys.argv[1]
    remove_duplicate_csvs(dataset_csv_path)