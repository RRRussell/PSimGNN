'''
@Author: your name
@Date: 2020-02-18 20:52:33
@LastEditTime: 2020-03-12 09:12:18
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /test/GraphMatching-master/model/OurMCS/main.py
'''
#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

from config import FLAGS
from model import Model
from data_model import load_train_test_data
from data_model import load_our_data
from data_model import load_dzh_data
from train import train, test, continue_train
from utils_our import get_model_info_as_str, check_flags, \
    convert_long_time_to_str
from utils import slack_notify, get_ts
from saver import Saver
from eval import Eval
from time import time
from os.path import basename
import torch
import traceback
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import time
import numpy as np
import community

def main():
    # train_data, test_data = load_train_test_data()
    # train_data, test_data = load_our_data()
    train_data, test_data = load_dzh_data()
    # if not FLAGS.traditional_method:
    print('Training...')
    if FLAGS.load_model is not None:
        print('loading model: {}', format(FLAGS.load_model))
        trained_model = Model(train_data).to(FLAGS.device)
        trained_model.load_state_dict(torch.load(FLAGS.load_model))
        print('model loaded')
        # continue_train(train_data,saver,trained_model,begin_epoch=23,test_data=test_data)
    else:
        trained_model = train(train_data, saver, test_data=test_data)

    print("====================================")
    print('Testing...')
    test(test_data, trained_model, saver)
    eval = Eval(trained_model, train_data, test_data, saver)
    eval.eval_on_test_data()

if __name__ == '__main__':
    # print(get_model_info_as_str())
    check_flags()
    saver = Saver()
    FLAGS.sub_graph_path = saver.get_log_dir()+"/subgraph/"
    # FLAGS.load_sub_graph_path = "/home/russell/russell/GraphMatching_BA/model/OurGED/logs/simgnn_fast_aids700nef_2020-05-02T01-20-03.573002/subgraph/"
    main()