# from scipy.stats import wasserstein_distance
#
# d = wasserstein_distance([[0, 1], [0, 1]], [[0, 1], [0, 1]])
#
# print(d)


# from utils import load
#
# loaded = load('/home/yba/Documents/GraphMatching/model/OurMCS/logs/prototype_transformer_linux_2019-07-03T10-42-50.985102/final_test_pairs.klepto')
#
# print(loaded)
#
#
# from eval_ranking import eval_ranking
#
# result_dict, true_m, pred_m = eval_ranking(
#     true_ds_mat, pred_ds_mat, FLAGS.dos_pred, time_mat)

# import numpy as np
#
# from eval_pairs import _rc_thres_lsap
# # threshold = 0.5
# mat = np.array([[0, 0.45, 0, 0, 0], [0, 0, 0.8, 0.1, 0], [0, 0.45, 0, 0.3, 0], [0, 0, 0, 0, 0]])
# print(mat)
# # sum_rows = mat.sum(axis=1)
# # sum_cols = mat.sum(axis=0)
# # print('sum_rows', sum_rows)
# # print('sum_cols', sum_cols)
# # select_rows = sum_rows > threshold
# # select_cols = sum_cols > threshold
# # print('select_rows', select_rows)
# # print('select_cols', select_cols)
# # rtn = np.zeros(mat.shape)
# # rtn[select_rows,] = mat[select_rows,]
# # rtn[:, select_cols] = mat[:, select_cols]
# print(_rc_thres_lsap(mat, 0.5))
# # print(mat)

# import numpy as np
#
# mat = np.array([[0, 0, 0, 1, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 0]])
# print(mat)
# ind = np.where(mat == 0.5)
# print(ind)


# from functools import wraps
# import errno
# import os
# import signal
#
# class TimeoutError(Exception):
#     pass
#
# def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
#     def decorator(func):
#         def _handle_timeout(signum, frame):
#             raise TimeoutError(error_message)
#
#         def wrapper(*args, **kwargs):
#             signal.signal(signal.SIGALRM, _handle_timeout)
#             signal.alarm(seconds)
#             try:
#                 result = func(*args, **kwargs)
#             finally:
#                 signal.alarm(0)
#             return result
#
#         return wraps(func)(wrapper)
#
#     return decorator
#
# from time import sleep
#
#
# #
# #
# # @timeout(1)
# def long_running_function2():
#     sum = 0
#     for i in range(1000000):
#         sum += i / 5031
#     sleep(10)
#     return sum
#
#
# # result = long_running_function2()
# #
# # print(result)
# import signal
#
#
# class timeout:
#     def __init__(self, seconds=1, error_message='Timeout'):
#         self.seconds = seconds
#         self.error_message = error_message
#
#     def handle_timeout(self, signum, frame):
#         raise TimeoutError(self.error_message)
#
#     def __enter__(self):
#         signal.signal(signal.SIGALRM, self.handle_timeout)
#         signal.alarm(self.seconds)
#
#     def __exit__(self, type, value, traceback):
#         signal.alarm(0)
#
# try:
#     with timeout(seconds=3):
#         result = long_running_function2()
#         print(result)
# except TimeoutError as e:
#     print(e)


# from utils import load
#
# x = load('/home/yba/Documents/GraphMatching/model/OurMCS/logs/gmn_icml_mlp_mcs_debug_2019-09-01T23-09-37.641091/FLAGS.klepto')
# print(x)
# FLAGS = x['FLAGS']
# print(FLAGS.theta)

# import torch
#
# index = torch.Tensor([[0, 1], [1, 2], [2, 3]]).type(torch.long)
# A2 = torch.Tensor(
#     [[0, 1, 1, 1, 1], [1, 0, 0, 1, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]).type(torch.long)
#
# print(index)
# print(A2)
#
# index_A2 = torch.Tensor([1, 2, 3]).type(torch.long)
# q = torch.index_select(A2, 1, index_A2)
# print(q)




