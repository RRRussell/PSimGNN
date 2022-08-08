from config import FLAGS
from model import Model
from utils_our import get_flag
from utils import Timer
from torch.utils.data import DataLoader
from batch import BatchData
import torch
import numpy as np
import os
import sys
from torch.optim.lr_scheduler import LambdaLR
from eval import Eval

def train(train_data, saver, test_data=None):
    print('creating model...')
    model = Model(train_data)
    model = model.to(FLAGS.device)
    # print(model)
    saver.log_model_architecture(model)
    model.train_data = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    # print("model parameters",model.parameters())
    # for name, p in model.named_parameters():
    #     print(name,":",p.size())
    num_epochs = get_flag('num_epochs')
    num_epochs = num_epochs if num_epochs is not None else int(1e5)
    num_iters_total = 0
    save_every_epochs = FLAGS.save_every_epochs
    for epoch in range(num_epochs):
        print("----------------")
        # model.train()
        loss, num_iters_total, stop = _train_epoch(
            epoch, num_iters_total, train_data, model, optimizer, saver)
        if FLAGS.save_model and epoch%save_every_epochs==0:
            saver.save_trained_model(model,epoch=str(epoch))

            # test(test_data, model, saver)
            # eval = Eval(model, train_data, test_data, saver)
            # eval.eval_on_test_data()
        # if epoch%500==0:
        # scheduler.step()
        # print("***")
        # test(test_data, model, saver)
        if stop:
            break
    return model


def _train_epoch(epoch, num_iters_total, data, model, optimizer, saver):
    epoch_timer = Timer()
    iter_timer = Timer()
    data_loader = DataLoader(data, batch_size=FLAGS.batch_size, shuffle=True)

    total_loss = 0
    num_iters = 0
    stop = False
    for iter, batch_gids in enumerate(data_loader):
        # print((batch_gids))
        # print(batch_gids.shape)
        # print(type(data.dataset))
        # print("-------------------------------------------------------------------")
        temp = data.dataset
        # print(temp.name)
        # print(temp.gs[0].nxgraph.graph)#['gid'])
        # print(temp.gs[1].nxgraph.graph['gid'])
        # print(temp.gs_map)
        # print(temp.id_map)
        # print(temp.natts)
        # print(temp.eatts)
        # print(temp.pairs[(1322, 1025)].y_true_dict_list)
        # print(temp.pairs[(1322, 1025)].y_true_mat_list)
        # print(temp.tvt)
        # print(temp.align_metric)
        # print(temp.node_ordering)
        # print(temp.glabel)
        
        batch_data = BatchData(batch_gids, data.dataset)
        # print(type(batch_data))
        # print(type(batch_data.pair_list[0].g1))
        loss = _train_iter(
            iter, num_iters_total, epoch, batch_data, data_loader, model,
            optimizer, saver, iter_timer)

        # print(iter, batch_data, batch_data.num_graphs, len(loader.dataset))
        total_loss += loss
        num_iters += 1
        num_iters_total += 1
        if num_iters == FLAGS.only_iters_for_debug:
            stop = True
            break
        num_iters_total_limit = get_flag('num_iters')
        if num_iters_total_limit is not None and \
                num_iters_total == num_iters_total_limit:
            stop = True
            break
    saver.log_tvt_info('Epoch: {:03d}, Loss: {:.7f} ({} iters)\t\t{}'.format(
        epoch + 1, total_loss / num_iters, num_iters,
        epoch_timer.time_msec_and_clear()))
    return total_loss, num_iters_total, stop


def _train_iter(iter, num_iters_total, epoch, batch_data, data_loader, model,
                optimizer, saver, iter_timer):
    # model.zero_grad()
    optimizer.zero_grad()
    loss = model(batch_data)
    loss.backward()
    optimizer.step()
    saver.writer.add_scalar('loss/loss', loss, epoch * len(data_loader) + iter)
    loss = loss.item()
    if np.isnan(loss):
        sys.exit(0)
    if iter == 0 or (iter + 1) % FLAGS.print_every_iters == 0:
        saver.log_tvt_info('\tIter: {:03d} ({}), Loss: {:.7f}\t\t{}'.format(
            iter + 1, num_iters_total + 1, loss,
            iter_timer.time_msec_and_clear()))
    return loss


def continue_train(train_data, saver, model, begin_epoch, test_data=None):
    print('creating model...')
    # model = Model(train_data)
    model = model.to(FLAGS.device)
    # print(model)
    saver.log_model_architecture(model)
    model.train_data = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    # print("model parameters",model.parameters())
    # for name, p in model.named_parameters():
    #     print(name,":",p.size())
    num_epochs = get_flag('num_epochs')
    num_epochs = num_epochs if num_epochs is not None else int(1e5)
    num_iters_total = 0
    save_every_epochs = FLAGS.save_every_epochs
    for epoch in range(begin_epoch, num_epochs):
        print("----------------")
        loss, num_iters_total, stop = _train_epoch(
            epoch, num_iters_total, train_data, model, optimizer, saver)
        if FLAGS.save_model and epoch%save_every_epochs==0:
            saver.save_trained_model(model,epoch=str(epoch))
        # if epoch%500==0:
        # scheduler.step()
        # print("***")
        # test(test_data, model, saver)
        if stop:
            break
    return model

def test(data, model, saver):
    model.eval()
    model.test_data = data
    _test_loop(data, model, saver)


def _test_loop(data, model, saver):
    import numpy as np
    data_loader = DataLoader(data, batch_size=FLAGS.batch_size, shuffle=False)
    num_iters = 0
    iter_timer = Timer()
    for iter, batch_gids in enumerate(data_loader):
        batch_data = BatchData(batch_gids, data.dataset)
        loss = model(batch_data)
        loss = loss.item()
        # print(iter,"score:",score)
        # print("true",true)
        # delta = score-true
        # mse = delta**2
        # mse = np.mean(mse.detach().numpy())
        # print(iter,"mse",mse)
        saver.log_tvt_info('\tIter: {:03d}, Test Loss: {:.7f}\t\t{}'.format(
            iter + 1, loss, iter_timer.time_msec_and_clear()))
        num_iters += 1
        if num_iters == FLAGS.only_iters_for_debug:
            break