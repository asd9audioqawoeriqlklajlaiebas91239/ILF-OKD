#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from datetime import datetime
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from module import teacher_ef, teacher_graph, student_graph, student_ef
import os
from tqdm import tqdm
import torch.nn.utils.prune as prune
import networkx as nx
from scipy import spatial
import collections
import argparse
from utils import read_data, get_sharpe_ratio, make_trading_decision, compute_factor_weights



class IDF_OKD:
    def __init__(self, Input_ef, Input_et, Input_co, Input_pt, Input_vt, Label_y, company_list, company_dates, Input_comp_idx, ret, args, seq_len=10):

        self.batch = args.batch_size
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.Lambda = args.Lambda
        self.args = args
        self.avg = torch.tensor(0.0).to(args.cuda)
        self.cnt = 0
        self.fisher = {}

        self.avg_s = torch.tensor(0.0).to(args.cuda)
        self.cnt_s = 0
        self.fisher_s = {}
        self.fisher1_s = {}

        train_size = int(0.7 * len(Input_ef))
        self.learning_rate = 0.01
        self.check_result_train = company_dates[: train_size]
        self.check_result_test = company_dates[train_size * self.batch: ]
        self.seq_len = seq_len

        ## train X ##
        self.train_x_ef = Input_ef[: train_size]
        self.train_x_et = Input_et[: train_size]
        self.train_x_co = Input_co[: train_size]
        self.train_x_pt = Input_pt[: train_size]
        self.train_x_vt = Input_vt[: train_size]
        self.train_x_comp_idx = Input_comp_idx[: train_size]

        ## train y ##
        self.train_y = Label_y[: train_size]

        ## test X ##
        self.test_x_ef = Input_ef[train_size:]
        self.test_x_et = Input_et[train_size:]
        self.test_x_co = Input_co[train_size:]
        self.test_x_pt = Input_pt[train_size:]
        self.test_x_vt = Input_vt[train_size:]
        self.test_x_comp_idx = Input_comp_idx[train_size: ]

        ## test y ##
        self.test_y = Label_y[train_size: ]
        self.test_ret = ret[train_size * self.batch:]

        self.company_list = company_list

        self.s_graph = student_graph(comps=company_list, seq_len=seq_len, args=args).to(args.cuda)
        self.s_ef = student_ef(comps=company_list, seq_len=seq_len, args=args).to(args.cuda)
        self.t_graph = student_graph(comps=company_list, seq_len=seq_len, args=args).to(args.cuda)
        self.t_ef = student_ef(comps=company_list, seq_len=seq_len, args=args).to(args.cuda)

        self.optimizer_s_graph_s_ef = torch.optim.SGD(list(self.s_graph.parameters()) + list(self.s_ef.parameters()), lr=self.learning_rate)
        self.optimizer_t_graph_t_ef = torch.optim.SGD(list(self.t_graph.parameters()) + list(self.t_ef.parameters()),  lr=self.learning_rate)
        self.student_net_params = list(self.s_graph.named_parameters()) + list(self.s_ef.named_parameters())
        self.teacher_net_params = list(self.t_graph.named_parameters()) + list(self.t_ef.named_parameters())
        self.profits = []

        # print(sum(p.numel() for p in self.teacher_net.parameters() if p.requires_grad))

    def _update_network_params(self, network_name, net_params):
        if network_name == 'teacher':
            self.teacher_net_params = net_params
        else:
            self.student_net_params = net_params

    def _updata_fisher(self, batch_size, labels, network_name):

        if network_name == 'teacher':
            net_params = self.teacher_net_params.copy()
        else:
            net_params = self.student_net_params.copy()

        if self.cnt == 0:
            for n, p in net_params:
                self.avg += torch.dist(torch.zeros_like(p.grad.data ** 2 / batch_size), p.grad.data ** 2 / batch_size)
                self.fisher[n] = p.grad.data ** 2 / batch_size
            self.cnt += 1
            self._update_network_params(network_name, net_params)
            return True

        else:
            tmp = torch.tensor(0.0).to(args.cuda)
            for n, p in net_params:
                tmp += torch.dist(self.fisher[n], p.grad.data ** 2 / batch_size, 2)
                self.fisher[n] = p.grad.data ** 2 / batch_size
            self._update_network_params(network_name, net_params)
            if tmp > self.avg:
                self.cnt += 1
                self.avg = (self.avg * (self.cnt - 1) + tmp) / self.cnt
                return True
            else:
                return False

    def Update(self, loss, labels, optimizer, network_name):

        loss = self.Lambda * loss
        loss.backward()

        with torch.no_grad():
            action = self._updata_fisher(1, labels, network_name)

        if action:
            optimizer.step()
            return loss.item()
        else:
            return 0.0

    def batch_train(self, epochs):

        self.train_x_ef = torch.tensor(self.train_x_ef).float().to(args.cuda)
        self.train_x_et = torch.tensor(self.train_x_et).float().to(args.cuda)
        self.train_y = torch.tensor(self.train_y).long().to(args.cuda)
        # print(self.train_y.size())
        self.train_x_pt = torch.tensor(self.train_x_pt).float().to(args.cuda)
        self.train_x_co = torch.tensor(self.train_x_co).float().to(args.cuda)
        self.train_x_vt = torch.tensor(self.train_x_vt).float().to(args.cuda)
        criterion_out = nn.CrossEntropyLoss().to(args.cuda)
        criterion_g = nn.MSELoss().to(args.cuda)
        criterion_ef = nn.MSELoss().to(args.cuda)

        for epoch in range(epochs):
            print('EPOCH: ', epoch + 1)
            a = datetime.now()

            if epoch == 20 or epoch == 40:
                self.optimizer_s_graph_s_ef = torch.optim.SGD(
                    list(self.s_graph.parameters()) + list(self.s_ef.parameters()), lr=self.learning_rate * 0.1)
                self.optimizer_t_graph_t_ef = torch.optim.SGD(
                    list(self.t_graph.parameters()) + list(self.t_ef.parameters()), lr=self.learning_rate * 0.1)


            confusion_matrix = torch.zeros(2, 2)
            Loss = 0.0

            for i in range(len(self.train_x_ef)):

                input_x_ef = self.train_x_ef[i]
                input_x_et = self.train_x_et[i]
                input_x_pt = self.train_x_pt[i]
                input_x_co = self.train_x_co[i]
                input_x_vt = self.train_x_vt[i]
                actual = self.train_y[i].to(args.cuda).view(-1)

                ## teacher losses ##
                self.optimizer_t_graph_t_ef.zero_grad()

                t_graph = self.t_graph(input_x_et, input_x_co, input_x_pt, input_x_vt, self.train_x_comp_idx[i])
                s_graph = self.s_graph(input_x_et, input_x_co, input_x_pt, input_x_vt, self.train_x_comp_idx[i])
                s_graph_copy = s_graph.detach().requires_grad_()
                t_graph_copy = t_graph.detach().requires_grad_()

                out1, ef1, _ = self.t_ef(input_x_ef, t_graph)
                out2, ef2, _ = self.s_ef(input_x_ef, s_graph)

                ef2_copy = ef2.detach().requires_grad_()
                out2_copy = out2.detach().requires_grad_()
                _, preds = torch.max(out2, 1)

                out1_sub, _, _ = self.s_ef(input_x_ef, t_graph)

                loss1 = criterion_out(out1, actual)
                loss1_graph = criterion_g(t_graph, s_graph)
                loss1_ef = criterion_ef(ef1, ef2)
                loss1_sub = criterion_out(out1_sub, actual)

                loss = loss1 + self.alpha * loss1_sub + self.beta * loss1_graph + self.gamma * loss1_ef
                _ = self.Update(loss, actual.view(-1), self.optimizer_t_graph_t_ef, 'teacher')

                ## student losses ##

                self.optimizer_s_graph_s_ef.zero_grad()

                out2_sub, ef2_sub, _ = self.t_ef(input_x_ef, s_graph_copy)
                t_graph = self.t_graph(input_x_et, input_x_co, input_x_pt, input_x_vt, self.train_x_comp_idx[i])
                _, ef1, _ = self.t_ef(input_x_ef, t_graph)

                loss2 = criterion_out(out2_copy, actual)
                loss2_graph = criterion_g(t_graph_copy, s_graph_copy)
                loss2_ef = criterion_ef(ef1, ef2_copy)
                loss2_sub = criterion_out(out2_sub, actual)

                loss = loss2 + self.alpha * loss2_sub + self.beta * loss2_graph + self.gamma * loss2_ef

                _ = self.Update(loss, actual.view(-1), self.optimizer_s_graph_s_ef, 'student')

                Loss += loss

                for t, p in zip(actual.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            confusion_matrix = confusion_matrix.detach().numpy()
            print(confusion_matrix)
            b = datetime.now()
            print('Training !')
            print('Loss of student network: ', Loss / len(self.train_x_ef), 'Time Cost: ', b - a)
            print('class 0 proportion: ', np.sum(confusion_matrix[0, :]) / np.sum(confusion_matrix))
            print('class 1 proportion: ', np.sum(confusion_matrix[1, :]) / np.sum(confusion_matrix))

            print('class 0 recall: ', np.sum(confusion_matrix[0, 0]) / np.sum(confusion_matrix[0, :]))
            print('class 1 recall: ', np.sum(confusion_matrix[1, 1]) / np.sum(confusion_matrix[1, :]))

            print('class 0 precision: ', np.sum(confusion_matrix[0, 0]) / np.sum(confusion_matrix[:, 0]))
            print('class 1 precision: ', np.sum(confusion_matrix[1, 1]) / np.sum(confusion_matrix[:, 1]))

            if epoch % args.test_interval == 0:
                self.batch_test(epoch)

    def batch_test(self, epoch):

        self.test_x_ef = torch.tensor(self.test_x_ef).float().to(args.cuda)
        self.test_x_et = torch.tensor(self.test_x_et).float().to(args.cuda)
        self.test_y = torch.tensor(self.test_y).long().to(args.cuda)
        self.test_x_pt = torch.tensor(self.test_x_pt).float().to(args.cuda)
        self.test_x_co = torch.tensor(self.test_x_co).float().to(args.cuda)
        self.test_x_vt = torch.tensor(self.test_x_vt).float().to(args.cuda)

        a = datetime.now()

        confusion_matrix = torch.zeros(2, 2)
        dates = []
        rets = collections.defaultdict(list)
        factors = collections.defaultdict(float)
        comp_name_dict = {}

        with torch.no_grad():
            for i in range(len(self.test_x_ef)):
                for b in range(self.batch):
                    input_x_ef = self.test_x_ef[i][b].unsqueeze(0)
                    input_x_et = self.test_x_et[i][b].unsqueeze(0)
                    input_x_co = self.test_x_co[i][b].unsqueeze(0)
                    input_x_pt = self.test_x_pt[i][b].unsqueeze(0)
                    input_x_vt = self.test_x_vt[i][b].unsqueeze(0)
                    input_x_comp_idx = self.test_x_comp_idx[i][b]

                    actual = self.test_y[i].to(args.cuda).view(-1)
                    s_graph = self.s_graph(input_x_et, input_x_co, input_x_pt, input_x_vt, input_x_comp_idx)
                    out2, _, attn = self.s_ef(input_x_ef, s_graph)
                    _, preds = torch.max(out2, 1)

                    # date, dates, rets, ret_data, ef, idx
                    num_x = i * b + b
                    dates, rets = make_trading_decision(self.check_result_test[i][1], dates, rets, self.test_ret, input_x_ef, num_x)

                    for t, p in zip(actual.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

                    # print('preds: ', preds.item())
                    if preds == actual and preds.item() == 1 and args.print_factor_weights:
                        compute_factor_weights(attn, factors, comp_name_dict, self.company_list, num_x)

        print('dates: ', len(dates), dates)
        confusion_matrix = confusion_matrix.detach().numpy()
        print(confusion_matrix)
        b = datetime.now()
        print('Testing Time Cost: {}.'.format(b - a))
        print('class 0 proportion: ', np.sum(confusion_matrix[0, :]) / np.sum(confusion_matrix))
        print('class 1 proportion: ', np.sum(confusion_matrix[1, :]) / np.sum(confusion_matrix))

        print('class 0 recall: ', np.sum(confusion_matrix[0, 0]) / np.sum(confusion_matrix[0, :]))
        print('class 1 recall: ', np.sum(confusion_matrix[1, 1]) / np.sum(confusion_matrix[1, :]))

        print('class 0 precision: ', np.sum(confusion_matrix[0, 0]) / np.sum(confusion_matrix[:, 0]))
        print('class 1 precision: ', np.sum(confusion_matrix[1, 1]) / np.sum(confusion_matrix[:, 1]))



def main(args):

    Input_ef, Input_et, Input_co, Input_pt, Input_vt, Label_y, company_list, company_dates, Input_comp_idx, ret = read_data(args)

    model = IDF_OKD(Input_ef, Input_et, Input_co, Input_pt, Input_vt, Label_y, company_list, company_dates, Input_comp_idx, ret, args)
    model.batch_train(args.epochs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SePaL')

    parser.add_argument('--model', type=str, default='IDF-OKD',
                        choices=['Event'], help='Event (default: Event)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of iterations to train (default: 3500)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed (default: 1)')
    parser.add_argument('--path', type=str, default='dataset/',
                        help='project directory')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='the number of different event types')
    parser.add_argument('--region', type=str, default='US', choices=['Europe'],
                        help='various market regions')
    parser.add_argument('--sector', type=str, default='technology', choices=['finance', 'healthcare'],
                        help='various market sectors')
    parser.add_argument('--print_factor_weights', type=bool, default=False,
                        help='it can be activated with RavenPack API access code')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='how many iterations between testing phases')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--Lambda', type=float, default=0.9)

    args = parser.parse_args()
    args.cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)