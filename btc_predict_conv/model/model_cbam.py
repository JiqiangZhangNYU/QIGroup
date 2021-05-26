# -*- coding: UTF-8 -*-
"""
@author: Jiqiang Zhang
"""

import torch
import torch.nn as nn
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
from CBAM.model_resnet import *

class FinLoss(Module):
    def __init__(self, multiplier, indicator='rf'):
        super(FinLoss, self).__init__()  # 继承Module属性
        self.multiplier = multiplier
        self.indicator = indicator

    def forward(self, pred_y, train_y):
        m1 = torch.nn.Tanh()
        gp = train_y * m1(pred_y * self.multiplier)
        cost = 0.002
        tick_size = 1
        cost_penalty = 1
        cost = cost * tick_size / 2.0
        cost = cost_penalty * cost
        delta_ps = torch.abs(torch.diff(gp, dim=1))
        gpcost = (delta_ps.sum(dim=1) * cost).unsqueeze(1)
        netp = torch.sub(gp, gpcost)
        profit = torch.cumsum(netp, dim=1)

        if self.indicator == 'rf':
            dd = torch.cummax(profit, dim=1).values - profit
            mdd = torch.max(dd, dim=1).values + 0.01
            rf = torch.mean(torch.sum(netp, dim=1) / mdd)
            ind_loss = -rf
        else:
            sharpe = torch.mean(torch.mean(netp, dim=1) / (torch.std(netp, dim=1)+0.001))
            ind_loss = -sharpe
        return ind_loss

def train(config, logger, train_and_valid_data):

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()     # 先转为Tensor
    train_X=train_X.unsqueeze(1)    # 4 dimensions
    train_Y=train_Y[:,-1]
    print(train_X.shape,train_Y.shape)
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)    # DataLoader可自动生成可训练的batch数据

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_X=valid_X.unsqueeze(1)    # 4 dimensions
    valid_Y=valid_Y[:,-1]
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU训练还是GPU
    model = ResidualNet( config.network_type, config.depth, config.num_classes, config.att_type)
    # totnum=0
    # for parameters in model.parameters():
    #     layernum=1
    #     for ele in parameters.shape:
    #         layernum=ele*layernum
    #     totnum+=layernum
    #     print(layernum)
    # print(totnum)
    
    model = model.to(device)      # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中
    model = nn.DataParallel(model)
    if config.add_train:                # 如果是增量训练，会先加载原模型参数
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    #scheduler = lr_scheduler.StepLR(80, 0.1)
    if config.loss_type == 'Fin':
        criterion = FinLoss(multiplier=10, indicator='sr')      # 这两句是定义优化器和loss
    elif config.loss_type == 'entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()                   # pytorch中，训练时要转换成训练模式
        train_loss_array = []
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            pred_Y = model(_train_X)    # 这里走的就是前向计算forward函数, 注意pred和train的shape
            #print(pred_Y.shape,_train_Y.shape)
            loss = criterion(pred_Y, _train_Y)  # 计算loss
            loss.backward()                     # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数
            train_loss_array.append(loss.item())

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()                    # pytorch中，预测时要转换成预测模式
        valid_loss_array = []
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y= model(_valid_X)
            loss = criterion(pred_Y, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
              "The valid loss is {:.6f}.".format(valid_loss_cur))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                logger.info(" The training stops early in epoch {}".format(epoch))
                break


def predict(config, test_X):
    # 获取测试数据
    test_X = torch.from_numpy(test_X).float()
    test_X=test_X.unsqueeze(1)    # 4 dimensions
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # 加载模型
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = ResidualNet( config.network_type, config.depth, config.num_classes, config.att_type)
    model = model.to(device)
    state_dict = torch.load(config.model_save_path + config.model_name,
                            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))   # 加载模型参数
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "module." in key:
            new_name = key[7:]
        else:
            new_name = key
        new_state_dict[new_name] = value
    model.load_state_dict(new_state_dict)   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        with torch.no_grad():
            pred_X= model(data_X)
        # if not config.do_continue_train: hidden_predict = None    # 实验发现无论是否是连续训练模式，把上一个time_step的hidden传入下一个效果都更好
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据
