# -*- coding: UTF-8 -*-
"""
@author: Jiqiang Zhang
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class Config:
    # 数据参数
    #filename='all_ret15min_volume15min'
    frame = "cbam"               # lstm, senet, cbam
    filename='btc_pv240min'
    feature_columns = list(range(4, 9))     # 要作为feature的列，按原数据从0开始计算，也可以用list 如 [2,4,6,8] 设置
    label_columns = [3]                  # 要预测的列，按原数据从0开始计算, 标准因子为第三列
    predict_day = 0            # label列平移数
    time_step = 20              # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数

    if frame =='lstm':
        input_size = len(feature_columns)
        output_size = len(label_columns)
        hidden_size = 128           # LSTM的隐藏层大小，也是输出大小
        lstm_layers = 2                # LSTM的堆叠层数
        dropout_rate = 0.2          # dropout概率        
    elif frame=='senet':
        num_classes = 1            #输出单元数
        reduction = 16
    elif frame=='cbam':
        network_type = 'ImageNet'    # "ImageNet", "CIFAR10", "CIFAR100"
        depth = 18                          # 18, 34, 50, 101
        num_classes = 1
        att_type = 'CBAM'               # BAM, CBAM
    else:
        raise Exception("Wrong frame selection")
    
    # 训练参数
    do_train = True   
    add_train = False           # 是否载入已有模型参数进行增量训练
    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型
    do_predict = True
    shuffle_train_data = True   # 是否对训练数据做shuffle
    use_cuda = True            # 是否使用GPU训练

    train_data_rate = 0.95      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.15      # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.001
    epoch = 100                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现
    weight_decay = 1e-6
    if_normalize = False
    loss_type = 'mse'           #损失函数，包括mse, entropy, finloss

    continue_flag = ""           # 连续模式，仅能以 batch_size = 1 训练
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # 训练模式
    debug_mode = False  # 调试模式下，是为了跑通代码，追求快
    debug_num = 500  # 仅用debug_num条数据来调试

    # 框架参数
    used_frame = frame  # 选择的深度学习框架
    model_postfix = {"lstm": ".pth", "senet":".pth", "cbam":".pth"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # 路径参数
    train_data_path = "./data/"+filename+".csv"
    model_save_path = "./checkpoint/" + used_frame + "/"+filename+"/"
    pred_save_path="./prediction/"+ used_frame + "/"+filename+"/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True                  # 是否将config和训练过程记录到log
    do_figure_save = True
    do_train_visualized = False
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name, self.data_index = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)
        self.time_step = self.config.time_step
        self.mean=0
        self.std=1
        self.norm_data= self.data
        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):                # 读取初始数据
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path, nrows=self.config.debug_num,
                                    usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.train_data_path,
                                    usecols=self.config.label_columns+ self.config.feature_columns)

        return init_data.values, init_data.columns.tolist(), init_data.index  # .columns.tolist() 是获取列名

    def get_train_and_valid_data(self):
        if self.config.if_normalize:
            self.mean = np.mean(self.data[:self.train_num], axis=0)  # 数据的均值和方差
            self.std = np.std(self.data[:self.train_num], axis=0)
            self.norm_data = (self.data - self.mean) / self.std  # 归一化，去量纲

        feature_data = self.norm_data[:self.train_num,1:]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,[0]]

        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            # 在连续训练模式下，每time_step行数据会作为一个样本，两个样本错开time_step行，
            # 比如：1-20行，21-40行。。。到数据末尾，然后又是 2-21行，22-41行。。。到数据末尾，……
            # 这样才可以把上一个样本的final_state作为下一个样本的init_state，而且不能shuffle
            # 目前本项目中仅能在pytorch的RNN系列模型中用
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:, 1:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)     # 防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        if self.config.frame == 'lstm':
            # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行, 比如：1-20行，21-40行。。。到数据末尾。
           test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                      for i in range(time_step_size)]
           label_data = self.norm_data[self.train_num + self.start_num_in_test:, [0]]
        else:
            # 在测试数据中，每time_step行数据会作为一个样本，连续取样，如1-20行，2-21行。。。
            test_x= [feature_data[self.start_num_in_test+i : self.start_num_in_test+i+sample_interval]
                     for i in range(feature_data.shape[0]-self.start_num_in_test-sample_interval)]
            label_data = self.norm_data[self.train_num + self.start_num_in_test + sample_interval:, [0]]
        
        if return_label_data:       # 实际应用中的测试集是没有label数据的
            return np.array(test_x), label_data
        else:
            return np.array(test_x)

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger


def pred_to_trade(config: Config, origin_data: Data, predict_norm_data: np.ndarray):
    # 保存预测结果，label长度和保存路径可调整
    df_total = pd.read_csv(config.train_data_path, index_col=0)
    if config.frame == 'lstm':
        label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test:, [0]]
        df_test = df_total[['index', 'symbol']].iloc[origin_data.train_num + origin_data.start_num_in_test:]
    else:
        label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test + origin_data.time_step:,[0]]
        df_test = df_total[['index', 'symbol']].iloc[origin_data.train_num + origin_data.start_num_in_test + origin_data.time_step:]

    if config.if_normalize:
        predict_data = predict_norm_data * origin_data.std[[0]] + origin_data.mean[[0]]
    else:
        predict_data = predict_norm_data

    df_test['real']=label_data
    df_test['pred'] = predict_data
    # 根据不同trigger计算仓位
    for trig in np.logspace(-4,-2,10):
        sr2=df_test['pred'].copy()
        #sr=sr2/sr2.shift(1)-1
        sr=sr2.copy()
        sr[sr>trig]=1
        sr[sr<-trig]=-1
        sr[(sr>-trig)&(sr<trig)]=np.nan
        sr=sr.fillna(method='ffill')
        df_test['trig_%.5f'%(trig)] = sr
    sr3=np.tanh(100 * df_test['pred'])
    df_test['trig_tanh'] = sr3
    df_test.to_csv(config.pred_save_path + 'pred.csv')

    for colname in df_test.columns:
        if colname not in ['index','symbol']:
            df_real=df_test[['index','symbol', colname]]
            real_data = pd.pivot(df_real, index="index", columns="symbol")
            real_data.columns=[ele[1] for ele in real_data.columns]
            real_data.index.name='index'
            factor_dir='../ps_backtest/factors/'+config.frame+'_btc/'
            if not os.path.exists(factor_dir):
                os.makedirs(factor_dir)
            real_data.to_csv(factor_dir +colname + '.csv')

def main(config):
    if config.frame =='lstm':
        from model.model_lstm import train, predict   
    elif config.frame=='senet':
        from model.model_senet import train, predict
    elif config.frame=='cbam':
        from model.model_cbam import train, predict
    else:
        raise Exception("Wrong frame selection")
    logger = load_logger(config)
    #try:
    np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
    data_gainer = Data(config)

    if config.do_train:
        train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
        train(config, logger, [train_X, train_Y, valid_X, valid_Y])

    if config.do_predict:
        test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
        print(test_X.shape, test_Y.shape)
        pred_result = predict(config, test_X)       # 这里输出的是未还原的归一化预测数据
        pred_to_trade(config, data_gainer, pred_result)
##    except Exception:
##        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--do_train", default=False, action="store_true", help="whether to train")
    parser.add_argument("-p", "--do_predict", default=False, action="store_true", help="whether to predict")
    #parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    parser.add_argument("-a", "--add_train", default=False, action="store_true", help="load model state")
    parser.add_argument("-r", "--train_data_rate", default=0.95, type=float, help="train data rate")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    main(con)
