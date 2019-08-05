"""
主程序，负责将所有模块组合起来，包括：
1.预训练数据处理保存和加载
2.词向量语预训练和保存
3.训练数据处理和加载
4.LSTM+CRF模型训练
5.结果输出
"""
import time
import pickle
import torch
import torch.nn as nn
from config import DefaultConfig
from utils import ParameterError
from utils import print_time
from data.dataset_pre import PreDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.word2vec import Word2Vec
from data_pre_analysis.pre_data_modify import data_process_for_word2vec_skipgram


# 首先导入基本参数
opt = DefaultConfig()
torch.manual_seed(opt.seed)

# 预训练数据处理保存和加载
# 首先判断是否需要对原始数据进行处理，或是直接加载处理后的新预训练数据
if opt.is_pre_process:
    if opt.pre_mode == "word2vec":
        # 对原始数据进行处理
        data_process_for_word2vec_skipgram(opt.data_path, opt.w2c_length)
    elif opt.pre_mode == "glove":
        # TODO
        pass
    elif opt.pre_mode == "elmo":
        # TODO
        pass
    elif opt.pre_mode == "openai":
        # TODO
        pass
    elif opt.pre_mode == "bert":
        # TODO
        pass
    else:
        raise ParameterError("参数pre_mode的具体值有问题，请修正！")


# 开始预训练
# 首先判断是否需要预训练
if opt.is_pre_train:
    # 预处理模型初始化
    if opt.pre_mode == "word2vec":
        pre_model = Word2Vec(opt.embedding_size, opt.word_num, opt.use_gpu)
    elif opt.pre_mode == "glove":
        # TODO
        pass
    elif opt.pre_mode == "elmo":
        # TODO
        pass
    elif opt.pre_mode == "openai":
        # TODO
        pass
    elif opt.pre_mode == "bert":
        # TODO
        pass
    else:
        raise ParameterError("参数pre_mode的具体值有问题，请修正！")

    # 读取预训练数据
    predata = PreDataSet(opt.word_num, opt.data_path)
    dataloader = DataLoader(predata,
                            batch_size=opt.embedding_batch,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)

    # 预训练阶段loss和optimizer确认
    pre_loss = nn.BCELoss(size_average=False)
    lr = opt.learning_rate
    optimizer = torch.optim.Adam(pre_model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-4)

    # 训练过程
    start = time.time()
    for epoch in range(1, opt.epoch + 1):
        epoch_start = time.time()
        for ii, (data, label) in enumerate(dataloader):
            t1 = time.time()
            # 神经网络的输入必须要是Variable类型
            input_data = Variable(data)
            input_label = Variable(label)

            # 判断是否需要转变成cuda类型
            if opt.use_gpu:
                input_data = input_data.cuda()
                input_label = input_label.cuda()

            # 预训练过程
            optimizer.zero_grad()
            output_data = pre_model(input_data)
            loss = pre_loss(output_data, input_label)
            loss.backward()

            if (ii + 1) % 100:
                t2 = time.time()
                print("第%d个epoch，第%d个batch的loss是%.4f" % (epoch + 1, ii + 1, loss), end='\t')
                print_time(t1, t2, "该阶段训练")

    end = time.time()
    print_time(start, end)

    # 保存训练完后得到的embedding表
    table_name = "word2vec_" + opt.w2c_mode + "_" + str(opt.w2c_length) + ".pkl"
    path = opt.embedding_table_save_path + table_name
    embedding_table = pre_model.save(path)
    print("预训练结束！")
else:
    table_name = "word2vec_" + opt.w2c_mode + "_" + str(opt.w2c_length) + ".pkl"
    embedding_table_path = opt.embedding_table_save_path + table_name
    embedding_table = pickle.load(embedding_table_path)

# 词向量表已经得到，开始利用CRF或者LSTM+CRF完成任务

