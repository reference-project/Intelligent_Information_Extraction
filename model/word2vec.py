"""
预训练模型之Word2Vec
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DefaultConfig
from data.dataset_pre import PreDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import print_time


class Word2Vec(nn.Module):
    def __init__(self, embedding_size, word_sum, use_gpu):
        super(Word2Vec, self).__init__()
        self.use_gpu = use_gpu
        self.projection = nn.Linear(word_sum, embedding_size)
        self.output = nn.Linear(embedding_size, word_sum)

    def forward(self, x):
        if self.use_gpu:
            x = x.cuda()
        x = self.projection(x)
        x = self.output(x)
        x = F.sigmoid(x)  # 返回的是一个概率向量
        return x

    def save(self, name='word2vec_embedding_table.pkl'):
        # 保存并返回训练好的embedding表
        torch.save(self.projection.state_dict(), name)
        return self.projection.parameters()


if __name__ == '__main__':
    # 训练word2vec模型
    # 导入超参数
    opt = DefaultConfig()

    # 模型初始化
    w2c = Word2Vec(opt.embedding_size, opt.word_num, opt.use_gpu)

    # 数据加载
    predata = PreDataSet(opt.word_num, opt.data_path)
    dataloader = DataLoader(predata,
                            batch_size=opt.embedding_batch,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)

    # loss和optimizer确认
    criterion = nn.BCELoss(size_average=False)
    lr = opt.learning_rate
    optimizer = torch.optim.Adam(w2c.parameters(),
                                 lr=lr,
                                 weight_decay=1e-4)  # weight_decay是放在正则项前面的系数

    # 开始预训练
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
            output_data = w2c(input_data)
            loss = criterion(output_data, input_label)
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
    w2c.save(path)

    print("预训练结束！")
