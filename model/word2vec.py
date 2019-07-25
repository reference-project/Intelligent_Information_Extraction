"""
预训练模型之Word2Vec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DefaultConfig
from data.dataset_pre import PreDataSet
from torch.utils.data import DataLoader


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
        # 保存训练好的embedding表
        torch.save(self.projection.state_dict(), name)


if __name__ == '__main__':
    # 训练word2vec模型
    opt = DefaultConfig()

    # 模型
    w2c = Word2Vec(opt.embedding_size, opt.word_num, opt.use_gpu)

    # 数据
    predata = PreDataSet(opt.word_num, opt.data_path)
    dataloader = DataLoader(predata,
                            batch_size=opt.embedding_batch,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)

    # 优化目标
    criterion = torch.nn.BCELoss(size_average=False)
    lr = opt.learning_rate
    optimizer = torch.optim.Adam(w2c.parameters(),
                                 lr=lr,
                                 weight_decay=)
