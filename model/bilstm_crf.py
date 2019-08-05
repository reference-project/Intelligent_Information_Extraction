"""
实现词性标注中使用最多的BiLSTM CRF模型，在pytorch官网学习板块找到的
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, word_embeds):
        super(BiLSTM_CRF, self).__init__()
        # 参数
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix  # 此项目中这个参数不需要，因为我在生成训练数据时已经将tag进行了转换
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = word_embeds  # word embedding表

        # 模型
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
