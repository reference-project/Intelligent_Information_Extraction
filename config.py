"""
配置文件
"""
import warnings


class DefaultConfig:
    # 训练基本信息
    use_gpu = False  # 是否使用GPU训练模型
    learning_rate = 0.001  # 学习率
    epoch = 20  # 训练总次数
    pre_mode = "word2vec"  # 预训练类型，包括word2vec、glove、elmo、openai和bert

    # 数据基本信息
    word_num = 21225
    data_path = 'D:\\Data\\datagrand\\'

    # word embedding阶段需要的超参数
    is_pre_process = False
    is_pre_train = False
    embedding_epoch = 20  # 预训练阶段epoch数量
    embedding_size = 128  # 单词对应embedding向量的长度
    embedding_batch = 64  # 预训练阶段的batch
    embedding_table_save_path = ".\\embedding_table\\"
    # word2vec相关参数
    w2c_mode = "skipgram"  # 训练方式，包括skipgram和cbow
    w2c_length = 5  # 上下文长度（包含中间词）

    def parse(self, kwargs):
        """
        根据命令行输入kwargs修改超参数
        """
        # 更新配置参数
        for k, v in kwargs:
            if not hasattr(self, k):
                warnings.warn("warning: opt has not attribute %s" % k)

        # 打印配置参数
        print("Config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
