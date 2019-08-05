"""
此文件放置一些辅助函数
"""
import torch


# 类
class ParameterError(Exception):
    def __init__(self, error_info):
        super(ParameterError, self).__init__()
        self.error_info = error_info

    def __str__(self):
        return self.error_info


# 函数
def print_time(start_time, end_time, des="整个过程"):
    spend_time = end_time - start_time
    if spend_time < 60:
        print(des + "耗时 %.2f 秒" % spend_time)
    elif 60 <= spend_time < 3600:
        print(des + "耗时 %.2f 分钟" % (spend_time / 60))
    elif spend_time >= 3600:
        print(des + "耗时 %.2f 小时" % (spend_time / 3600))


def argmax(vec):
    # 返回向量vec中最大值对应的序号
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    # 将词性和对应编号进行相互转换
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtpye=torch.long)


def log_sum_exp(vec):
    # 注：这是pytorch学习教程上的代码
    # TODO:还不太清楚这个函数的作用，待补充
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.long(torch.sum(torch.exp(vec - max_score_broadcast)))
