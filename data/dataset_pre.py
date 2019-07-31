"""
预训练过程数据输入函数
"""
import os
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from utils import ParameterError


class PreDataSet(data.Dataset):
    def __init__(self, word_sum, data_path):
        super(PreDataSet, self).__init__()

        self.sum = word_sum
        self.path = data_path
        pre_data = []
        for file in os.listdir(self.path):
            if 'pre' not in file:
                continue
            path = os.path.join(self.path, file)
            with open(path, 'r') as f:
                for line in f:
                    pre_data.append(line.split('\t'))
        if not self.pre_data:
            raise ParameterError("data_path中没有用于预处理的文件！")
        self.pre_data = np.array(pre_data)
        self.len = self.pre_data.shape[0]

    def __getitem__(self, index):
        train_data = [0] * self.sum
        train_data[int(self.pre_data[index][0])] = 1
        label = [0] * self.sum
        for i in [int(k) for k in self.pre_data[index][1].split('_')]:
            label[i] = 1
        return np.array(train_data), np.array(label)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    testSet = PreDataSet(10, 'D:\\Data\\datagrand\\debug_file\\')
    dataloader = DataLoader(testSet,
                            batch_size=4,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)
    for ii, (x, y) in enumerate(dataloader):
        print(x)
        print(y)
        if ii == 0:
            break
