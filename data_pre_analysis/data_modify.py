"""
对原始数据进行处理，得到训练过程和测试过程所需要的格式数据
train_data_path = 'D:\\Data\\datagrand\\train.txt'
test_data_path = 'D:\\Data\\datagrand\\test.txt'
"""


labels = {'o': '0', 'a': '1', 'b': '2', 'c': '3'}


# 对训练集进行处理得到训练过程需要的新格式的数据集
def train_process(path):
    train_data = open(path + 'train_data.txt', 'w')
    train_label = open(path + 'train_label.txt', 'w')
    with open(path + 'train.txt', 'r') as train:
        for line in train:
            line = line.strip('\n')
            for part in line.split('  '):
                words, flag = part.split('/')
                for word in words.split('_'):
                    train_data.write(word)
                    train_data.write('\t')
                    train_label.write(labels[flag])
                    train_label.write('\t')
            train_data.write('\n')
            train_label.write('\n')
    train_data.close()
    train_label.close()


# 对测试集进行处理得到测试过程需要的新格式的数据集
def test_process(path):
    test_data = open(path + 'test_data.txt', 'w')
    with open(path + 'test.txt', 'r') as test:
        for line in test:
            for word in line.split('_'):
                test_data.write(word)
                test_data.write('\t')
            test_data.write('\n')
    test_data.close()


if __name__ == '__main__':
    train_process('D:\\Data\\datagrand\\')
    test_process('D:\\Data\\datagrand\\')
