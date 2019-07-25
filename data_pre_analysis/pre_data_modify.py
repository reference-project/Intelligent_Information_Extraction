"""
对原始数据进行处理，得到预训练过程所需要的格式数据
pre_train_data_path = 'D:\\Data\\datagrand\\corpus.txt'
"""


# 对预训练集进行处理得到可用于word2vec训练的
def data_process_for_word2vec_skipgram(path, length):
    pre_data = open(path + 'pre_data_for_word2vec_skipgram_' + str(length) + '.txt', 'w')
    pre_train_data = open(path + 'pre_train_data_for_word2vec_skipgram_' + str(length) + '.txt', 'w')
    pre_test_data = open(path + 'pre_test_data_for_word2vec_skipgram_' + str(length) + '.txt', 'w')

    # 处理corpus.txt文件得到pre_data.txt文件
    with open(path + 'corpus.txt', 'r') as pre_train:
        for ii, line in enumerate(pre_train):
            try:
                words = line.strip('\n').split('_')
                for i in range(len(words)):
                    label = []
                    for j in range(1, length // 2 + 1):
                        if i - j >= 0:
                            label.append(words[i - j])
                        if i + j < len(words):
                            label.append(words[i + j])
                    label = '_'.join(label)
                    pre_data.write(words[i])
                    pre_data.write('\t')
                    pre_data.write(label)
                    pre_data.write('\n')
                if (ii + 1) % 1000 == 0:
                    print(ii + 1, words[i], label)
            except ValueError:
                continue
        print('corpus.txt处理完成%d行！' % (ii + 1))

    # 处理train.txt和test.txt文件得到pre_train_data.txt文件和pre_test_data.txt文件
    with open(path + 'train.txt', 'r') as train:
        for ii, line in enumerate(train):
            try:
                words = line.strip('\n').replace(' ', '') \
                           .replace('/a', '_').replace('/b', '_').replace('/c', '_').replace('/o', '_')[:-1] \
                           .split('_')
                for i in range(length // 2, len(words)):
                    label = []
                    for j in range(1, length // 2 + 1):
                        if i - j >= 0:
                            label.append(words[i - j])
                        if i + j < len(words):
                            label.append(words[i + j])
                    label = '_'.join(label)
                    pre_train_data.write(words[i])
                    pre_train_data.write('\t')
                    pre_train_data.write(label)
                    pre_train_data.write('\n')
                if (ii + 1) % 100 == 0:
                    print(ii + 1, words[i], label)
            except ValueError:
                continue
        print('train.txt处理完成%d行！' % (ii + 1))

    with open(path + 'test.txt', 'r') as test:
        for ii, line in enumerate(test):
            try:
                words = line.strip('\n').split('_')
                for i in range(length // 2, len(words)):
                    label = []
                    for j in range(1, length // 2 + 1):
                        if i - j >= 0:
                            label.append(words[i - j])
                        if i + j < len(words):
                            label.append(words[i + j])
                    label = '_'.join(label)
                    pre_test_data.write(str(words[i]))
                    pre_test_data.write('\t')
                    pre_test_data.write(label)
                    pre_test_data.write('\n')
                if (ii + 1) % 10 == 0:
                    print(ii + 1, words[i], label)
            except ValueError:
                continue
        print('test.txt处理完成%d行！' % (ii + 1))

    pre_data.close()
    pre_train_data.close()
    pre_test_data.close()


if __name__ == '__main__':
    data_process_for_word2vec_skipgram('D:\\Data\\datagrand\\', 5)
