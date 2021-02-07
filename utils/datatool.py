import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
import torchvision
from PIL import Image



# 处理数据和标签，方便写到DataLoader，将数据变化成可迭代类型
class TrainSet(Dataset):
    def __init__(self, data_x, data_y, transform=None):
        # 定义好 image 的路径
        self.data, self.label = data_x, data_y
        self.transform = transform

    def __getitem__(self, index):
        if self.transform != None:
            a = Image.fromarray(np.uint8(self.data[index].reshape(28,28)), 'L')
            b = self.transform(a)
            return b, self.label[index]
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    # 这里的F.to_tensor是将图像处理为像素
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = torch.from_numpy(np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)).long()

    with gzip.open(images_path, 'rb') as imgpath:
        images = (np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(-1, 1, 28, 28))
    return images, labels

# 定义好参与模型的数据
def set_dataloader(x, y , mini_batch, transform_func=None):
        # 32列作为图片特征，
        # print(x.shape, y.shape)
        # 将数据放到TrainSet中处理，组装测试样本和目标值
        if transform_func !=None:
            trainset = TrainSet(x, y, transform=transform_func)
        else:
            trainset = TrainSet(x, y)
        # batch_size=3: 每次迭代的样本为3，shuffle=True:随机打乱， DataLoader将数据变成可迭代的对象
        # drop_last=true, 如果数据集大小不能被batch_size整除，则默认删除最后一个
        trainloader = DataLoader(trainset, batch_size=mini_batch, shuffle=True)
        return trainloader

# 训练中画loss函数模块
def train_valid_plot(x, y1, y2):
    # 一边训练，一边画loss 曲线， 判断是否过拟合
    plt.figure(num=3, figsize=(8, 5))
    plt.style.use('ggplot')
    plt.title('train valid loss plot')
    plt.plot(x, y1, color='red', marker='.', label='train_loss')
    plt.plot(x, y2, color='blue', marker='.', label='valid_loss')
    plt.legend()
    plt.grid()
    plt.show()
    return

# 模型权重参数正则化
def regularization_loss(model, weight_decay, p):
    '''
    :param model: 根据model获取model的参数
    :param weight_decay: 正则化参数权重系数
    :param p: 范数，确定是l1,l2....
    :return:
    '''
    weight_list = []
    # 权重名和权重值 绑在一个tuple，除了bias
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = (name, param)
            weight_list.append(weight)
    total_loss = 0
    for name, param in weight_list:
        reg_loss = torch.norm(param, p)
        total_loss += reg_loss
    total_loss = weight_decay * total_loss
    return total_loss


if __name__=='__main__':
    a, b = load_mnist('../dataset/fashion_mnist', 't10k')
    print(a.shape, b.shape)

