import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义随机数种子
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 定义好参与模型的数据
def dataset(x, y):
        # 32列作为图片特征，
        print(x.shape, y.shape)
        # 将数据放到TrainSet中处理，组装测试样本和目标值
        trainset  = TrainSet(x,y)
        # batch_size=3: 每次迭代的样本为3，shuffle=True:随机打乱， DataLoader将数据变成可迭代的对象
        trainloader = DataLoader(trainset, batch_size=3, shuffle=True, drop_last=True) # drop_last=true, 如果数据集大小不能被batch_size整除，则默认删除最后一个，不然传入全连接层会报错
        return trainloader

# 继承nn.Module并构造初始化方法，forward
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        # convolution 输入1深度，输出5深度，感受野大小为5
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.conv2 = nn.Conv2d(5, 32, 5)
        # 全连接层
        # 线性回归函数,第一层输入，用flatten后的维度，需要根据输入的维度，以及convolution变化后，计算出来，假设我们input的维度是11
        self.fun1 = nn.Linear(800, 500)
        self.fun2 = nn.Linear(500, 50)
        self.fun3 = nn.Linear(50, 1)
        # mae做loss
        self.loss = nn.L1Loss()
        # adam 做梯度下降
        self.optimizer = torch.optim.Adam(TestNet.parameters(self), lr=0.1)

    def forward(self, x):
        # 实现前馈
        # 在convolution后激活，并做max_pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 传入全连接层之前，进行flatten, 切记： 不能将第一个维度batch展开
        x = x.flatten(1,-1).unsqueeze(0)
        print('展开后数据的维度：%s' %x.size())
        # 传入三层全连接网路
        x = F.relu(self.fun1(x))
        x = F.relu(self.fun2(x))
        x = self.fun3(x)
        return x

# 处理数据和标签，方便写到DataLoader
class TrainSet(Dataset):
    def __init__(self, data_x, data_y):
        # 定义好 image 的路径
        self.data, self.label = data_x, data_y

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# 通过nn.Sequential 构建深度网络序列
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # mae做loss
        self.loss = nn.L1Loss()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 5, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.optimizer = torch.optim.Adam(Model2.parameters(self), lr=0.1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # 切记： 不能将第一个维度batch展开
        out = out.flatten(1, -1)
        out = self.fc(out)
        return out

# 调用模型的train, 推荐使用第二种或者第三种模型保存装载方式
def train():
    x = torch.rand(10, 1, 32, 32)
    y = torch.rand(10, 1)
    trainloader = dataset(x, y)
    model1 = Model2()
    for epoch in range(10):
        for i, (x1, y1) in enumerate(trainloader):  # 默认使用minbatch 的方式对数据进行迭代训练
            outputs = model1.forward(x1)
            loss = model1.loss(outputs, y1)
            # 手动清空梯度
            model1.optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # adam 参数更新
            model1.optimizer.step()
            print("迭代次数：[%s/10], Step: [%s/%s], 损失：%s" % (epoch, i, x1.size()[0], loss.item()))

    print('最终loss', loss)
    # 查看模型参数：
    # 方法1：通过model.state_dict(), 如果查看优化器参数，则model1.optimizer.state_dict()
    for param in model1.state_dict():
        print('参数: %s, 大小：%s' %(param, model1.state_dict()[param].size()))

    # 方法2：通过model.parameters(), 但是这种方式不能很直观知道参数对应的参数名称
    for param1 in model1.parameters():
        # print('参数: %s' % (param1.size()))
        print('参数：%s' %(str(param1.size())))

    # 方法3： 通过model.parameters()
    for param in model1.named_parameters():
        print('参数: %s, 大小：%s' %(param[0], param[1].size()))

    # 保存模型方式1：保存整个模型
    torch.save(model1, './model1.pkl')
    # 保存模型方式2：保存模型参数字典, 保存方式与方式1相同，只是加载时不同
    torch.save(model1.state_dict(), './model2.pkl')
    # 保存模型方式3：通过定义字段传入save，字典可传入模型中的各种参数
    torch.save({
        'epoch': 10,
        'model_state_dict': model1.state_dict(),
        'optimizer_state_dict': model1.optimizer.state_dict(),
        'loss': model1.loss
    }, './model3.pkl')
    return outputs


def test(model='model1'):
    data_x = torch.rand(100, 1, 32, 32)
    data_y = torch.rand(100, 1)
    # 如果用DataLoader的方式，则会用每次取minbatch个样本进行预测，这种是为了解决内存或者显存不够时，数据的分批次训练或者预测
    testloader = dataset(data_x, data_y)
    if model == 'model1':
        model = torch.load('./model1.pkl') # 直接装载整个模型
    elif model == 'model2':
        model = Model2() #实例化模型
        model.load_state_dict(torch.load('./model2.pkl'))
    else:
        model = Model2()
        checkpoint = torch.load('./model3.pkl')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.epoch = checkpoint['epoch']
        model.loss = checkpoint['loss']

    # 很重要的一步：调用model.eval() 是将dropout和batch normalization 层设置为测试模式
    model.eval()
    # for param in model.optimizer.state_dict():
        # print('优化器：%s, 值：%s' %(param, model.optimizer.state_dict()[param]))
    for param in model.state_dict():
        print('参数名：%s， 值：%s' %(param, model.state_dict()[param].size()))
    with torch.no_grad():
        data_x = data_x.to(device)
        data_y = data_y.to(device)
        outputs = model(data_x)
        mae = torch.mean(torch.abs(data_y-outputs))
        print('mae: %s' %(mae.data))
    print(model.loss)
    # print(model.epoch)
    return outputs


if __name__=='__main__':
    # seed_torch()
    # data = TestNet().train().to(device)
    # train().to(device)
    # test('model2')
    print(Model2())
