import torch
from utils.datatool import load_mnist, set_dataloader, train_valid_plot, regularization_loss
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import arrow
import numpy as np
import gzip
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_func = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        # transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转
        # transforms.RandomRotation((30, 70), resample=False, expand=False, center=None), #随机旋转角度
        # transforms.RandomCrop(size=16, padding=6), # 图片重新裁剪转化
        transforms.ToTensor()
    ])


def load_data(cv=True):
    train_x, train_y = load_mnist('../dataset/fashion_mnist', kind='train')
    test_x, test_y = load_mnist('../dataset/fashion_mnist', kind='t10k')
    test_x = torch.from_numpy(test_x).float().div(255)
    test_data = set_dataloader(test_x, test_y, 1000, transform_func=None)
    if cv == True:
        X_train, X_valid, y_train, y_valid = train_test_split(np.array(train_x), np.array(train_y), test_size=0.1,
                                                              stratify=train_y, random_state=1)
        train_data = set_dataloader(X_train, y_train, 800, transform_func=transform_func)
        return train_data, test_data, torch.tensor(X_valid).float().div(255), torch.tensor(y_valid).long()
    train_data = set_dataloader(train_x, train_y, 1000)
    return train_data, test_data


# def test():
#     transform = transforms.Compose(
#         [
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomGrayscale(),
#             transforms.ToTensor()])
#
#     transform1 = transforms.Compose(
#         [
#             transforms.ToTensor()])
#     trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
#                                                  download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000,
#                                               shuffle=True, num_workers=2)
#     for i, (images,label) in enumerate(trainloader):
#         print(images.size())
#         print(label.size())
#     return trainloader


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 60, 5, padding=2),
            # 随机丢弃50%神经元
            # nn.Dropout2d(0.5),
            # 对channel做归一化处理
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(60, 240, 5, padding=2),
            # nn.Dropout2d(0.6),
            # nn.BatchNorm2d(240),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(11760, 640),
            # nn.Dropout2d(0.6),
            nn.ReLU(),
            nn.Linear(640, 10)
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output.flatten(1, -1)
        output = self.fc(output)
        return output


class CnnModel(CnnNet):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.epoch = 24

    def model_train(self, inputdata, path, lr=0.03, X_valid=None, y_valid=None, cv=True):
        loss = nn.CrossEntropyLoss().to(device)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.8)
        epoch_ls = []
        train_loss_ls = []
        cv_loss_ls = []
        for epoch in range(self.epoch):
            timestart = arrow.utcnow().timestamp
            train_total = 0
            train_correct = 0
            train_loss_cal = 0
            cv_correct = 0
            cv_loss_cal = 0
            cv_total = 0
            for raw, (images, labels) in enumerate(inputdata):
                X_train, y_train = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(X_train)
                l = loss(outputs, y_train)
                # l = loss(outputs, y_train) + regularization_loss(self, 0.01, 2).to(device)
                l.backward()
                optimizer.step()
                # 测试机结果输出
                train_loss_cal += l.item()
                outputs = torch.max(outputs, 1)[1]
                train_correct += (outputs == y_train).sum().item()
                train_total += len(y_train)
                if cv == True:
                # 验证集结果输出
                    self.eval() # 让模型变成预测模式
                    X_valid = X_valid.to(device)
                    y_valid = y_valid.to(device)
                    cv_outputs = self(X_valid)
                    cv_loss = loss(cv_outputs, y_valid)
                    # cv_loss = loss(cv_outputs, y_valid) + regularization_loss(self, 0.03, 2).to(device)
                    cv_outputs = torch.max(cv_outputs, 1)[1]
                    cv_loss_cal += cv_loss.item()
                    cv_correct += (cv_outputs == y_valid).sum().item()
                    cv_total += len(y_valid)
                    self.train() # 让模型重新回到训练模式
            train_currary = train_correct / train_total
            print("[train: epoch: %s, loss: %.4f, train accuracy：%.4f, coss time %3f sec]" %(epoch, train_loss_cal/50,
                                                                                      train_currary,
                                                                                    arrow.get().timestamp - timestart,
                                                                            ))
            if cv==True:
                cv_currary = cv_correct / cv_total
                print("[--valid: epoch: %s, loss: %.4f, valid accuracy：%.4f, coss time %3f sec]" %(epoch, cv_loss_cal/10,
                                                                                      cv_currary,
                                                                                    arrow.get().timestamp - timestart,
                                                                            ))
                # 确定好x轴和y轴的数据，x轴和y轴的数据是不断相加的
                train_loss_ls.append(train_loss_cal/50)
                cv_loss_ls.append(cv_loss_cal/10)
                epoch_ls.append(epoch)
                # 画loss曲线
                if (epoch % 5 == 4) or (epoch == (self.epoch - 1)):
                    train_valid_plot(epoch_ls, train_loss_ls, cv_loss_ls)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path)
        print('Finish Training')

    def model_test(self, inputdata, model_file):
        model = CnnModel()
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint['loss']
        #  查看模型参数
        for param in model.state_dict():
            print('参数: %s, 大小：%s' % (param, model.state_dict()[param].size()))
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for (images, labels) in inputdata:
                outputs = model(images)
                l = loss(outputs, labels)
                total_loss += l.item()
                predict_label = torch.max(outputs, 1)[1]
                correct_num = (labels == predict_label).sum().item()
                correct += correct_num
                total += labels.size()[0]
            print('测试集的loss: %.4f, 预测准确率：%.4f' %(total_loss/10, correct/total))


if __name__ == '__main__':
    train_data, test_data, x_valid, y_valid = load_data(cv=True)
    # 训练阶段
    # model = CnnModel().to(device).model_train(train_data, './model_file/model8-2.pkl', X_valid=x_valid, y_valid=y_valid, cv=True)
    # 推理阶段
    CnnModel().to(device).model_test(test_data, './model_file/model8-2.pkl')


