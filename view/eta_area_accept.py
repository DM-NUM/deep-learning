import torch
from utils.datatool import load_mnist, set_dataloader, train_valid_plot, regularization_loss
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import arrow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pickle import dump, load, PicklingError, UnpicklingError, HIGHEST_PROTOCOL
# from sklearn.externals import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pickle(filepath, using_joblib=False):
    filepath = filepath
    try:
        if using_joblib:
            return joblib.load(filepath)
        else:
            with open(filepath, 'rb') as file:
                try:
                    data = load(file)
                    return data
                except UnicodeDecodeError:
                    import hashlib
                    return hashlib.sha1(file.read()).hexdigest()
    except (IOError, EOFError):  # File Not Found
        print('file_not_found: {}'.format(filepath))
        return None
    except UnpicklingError:
        return None

def data_loader(data, cv=False, transform_func=None):
    # features = sorted(['area_id', 'weekday', 'hour', 'schedule_free_num', 'delivery_num', 'pick_up_num',
    #                    'order_backlog', 'scaling_factor', 'distribute_order_num', 'shortag_tasker',
    #                    'busy_divide_free', 'busy_divide_free_delivery', 'busy_divide_free_delivery_pickup',
    #                    'extra_require_biker', 'distribute_waiting_num', 'distribute_sort', "wrainc", "wtcsgnlc",
    #                    "user_distribute_order_num", "user_pickup_tasker_num", "user_brand_non_schedule_free_tasker_num",
    #                    "user_no_brand_non_schedule_free_tasker_num", "user_shortage_tasker_num"
    #                    ])
    # data = data.fillna({'distribute_sort': 1})
    # data['distribute_waiting_num'] = data['distribute_sort'] - data['schedule_free_num']
    # # 分类数据做embedding
    # embedding_feature = sorted([
    #     'area_id', 'weekday', 'hour', 'schedule_free_num', 'delivery_num', 'pick_up_num',
    #     'distribute_order_num', 'shortag_tasker','extra_require_biker', 'distribute_waiting_num',
    #     'distribute_sort',"user_distribute_order_num", "user_pickup_tasker_num",
    #     "user_brand_non_schedule_free_tasker_num","user_no_brand_non_schedule_free_tasker_num",
    #     "user_shortage_tasker_num"])
    #
    # no_embedding_feature = sorted(set(features) - set(embedding_feature))
    # data_continus = torch.from_numpy(np.array(data[no_embedding_feature])).float()
    # for i in embedding_feature:
    #     len_uniqu = len(data[i].unique())
    #     max_uniqu = max(data[i].unique())
    #     embedding_max = torch.nn.Embedding(max_uniqu+1, len_uniqu)
    #     data_continus = torch.cat([data_continus, embedding_max], 1)
    # print('============', data_continus.shape)
    # target = torch.from_numpy(np.array(data['accept']))

    features = sorted(['area_id', 'weekday', 'hour', 'schedule_free_num', 'delivery_num', 'pick_up_num',
                       'order_backlog', 'scaling_factor', 'distribute_order_num', 'shortag_tasker',
                       'busy_divide_free', 'busy_divide_free_delivery', 'busy_divide_free_delivery_pickup',
                       'extra_require_biker', 'distribute_waiting_num', 'distribute_sort', "wrainc", "wtcsgnlc",
                       "user_distribute_order_num", "user_pickup_tasker_num", "user_brand_non_schedule_free_tasker_num",
                       "user_no_brand_non_schedule_free_tasker_num", "user_shortage_tasker_num"
                       ])
    data = data.fillna({'distribute_sort': 1})
    data['distribute_waiting_num'] = data['distribute_sort'] - data['schedule_free_num']
    # 分类数据做embedding
    embedding_feature = sorted([
        'area_id', 'weekday', 'hour', 'schedule_free_num', 'delivery_num', 'pick_up_num',
        'distribute_order_num', 'shortag_tasker', 'extra_require_biker', 'distribute_waiting_num',
        'distribute_sort', "user_distribute_order_num", "user_pickup_tasker_num",
        "user_brand_non_schedule_free_tasker_num", "user_no_brand_non_schedule_free_tasker_num",
        "user_shortage_tasker_num"])

    no_embedding_feature = sorted(set(features) - set(embedding_feature))
    # data_continus = torch.from_numpy(np.array(data[no_embedding_feature])).float()
    concat_data = torch.from_numpy(np.array(data[no_embedding_feature+embedding_feature]))
    # for i in embedding_feature:
    #     len_uniqu = len(data[i].unique())
    #     max_uniqu = max(data[i].unique())
    #     embedding_max = torch.nn.Embedding(max_uniqu + 1, len_uniqu)
    #     data_continus = torch.cat([data_continus, embedding_max], 1)
    # print('============', data_continus.shape)
    target = torch.from_numpy(np.array(data['accept']))

    if cv == True:
        X_train, X_valid, y_train, y_valid = train_test_split(np.array(concat_data), np.array(target), test_size=0.1,
                                                              stratify=target, random_state=1)
        data = set_dataloader(X_train, y_train, 10000, transform_func=transform_func)
        return data,  X_valid, y_valid
    data = set_dataloader(concat_data, target, 10000, transform_func=transform_func)
    return data


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(11760, 640),
            # nn.Dropout2d(0.6),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear(640, 10)
        )

    def forward(self, x):
        # 分类数据做embedding
        # 前7列的数据是连续数据
        data_continus = x[:, :7]
        for i in range(7, x.size().item()):
            len_uniqu = len(data[:, i].unique().size().item())
            max_uniqu = max(data[:, i].unique().size().item())
            embedding_max = nn.Embedding(max_uniqu + 1, len_uniqu)
            data_continus = torch.cat([data_continus, embedding_max], 1)
        print('============', data_continus.shape)
        output = data_continus.flatten(1, -1)
        output = self.fc(output)
        return output

class CnnModel(CnnNet):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.epoch = 24

    def model_train(self, inputdata, path, lr=0.0001, X_valid=None, y_valid=None, cv=True):
        loss = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.8)
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
            print('测试集的loss: %.4f, 预测准确率：%.4f' % (total_loss/10, correct/total))


if __name__=='__main__':
    # 特征剔除 push_time, delay_time
    # x, y = load_pickle('../dataset/eta/local-hk-filter_data_accept.pkl')
    data = pd.read_csv('../dataset/eta/release-hk-filter-data.csv')
    print(len(data), data.time.min(), data.time.max())










