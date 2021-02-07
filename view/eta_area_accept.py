import torch
from utils.datatool import load_mnist, set_dataloader, train_valid_plot, regularization_loss
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import arrow
import numpy as np
from sklearn.model_selection import train_test_split
from pickle import dump, load, PicklingError, UnpicklingError, HIGHEST_PROTOCOL
from sklearn.externals import joblib


def load_pickle(filepath, feature, target, using_joblib=False):
    filepath = filepath
    try:
        if using_joblib:
            return joblib.load(filepath)
        else:
            with open(filepath, 'rb') as file:
                try:
                    data = load(file)
                    return data.loc[:, feature], data.loc[:, target]
                except UnicodeDecodeError:
                    import hashlib
                    return hashlib.sha1(file.read()).hexdigest()
    except (IOError, EOFError):  # File Not Found
        print('file_not_found: {}'.format(filepath))
        return None
    except UnpicklingError:
        return None

def data_loader(x, y, cv=False, transform_func=None):
    x = x.fillna({'distribute_sort':1})
    x['distribute_waiting_num'] = x['distribute_sort']- x['schedule_free_num']
    # 分类数据做embedding
    embedding_feature = ['area_id', 'weekday', 'hour', 'schedule_free_num', 'delivery_num', 'pick_up_num',
                         'distribute_order_num', 'shortag_tasker','extra_require_biker', 'distribute_waiting_num',
                         'distribute_sort',"user_distribute_order_num", "user_pickup_tasker_num", "user_brand_non_schedule_free_tasker_num",
                "user_no_brand_non_schedule_free_tasker_num", "user_shortage_tasker_num"]

    for i in embedding_feature:
        len_uniqu = len(x[i].unique())
        max_uniqu = max(x[i].unique())
        embedding_max = torch.nn.Embedding(max_uniqu+1, len_uniqu)

    if cv == True:
        X_train, X_valid, y_train, y_valid = train_test_split(np.array(x), np.array(y), test_size=0.1,
                                                              stratify=y, random_state=1)
        data = set_dataloader(X_train, y_train, 10000, transform_func=transform_func)
        return data,  X_valid, y_valid
    data = set_dataloader(x, y, 10000, transform_func=transform_func)
    return data


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()

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


if __name__=='__main__':
    # 剔除 push_time, delay_time
    features = sorted(['area_id', 'weekday', 'hour', 'schedule_free_num', 'delivery_num', 'pick_up_num',
                'order_backlog', 'scaling_factor', 'distribute_order_num', 'shortag_tasker',
                'busy_divide_free', 'busy_divide_free_delivery', 'busy_divide_free_delivery_pickup',
                'extra_require_biker', 'distribute_waiting_num', 'distribute_sort', "wrainc", "wtcsgnlc",
                "user_distribute_order_num", "user_pickup_tasker_num", "user_brand_non_schedule_free_tasker_num",
                "user_no_brand_non_schedule_free_tasker_num", "user_shortage_tasker_num"
                ])
    x, y = load_pickle('../dataset/eta/local-hk-filter_data_accept.pkl', features, 'accept')

    print(x.columns.tolist())
    print(x.describe().T)








