import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
import torch
import torch.utils.data as data
import sys
import os


def excel_loader(filename, sheet_number=0):
    """读取训练数据"""
    data = pd.read_excel(filename, sheet_name=sheet_number)
    data['时间'] = data['时间'].dt.dayofyear
    return data.values

def remove_nan(data):
    if np.isnan(data).sum() > 0:
        nan_matrix = np.isnan(data)
        nan_index = np.where(nan_matrix == True)
        print(nan_index)
        for x, y in zip(nan_index[0], nan_index[1]):
            print(x,y)
            if data[x+1, y] + data[x-1, y] != np.nan:
                print("two is not nan")
                data[x,y] = data[x+1, y] + data[x-1, y]
            elif data[x-1,y]+ data[x-2, y] != np.nan:
                print("next is nan")
                data[x,y] = data[x-1, y] + data[x-2, y]
            else:
                print('not change')
    return data

class PowerDataset(data.Dataset):
    def __init__(self, opt, loader=excel_loader, transform=None, prepocess_path='data/a.csv'):
        super(PowerDataset, self).__init__()
        self.opt = opt
        self.loader = loader
        self.prepocess_path = prepocess_path
        
        if os.path.exists(prepocess_path):
            self.data = pd.read_csv(prepocess_path).values[:,1:]
            # print(self.data)
        else:
            self.data = self.loader(opt.filename, opt.sheet_number)
            # pd.DataFrame(self.data).to_csv('data.csv')
            if np.isnan(self.data).sum() > 0:
                self.data = remove_nan(self.data)
            self.data = self.pre_process(self.data)
        
        self.len = len(self.data) - self.opt.seq_length - 1
        self.transform = transform
    
    def __getitem__(self, index):
        input = self.data[index:index+self.opt.seq_length, :]
        target = self.data[index+self.opt.seq_length+1, -1]
        if self.transform is not None:
            # print(input.shape, target.shape)
            # input = input.reshape(input.shape[0], input.shape[1], 1)
            # target = target.reshape(1, 1, 1)
            # input, target = self.transform(input), self.transform(target)
            input, target = torch.FloatTensor(input),  torch.FloatTensor(target.reshape(1))
            
            # print(input.size(), target.size())
        return input, target

    def __len__(self):
        return  self.len
    
    def pre_process(self, data):
        #天数，小时，是否供冷季节，是否工作日，电耗
        a = np.zeros((len(data)*24,5))
        time_hours = [i for i in range(24)]
        for i in range(0,len(data)):
            a_index = i*24
            a[a_index:a_index+24, 0] = data[i,0]
            a[a_index:a_index+24, 1] = time_hours
            a[a_index:a_index+24, 2] = data[i,-2]
            a[a_index:a_index+24, 3] = data[i,-1] 
            a[a_index:a_index+24, 4] = data[i, 1:-2]
        # minmax_scaler = MinMaxScaler()
        a_scale = normalize(a[:, -1].reshape(1, -1), norm='l1')
        a[:, -1] = a_scale.reshape(-1)
        if not os.path.exists(self.prepocess_path):
            pd.DataFrame(a).to_csv(self.prepocess_path)
        return a