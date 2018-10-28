import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import sys

def transform(self, data):
    self.data.to_csv('a.csv', index=False,header=False, float_format='%.3f')


def excel_loader(filename,sheet_number):
    """读取训练数据"""
    # data = pd.read_excel(filename, sheet_name=0) #'air-condition-consumption2.xlsx'
    # print(data.values[:,:])
    data = pd.read_excel(filename, sheet_name=sheet_number)
    data['时间'] = data['时间'].dt.dayofyear
    return data.values

class PowerDataset(data.Dataset):
    def __init__(self, opt, loader=excel_loader, transform=None):
        super(PowerDataset, self).__init__()
        self.opt = opt
        self.loader = loader
        self.data = self.loader(opt.filename, opt.sheet_number)
        # pd.DataFrame(self.data).to_csv('data.csv')
        self.data = self.pre_process(self.data)
        self.transform = transform
    
    def __getitem__(self, index):
        input = self.data[index:index+self.opt.seq_length, :]
        target = self.data[index+self.opt.seq_length+1, -1]
        if self.transform is not None:
            # print(input.shape, target.shape)
            # input = input.reshape(input.shape[0], input.shape[1], 1)
            # target = target.reshape(1, 1, 1)
            input, target = torch.Tensor(input), torch.Tensor(target.reshape(1, 1))
            # print(input.size(), target.size())
        return input, target

    def __len__(self):
        return self.data.shape[0] - self.opt.seq_length - 1
    
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
        # pd.DataFrame(a).to_csv('a.csv')
        return a