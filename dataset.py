import pandas as pd
import numpy as np
import torch
import torch.data as data
import torch.transform
def transforms():
    return 

def excel_loader(filename,sheet_number):
    """读取训练数据"""
    data = pd.read_excel(filename, sheet_name=sheet_number)
    return data.values[:,3:]

class PowerDataset(data.Dataset):
    def __init__(self, opt, loader=excel_loader, transform=None):
        super(PowerDataset, self).__init__()
        self.opt = opt
        self.loader = loader
        self.data = self.loader(opt.filename, opt.sheet_number).values
    
    def __getitem__(self, index):
        input, target = self.get_batch(self.data, index)
        return input, target

    def __len__(self):
        return len(self.data.size - self.opt.seq_length)
    
    def get_batch(self, data, index):
        """对数据进行分批取样操作"""
        # 随机对data里面的样本进行抽取，生成随机抽取样本的索引
        # x_index = np.random.randint(low=0, high=data.shape[0]-1, size=self.batch_size)
        batch_start = np.random.randint(low=0, high=data.shape[0] * data.shape[1] - 1 - (1 + self.opt.seq_length)) # 随机取样的横轴坐标 (1 + self.opt.seq_length)是训练样本加label的长度, (data.shape[0] * data.shape[1] - 1)是总数据的矩阵最大下标 0 ~ max
        x_start = batch_start % data.shape[1] 
        y_start = batch_start % data.shape[0]
        tmp = x_start + self.opt.seq_length - 1 
        x_end =  tmp % data.shape[1] # 第一列是0点,所以对24取模
        y_end = y_start + 1 if tmp >= data.shape[1] else y_start
        # 在所有样本中取出input sequence, 从data[y_start, x_start]一直取到 data[y_end, x_end], target则为 data[y_end, x_end+1]
        if tmp < data.shape[1]:
            if x_end+1 < data.shape[1]:
                input = data[y_start, x_start:x_end+1]  
                target = data[y_end, x_end+1]
            else:
                input = data[y_start, x_start:]
                target = data[y_end+1, 0]
        else:
            input = []
            input.append(list(data[y_start, x_start:]))
            input.append(list(data[y_end, 0:x_end+1]))
            np.array(input)
            target = data[y_end, x_end+1]
        return input, target