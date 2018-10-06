import pandas as pd
import numpy as np
def load_data(filename,sheet_number):
    """读取训练数据"""
    return pd.read_excel(filename, sheet_name=sheet_number)
     
def change_data_to_input(data, input_size=1,seq_length=24):
    """把数据中转化为(samples_num, input_size, seq_length)维度的数组，每个样本的维度是(input_size,seq_length)"""
    data = np.array(data)
    # print(data)
    data = data.reshape(input_size, data.size//input_size)    # 注意: 如果data的size不整除input_size，那么多出来的那些数据，numpy是如何解决的
    # print(data)
    s = []
    # label = []
    iter_num = ( data.size//input_size ) - seq_length  #// seq_length
    for i in range(iter_num):
        s.append(data[:, i: i+seq_length ])
        # s.append(data[:, i*seq_length : (i+1)*seq_length ])
        # if i<iter_num-1:
        #     label.append(data[:, i*seq_length+1 : (i+1)*seq_length+1 ])
        # elif (i+1)*seq_length+1< data.shape[0]:
        #     label.append(data[:, i*seq_length+1 : (i+1)*seq_length ])
    data = np.array(s)
    return data
    
    
def preprocess_data(data, input_size=1, seq_length=10, month=None):
    """对数据进行预处理"""
    if month is not None:
        data_train = data.loc[data["月份"]==month]
        data_train = data_train.iloc[0:31,3:] # 前几个位置的数据是时间和其他属性
        data_test = data_train.iloc[31:,3:]   
    else:
        pass
    
    data_train = change_data_to_input(data_train, input_size, seq_length) # 对训练数据进行的处理, 把训练的数据处理成输入的序列
    return data_train, data_test

if __name__ == "__main__":
    data_file = "data/air-condition-consumption.xlsx"
    sheet_number=0
    month = 7
    input_size=1
    seq_length=24
    data = load_data(data_file, sheet_number=sheet_number)
    # 对空调的数据进行刷选，仅使用12月份的数据对模型进行训练，同时对数据进行预处理
    data_train, data_test = preprocess_data(data, month=month)

