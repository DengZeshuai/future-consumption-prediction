import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, opt, batch_first=True, dropout=0.5):
        super(LSTM, self).__init__()
        self.opt = opt
        self.input_size = opt.input_size
        self.hidden_dim = opt.hidden_dim
        self.output_size = opt.output_size
        self.num_layers = opt.num_layers
        self.seq_length = opt.seq_length
        self.batch_first = batch_first # 是否输入输出的第一维为batchsize True or False
        self.dropout = dropout # 
        
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers, batch_first=batch_first, dropout=dropout)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_size)
        self.fc2 = nn.Linear(self.seq_length, self.output_size)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.opt.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.opt.batch_size, self.hidden_dim))
    
    def forward(self, sequence):
        # print('sequence.size()', sequence.size())
        
        lstm_out, self.hidden = self.lstm(sequence, self.hidden)
        # print(lstm_out.size())
        output = self.fc1(lstm_out)
        # # print(output.size())
        # output = self.fc2(output.view(output.size(0), 1, -1))
        # print(output.size())
        return output[:,-1,:]