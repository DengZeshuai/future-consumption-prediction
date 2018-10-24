import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, opt, batch_first=True, dropout=False):
        super(LSTM, self).__init__()
        self.opt = opt
        self.input_size = opt.input_size
        self.hidden_dim = opt.hidden_dim
        self.output_size = opt.output_size
        self.num_layers = opt.num_layers
        self.seq_length = opt.seq_length
        self.batch_first = batch_first # 是否输入输出的第一维为batchsize True or False
        self.dropout = dropout # 
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, self.num_layers, batch_first=batch_first, dropout=dropout)
        self.fc = nn.Linear(self.hidden_dim, self.output_size) 
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))
    
    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence.view, self.hidden)
        output = self.fc(lstm_out.view(self.seq_length, -1))
        return output