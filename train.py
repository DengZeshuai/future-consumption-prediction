import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import LSTM
from dataset import PowerDataset
from loss import L2loss 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="parse the parameter")
parser.add_argument('--epochs', type=int, default=10, help='the number of train epoch, each epoch will scan the data totally')
parser.add_argument('--train', type=bool, default=True, help="train or test")
parser.add_argument('--month', type=int, default=7, help="the power consumption used to train or test")
parser.add_argument('--sheet_number', type=int, default=0, help="the nubmer of sheet in excel file")
parser.add_argument('--filename', default='data/air-condition-consumption2.xlsx', help="the file name of data")
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='mini-batch size')
parser.add_argument('--n_threads', type=int, default=4, help='the thraeds number fo dataloader')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--log_path', default='logs/', help='log path')
parser.add_argument('--input_size', type=int, default=5, help='input size of the data from model')
parser.add_argument('--seq_length', type=int, default=24, help='the sequence length of input, less than 24')
parser.add_argument('--output_size', type=int, default=1, help='output size of the data from model')
parser.add_argument('--hidden_dim', type=int, default=5, help='the dimentional of hidden_dim in lstm')
parser.add_argument('--num_layers', type=int, default=5, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='')
parser.add_argument('--print_every', type=int, default=100, help='')
parser.add_argument('--pre_train', action='store_true', help='')
parser.add_argument('--save_path', type=str, default='model.pkl', help='')
parser.add_argument('--test_only', action='store_true', help='')
parser.add_argument('--prepocess_path', default='data/a_scale.csv', help='')
opt = parser.parse_args()
# print(opt)
torch.manual_seed(1)

def plot_result(output_list, target_list):
    # plt.ion()
    plt.plot(target_list, color='blue', label='ground truth')
    plt.plot(output_list, color='orange', label='prediction')
    plt.show()
    # plt.draw()
    # plt.pause(0.3)


def test(model, dataloader):
    output_list = []
    target_list = []
    for _, (inputs, target) in enumerate(dataloader):
        if len(inputs) < opt.batch_size:
            continue
        outputs = model(inputs.detach())
        output_list += list(outputs.data.numpy())
        target_list += list(target.data.numpy())
    plot_result(output_list, target_list)

def train(model, loss_function, optimizer, dataloader):
    for iter, (inputs, target) in enumerate(dataloader, 1):
        optimizer.zero_grad()
        # print(iter, "input size: {} target: {}".format(inputs.size(), target.size()))
        if len(inputs) < opt.batch_size:
            continue
        output = model(inputs)
        # print(iter, "output: {}".format(output.size()))
        # output, target = output.unsqueeze(3), target.unsqueeze(3)
        # print(iter, "output size: {} target: {}".format(output.size(), target.size()))
        # loss = loss(output.view(output.size(0), 1, output.size(1), output.size(2)), target.type(torch.FloatTensor))
        
        loss = loss_function(output, target)
        # for param in model.parameters():
        #     loss += param.data.abs().sum()
        loss.backward(retain_graph=True)
        
        optimizer.step()
        
        if iter % 10 == 0:
            print('loss: {}'.format(loss))

        
def main(opt):
    model = LSTM(opt, batch_first=True, dropout=opt.dropout)
    if opt.pre_train:
        model.load_state_dict(torch.load(opt.save_path))
    optimizer = optim.Adam(model.parameters(), opt.learning_rate)
    mseloss = nn.MSELoss()
    
    dataset = PowerDataset(opt, prepocess_path=opt.prepocess_path, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    train_dataset = data.Subset(dataset, indices=range(8664))
    test_dataset = data.Subset(dataset, indices=range(8664, len(dataset)))
    train_dataloader = data.dataloader.DataLoader(
        train_dataset, 
        num_workers=opt.n_threads,
        batch_size=opt.batch_size,
        shuffle=True
    )
    test_sampler = data.SequentialSampler(test_dataset)
    test_dataloader = data.dataloader.DataLoader(
        test_dataset, 
        num_workers=opt.n_threads,
        batch_size=opt.test_batch_size,
        shuffle=False,
        sampler=test_sampler
    )
    
    for e in range(opt.epochs):
        if opt.test_only:
            test(model, test_dataloader)
            break
        print('epoch: ', e)
        train(model, mseloss, optimizer, train_dataloader)
        test(model, test_dataloader)
        torch.save(model.state_dict(), opt.save_path)

if __name__ == '__main__':
    main(opt)