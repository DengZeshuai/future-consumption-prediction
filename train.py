import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import LSTM
from dataset import PowerDataset 
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="parse the parameter")
parser.add_argument('--epochs', type=int, default=1000, help='the number of train epoch, each epoch will scan the data totally')
parser.add_argument('--train', type=bool, default=True, help="train or test")
parser.add_argument('--month', type=int, default=7, help="the power consumption used to train or test")
parser.add_argument('--sheet_number', type=int, default=0, help="the nubmer of sheet in excel file")
parser.add_argument('--filename', default='air-condition-consumption2.xlsx', help="the file name of data")
parser.add_argument('--batch_size', type=int, default=10, help='mini-batch size')
parser.add_argument('--n_threads', type=int, default=1, help='the thraeds number fo dataloader')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--save_path', default='logs/', help='log path')
parser.add_argument('--input_size', type=int, default=5, help='input size of the data from model')
parser.add_argument('--seq_length', type=int, default=10, help='the sequence length of input, less than 24')
parser.add_argument('--output_size', type=int, default=1, help='output size of the data from model')
parser.add_argument('--hidden_dim', type=int, default=5, help='the dimentional of hidden_dim in lstm')
parser.add_argument('--num_layers', type=int, default=1, help='')
parser.add_argument('--print_every', type=int, default=100, help='')
opt = parser.parse_args()
# print(opt)
torch.manual_seed(1)

def test():
    pass

def main(opt):
    model = LSTM(opt,batch_first=True)
    optimizer = optim.Adam(model.parameters(), opt.learning_rate)
    loss = nn.MSELoss()
    
    dataset = PowerDataset(opt, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    dataloader = data.dataloader.DataLoader(
        dataset, 
        num_workers=opt.n_threads,
        batch_size=opt.batch_size,
        shuffle=True
    )

    for _ in range(opt.epochs):
        for iter, (inputs, target) in enumerate(dataloader, 1):
            optimizer.zero_grad()
            output = model(inputs)
            print(iter, "input size: {} target: {} output: {}".format(inputs.size(), target.size(), output.size()))
            loss = loss(output, target)
            loss.backward()
            optimizer.step()
            if iter % 100 == 0:
                print('loss: {}'.format(loss))

if __name__ == '__main__':
    main(opt)