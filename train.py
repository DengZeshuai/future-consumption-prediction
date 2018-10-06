import torch
import argparse

parser = argparse.ArgumentParser(description="parse the parameter")

parser.add_argument('--train', type=bool, default=True, help="train or test")
parser.add_argument('--month', type=int, default=7, help="the power consumption used to train or test")
parser.add_argument('--sheet_number', type=int, default=0, help="the nubmer of sheet in excel file")
parser.add_argument('--filename', default='data/air-condition-consumption.xlsx', help="the file name of data")
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--save_path', default='logs/', help='log path')
parser.add_argument('--input_size', type=int, default=1, help='input size of the data from model')
parser.add_argument('--seq_length', type=int, default=10, help='the sequence length of input')
parser.add_argument('--output_size', type=int, default=1, help='output size of the data from model')
parser.add_argument('--cell_size', type=int, default=10, help='the number of cell in lstm')

arg = parser.parse_args()
print(arg)

def train():
    pass

def test():
    pass

def main(arg):
    pass


if __name__ == '__main__':
    main(arg)