import pandas as pd
import numpy as np
import torch
import torch.data as data

def excel_loader(filename,sheet_number):
    """读取训练数据"""
    return pd.read_excel(filename, sheet_name=sheet_number)

class PowerDataset(data.Dataset):
    def __init__(self, filename, loader=excel_loader):
        self.loader = excel_loader
    
    def __getitem__(self):
        
        return None

    def __len__(self):
        pass