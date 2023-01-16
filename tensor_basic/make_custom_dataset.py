import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import dataloader

class CustomDataset(Dataset):
    def __init__(self, csv_file): #csv_file 파라미터를 통해 데이터 셋을 불러옵니다. 
        self.label  = pd.read_csv(csv_file)
    
    
    def __len__(self): #전체 데이터셋의 크기를 반환한다. 
        return len(self.label)
    
    def __get_item__(self, idx):
        sample = torch.tensor(self.label.iloc[idx, 0:3]).int()
        label = torch.tensor(self.label.iloc[idx,3]).int()
        return sample , label
    
tensor_dataset = CustomDataset('./dataset.csv')

dataset = dataloader(tensor_dataset, batch_size = 64 , shuffle  =True)    
        
        
