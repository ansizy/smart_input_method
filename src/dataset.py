"""
定义 Dataset
提供一个获取 dataloader 的方法
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import config


class InputMethodDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = pd.read_json(path, orient='records', lines=True).to_dict(orient='records')

    def __getitem__(self, index):
        input = self.data[index]['input']
        input_tensor = torch.tensor(input, dtype=torch.long)

        target = self.data[index]['target']
        target_tensor = torch.tensor(target, dtype=torch.long)
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.data)


# 返回 dataloader
def get_dataloader(train=True):

    if train:
        path = config.PROCESSED_DATA_DIR / "train.jsonl"
    else:
        path = config.PROCESSED_DATA_DIR / "test.jsonl"

    dataset = InputMethodDataset(path)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train)
    return dataloader