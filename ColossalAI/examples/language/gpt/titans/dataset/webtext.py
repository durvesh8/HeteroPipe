import json
import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import glob
from tqdm import tqdm

from colossalai.registry import DATASETS


@DATASETS.register_module
class WebtextDataset(Dataset):

    def __init__(self, path: Optional[str] = None, seq_len=1024) -> None:
        super().__init__()
        if path is not None:
            root = os.path.dirname(path)
            self.files = glob.glob(os.path.join(path, '*.txt'))
            self.files = sorted(self.files)
            encoded_data_cache_path = f'gpt_webtext_{seq_len}.pt'
            if os.path.isfile(encoded_data_cache_path):
                data, attention_mask = torch.load(encoded_data_cache_path)
                self.data = data
                self.attention_mask = attention_mask
                return
            raw_data = []
            print("Loading Data")
            print("Seq Len",seq_len)
            for i in tqdm(range(0,100)):
                filepath = self.files[i]
                with open(filepath) as f:
                    for line in f.readlines():
                        raw_data.append(line)
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.unk_token
                encoded_data = tokenizer(raw_data, padding='max_length', truncation=True, max_length=seq_len, return_tensors='pt')
                self.data = encoded_data['input_ids']
                self.attention_mask = encoded_data['attention_mask']
            print("Finished Loading Data")
            #print(self.data.shape)
            #print(self.data)
            torch.save((self.data, self.attention_mask), encoded_data_cache_path)
        else:
            self.data = torch.randint(0, 50257, (10240, seq_len))
            self.attention_mask = torch.ones_like(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'input_ids': self.data[index], 'attention_mask': self.attention_mask[index]}, self.data[index]
