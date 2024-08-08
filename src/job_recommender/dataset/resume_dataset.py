from torch.utils.data import Dataset
from job_recommender.utils.common import get_emb_model
from pathlib import Path
import pandas as pd
import torch
import os

class ResumeDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.prompt = "Please give a feedback on the given resume for applying a job."
        self.path = path
        self.listdir = os.listdir(path)
    
    def __getitem__(self, index):
        path = Path(self.path, self.listdir[index])
        subgraph = torch.load(Path(path, 'subg.pt'))
        desc = open(Path(path, 'desc.txt')).read()
        question = open(Path(path, 'question.txt')).read()
        label = open(Path(path, 'label.txt')).read()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': subgraph,
            'desc': desc,
        }
    
    def __len__(self):
        return len(self.listdir)