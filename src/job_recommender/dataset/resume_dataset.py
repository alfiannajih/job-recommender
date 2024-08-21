from torch.utils.data import Dataset
from job_recommender.utils.common import get_emb_model
from torch_geometric.data import Batch
from pathlib import Path
import pandas as pd
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from job_recommender.config.configuration import ResumeDatasetConfig
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.pipeline.knowledge_graph import KnowledgeGraphRetrievalPipeline

tokenizer = AutoTokenizer.from_pretrained("alfiannajih/g-retriever", add_bos_token=False)
BOS = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
EOS_USER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
EOS = "<|eot_id|><|end_of_text|>"
BOS_LENGTH = len(tokenizer(BOS)["input_ids"])
eos_user = tokenizer(EOS_USER)["input_ids"]

IGNORE_INDEX = -100

class PreprocessedResumeDataset:
    def __init__(
            self,
            config: ResumeDatasetConfig,
            neo4j_connection: Neo4JConnection,
            retriever: KnowledgeGraphRetrievalPipeline
        ):
        self.config = config
        self.neo4j_connection = neo4j_connection
        self.retriever = retriever

    def retrieve_graph(
            self,
            resume: str,
            description: str
        ):
        subgraph, textualized_graph = self.retriever.graph_retrieval_pipeline(
            resume=resume,
            desc=description,
            top_emb=5,
            top_rerank=150
        )

        return subgraph, textualized_graph

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
            'question': question,
            'label': label,
            'graph': subgraph,
            'desc': desc,
        }
    
    def __len__(self):
        return len(self.listdir)

def collate_fn(batch):
    samples = {key: [d[key] for d in batch] for key in batch[0].keys()}
    
    inputs = []
    question = tokenizer(samples["question"])["input_ids"]
    desc = tokenizer(samples["desc"])["input_ids"]
    graph_index = []
    
    for i in range(len(batch)):
        inputs.append(torch.tensor([-1]*(BOS_LENGTH + 1) + question[i]+desc[i][:256]+eos_user).flip(dims=[0]))
    
    inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id).flip(dims=[1])
    temp_labels = tokenizer(samples["label"], padding=True, return_tensors="pt")["input_ids"][:, :32]
    
    inputs = torch.cat([inputs, temp_labels], dim=1)
    labels = torch.zeros_like(inputs).fill_(IGNORE_INDEX)
    labels[:, -32:] = temp_labels
    
    graph = Batch.from_data_list(samples["graph"])
    
    attention_mask = torch.logical_and(inputs!=0, inputs!=-1).long()
    
    return inputs, graph, labels, attention_mask