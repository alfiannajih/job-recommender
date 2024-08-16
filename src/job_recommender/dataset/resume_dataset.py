from torch.utils.data import Dataset
from job_recommender.utils.common import get_emb_model
from pathlib import Path
import pandas as pd
import torch
import os

from job_recommender.config.configuration import ResumeDatasetConfig
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.pipeline.knowledge_graph import KnowledgeGraphRetrievalPipeline

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
            'id': index,
            'question': question,
            'label': label,
            'graph': subgraph,
            'desc': desc,
        }
    
    def __len__(self):
        return len(self.listdir)