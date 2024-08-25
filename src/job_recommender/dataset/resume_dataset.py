from torch.utils.data import Dataset
from torch_geometric.data import Batch
from pathlib import Path
import torch
import os

from job_recommender.config.configuration import ResumeDatasetConfig
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.pipeline.knowledge_graph_retrieval import KnowledgeGraphRetrievalPipeline

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

def collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    return batch

class ResumeDataset(Dataset):
    def __init__(self, path):
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