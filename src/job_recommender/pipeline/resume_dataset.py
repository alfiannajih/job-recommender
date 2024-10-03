import json
import os
import torch
import re
from tqdm import tqdm

from job_recommender.dataset.resume_dataset import PreprocessedResumeDataset
from job_recommender.config.configuration import ResumeDatasetConfig
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.pipeline.knowledge_graph_retrieval import KnowledgeGraphRetrievalPipeline
from job_recommender import logger

class PreprocessedResumeDatasetPipeline(PreprocessedResumeDataset):
    def __init__(
        self,
        config: ResumeDatasetConfig,
        neo4j_connection: Neo4JConnection,
        retriever: KnowledgeGraphRetrievalPipeline
    ):
        PreprocessedResumeDataset.__init__(self, config, neo4j_connection, retriever)
        self.files = os.listdir(self.config.input_dir)
    
    def retrieve_batch_graph(self):
        for i, file in enumerate(self.files):
            logger.info("Processing file: {} [{}/{}]".format(file, i, len(self.files)))
            path = os.path.join(self.config.input_dir, file)
            with open(path, "r") as fp:
                json_files = [json.loads(line.rstrip()) for line in fp]
            
            for j, inputs in enumerate(tqdm(json_files)):
                output_dir = os.path.join(self.config.resume_dir, "resume_{}_{}".format(file, j))
                
                if os.path.exists(output_dir):
                    continue
                
                question = inputs["input"]
                # label = inputs["output"]

                word_split = ["\neducation\n", "\nexperience\n", "\nproject\n", "\nskills\n", "RESUME:\n","\nDESCRIPTION:\n"]
                pattern = '|'.join(map(re.escape, word_split))
                splitted = re.split(pattern, question)
                
                resume = splitted[1:-1]
                desc = splitted[-1]

                subgraph, textualized_graph = self.retrieve_graph(resume, desc)
                # output_dir = os.path.join(self.config.resume_dir, "resume_{}_{}".format(file[:-6], j))
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Write question
                with open(os.path.join(output_dir, "question.txt"), "w") as fp:
                    fp.write(question)

                # Write label
                # with open(os.path.join(output_dir, "label.txt"), "w") as fp:
                #     fp.write(label)

                # Write textualized graph
                with open(os.path.join(output_dir, "desc.txt"), "w") as fp:
                    fp.write(textualized_graph)
                
                # Write subgraph
                torch.save(subgraph, os.path.join(output_dir, "subg.pt"))

                    
