import os
import pandas as pd

from job_recommender.dataset.named_entity_recognition import EntityRecognition
from job_recommender.config.configuration import NerConfig
from job_recommender import logger

class EntityRecognitionPipeline(EntityRecognition):
    def __init__(
        self,
        config: NerConfig
    ):
        EntityRecognition.__init__(self, config)
        
        desc_path = os.path.join(self.config.preprocessed_path, "relations/offered_by.csv")
        self.job_desc = pd.read_csv(desc_path, usecols=[0, 2])

    def get_job_desc(self, idx):
        desc = self.job_desc.iloc[idx]

        return desc.h_id, desc.description

    def extract_entity_pipeline(self, relation_label="required_by"):
        for idx in range(len(self.job_desc)):
            job_id, job_desc = self.get_job_desc(idx)

            ents = self.detect_entity(job_desc)
            
            for node_label in ents:
                head_ids = self.update_node_file(node_label, ents[node_label])
                self.update_rel_file(relation_label, node_label, job_id, head_ids)
            
            if idx%100==0:
                logger.info("[{}\{} Perform named entity recognition on job description".format(idx, len(self.job_desc)))