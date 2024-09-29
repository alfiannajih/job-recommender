import spacy
import os
import pandas as pd

from job_recommender.config.configuration import NerConfig
from job_recommender.utils.dataset import nltk_process

class EntityRecognition:
    def __init__(
        self,
        config: NerConfig,
    ):
        self.config = config
        self.nlp = spacy.load(self.config.ner_model_path)

    def detect_entity(self, desc) -> dict:
        # initialize nlp doc
        doc = self.nlp(desc)

        # initialize empty list to the corresponding label
        ents = {label: [] for label in self.config.labels}

        for ent in doc.ents:
            normalized_ent = nltk_process(ent.text)
            if normalized_ent:
                ents[ent.label_].append(normalized_ent)
        
        return ents # {label1: [], label2: [], ...}

    def update_node_file(self, node_label, update_nodes) -> list:
        # Define path for the node label
        node_path = os.path.join(self.config.preprocessed_path, "nodes", f"{node_label}.csv")

        # Create csv file if not exits
        if not os.path.exists(node_path):
            with open(node_path, "w") as fp:
                fp.write("prop_id,main_prop\n")
        
        # Retrieve node csv
        temp_nodes = pd.read_csv(node_path)["main_prop"].to_dict()
        current_nodes = {j: i for i, j in temp_nodes.items()}

        new_nodes = []
        head_ids = []
        
        # Iterate over update_nodes to check duplicated in current_nodes
        for update_node in update_nodes:
            if update_node in current_nodes:
                head_ids.append(current_nodes[update_node])
            else:
                new_ids = len(current_nodes)
                new_nodes.append(f'{new_ids},"{update_node}"')
                head_ids.append(new_ids)
                current_nodes[update_node] = new_ids
        
        # Update file with new nodes
        if new_nodes != []:
            with open(node_path, "a") as fp:
                fp.write("\n".join(new_nodes) + "\n")
        
        return list(set(head_ids))

    def update_rel_file(self, relation_label, node_label, job_id, head_ids) -> None:
        # Get relation_label that corresponds to node_label
        relation_path = os.path.join(self.config.preprocessed_path, "relations", f"{relation_label}_{node_label}.csv")

        # Create csv file if not exits
        if not os.path.exists(relation_path):
            with open(relation_path, "w") as fp:
                fp.write("h_id,t_id\n")
        
        # Filter data that only occures in job_id
        rel_df = pd.read_csv(relation_path)
        current_head_ids = rel_df[rel_df["t_id"] == job_id]["h_id"].tolist()
        
        # Initialize list of head_ids that connect to job_id
        new_connection = []

        # Iterate over filtered data with updated_nodes (head_ids)
        for head_id in head_ids:
            # Ignored if already there
            if head_id in current_head_ids:
                continue
            else:
                new_connection.append(f"{head_id},{job_id}")
        
        # Update file with new connection
        if new_connection != []:
            with open(relation_path, "a") as fp:
                fp.write("\n".join(new_connection) + "\n")