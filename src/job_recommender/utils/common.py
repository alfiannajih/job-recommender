import yaml
import json
from box import ConfigBox
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

def read_json(path):
    with open(path, "r") as fp:
        return json.load(fp)

def read_yaml(path):
    with open(path) as fp:
        return ConfigBox(yaml.safe_load(fp))
    
def get_emb_model(path):
    model = SentenceTransformer(model_name_or_path=path)

    return model

def get_rerank_model(path):
    model = CrossEncoder(model_name=path)

    return model