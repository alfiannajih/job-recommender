import yaml
import json
from box import ConfigBox

def read_json(path):
    with open(path, "r") as fp:
        return json.load(fp)

def read_yaml(path):
    with open(path) as fp:
        return ConfigBox(yaml.safe_load(fp))