import csv
import os

def get_csv_header(path):
    with open(path, "r") as f:
        dict_reader = csv.DictReader(f)
        headers = dict_reader.fieldnames
        
    return headers

def get_label_from_path(path):
    _, label = os.path.split(path)
    label = label.replace(".csv", "")

    return label

def list_csv_files(path):
    files = os.listdir(path)

    return [f for f in files if f.endswith(".csv")]

def count_csv_rows(path):
    with open(path, encoding="utf8") as f:
        return sum(1 for line in f)
    
def textualize_property(property):
    return "\n".join(["{}: {}".format(k, v) for k, v in property.items()])