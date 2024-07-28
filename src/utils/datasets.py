import csv
import os
import json

def get_header(path):
    with open(path, "r") as f:
        dict_reader = csv.DictReader(f)
        headers = dict_reader.fieldnames
        
    return headers
def get_label(path):
    _, label = os.path.split(path)
    label = label.replace(".csv", "")

    return label

def create_merge_node_statement(path, node_label):
    headers = get_header(path)
    headers.remove("prop_id")
    headers.remove("main_prop")

    sub_prop = ""
    if len(headers) > 0:
        sub_prop += ", ".join(["{}: row.{}".format(x, x) for x in headers])

    main_prop = "id: row.prop_id, name: row.main_prop"
    full_prop = main_prop + sub_prop

    statement = "MERGE (n: {} {{{}}})"\
        .format(node_label, full_prop)
    
    return statement

def create_merge_relation_statement(path, rel_label):
    headers = get_header(path)
    headers.remove("h_id")
    headers.remove("t_id")

    sub_prop = ""
    if len(headers) > 0:
        sub_prop += ", ".join(["{}: row.{}".format(x, x) for x in headers])

    statement = "MERGE (h)-[r: {} {{{}}}]->(t)"\
        .format(rel_label, sub_prop)
    
    return statement

def create_match_statement(mapping):
    head_label = mapping["head"]
    tail_label = mapping["tail"]

    statement = "MATCH (h: {} {{id: row.h_id}}), (t: {} {{id: row.t_id}})".format(head_label, tail_label)

    return statement