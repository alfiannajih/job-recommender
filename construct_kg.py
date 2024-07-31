import argparse
import os
import json

from job_recommender import logger
from job_recommender.dataset.neo4j_connection import Neo4JConnection

def main(args):
    # Create connection to Neo4j
    neo4j_conn = Neo4JConnection(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password
    )
    NODES_PATH = os.path.join(args.input_directory, "nodes")
    REL_PATH = os.path.join(args.input_directory, "relations")

    # List all csv files for nodes from input directory
    nodes_csv_files = os.listdir(NODES_PATH)
    
    # For loop insert csv files into Neo4j
    for nodes_csv_file in nodes_csv_files:
        if nodes_csv_file.endswith(".csv"):
            path = os.path.join(NODES_PATH, nodes_csv_file)
            neo4j_conn.load_csv_to_nodes(path)

            logger.info("Loaded nodes from {}".format(path))

    # List all csv files for relation from input directory
    rels_csv_files = os.listdir(REL_PATH)

    # List mapping rel_label to node_labels
    rel_mapping_path = os.path.join(REL_PATH, "rel_mapping.json")

    with open(rel_mapping_path, "r") as fp:
        rel_to_nodes = json.load(fp)

    # For loop insert csv files into Neo4j
    for rels_csv_file in rels_csv_files:
        if rels_csv_file.endswith(".csv"):
            path = os.path.join(REL_PATH, rels_csv_file)
            neo4j_conn.load_csv_to_relations(path, rel_to_nodes)

            logger.info("Loaded relations from {}".format(path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_directory", type=str, help="Input file.")
    parser.add_argument("--neo4j_uri", type=str, help="Neo4j URI.")
    parser.add_argument("--neo4j_user", type=str, help="Neo4j user.")
    parser.add_argument("--neo4j_password", type=str, help="Neo4j password.")

    args = parser.parse_args()

    main(args)