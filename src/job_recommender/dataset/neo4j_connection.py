from neo4j import GraphDatabase
import os

from job_recommender.utils.dataset import create_merge_node_statement, create_merge_relation_statement, create_match_statement, get_label

ABSOLUTE_PATH = os.path.abspath(".")

class Neo4JConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_session(self):
        return self.driver.session()
    
    def load_csv_to_nodes(self, path):
        label = get_label(path)
        merge_statement = create_merge_node_statement(path, label)
        
        csv_path = os.path.join(ABSOLUTE_PATH, path)

        with self.get_session() as session:
            session.run(
                """
                LOAD CSV WITH HEADERS FROM 'file://{}' AS row
                {}
                """.format(csv_path, merge_statement))

    def load_csv_to_relations(self, path, mapping):
        label = get_label(path)
        
        match_statement = create_match_statement(mapping[label])
        merge_statement = create_merge_relation_statement(path, label)

        csv_path = os.path.join(ABSOLUTE_PATH, path)

        with self.get_session() as session:
            session.run(
                """
                LOAD CSV WITH HEADERS FROM 'file://{}' AS row
                {}
                {}
                """.format(csv_path, match_statement, merge_statement))

    def close(self):
        self.driver.close()