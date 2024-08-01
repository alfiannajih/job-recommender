from neo4j import GraphDatabase
import os

from job_recommender.utils.dataset import create_merge_node_statement, create_merge_relation_statement, create_match_statement, get_label

ABSOLUTE_PATH = os.path.abspath(".")

class Neo4JConnection:
    def __init__(self, uri, user, password, db):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.db = db

    def get_session(self):
        return self.driver.session(database=self.db)
    
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

    def get_node_labels(self):
        with self.get_session() as session:
            label_result = session.run("MATCH (n) RETURN DISTINCT LABELS(n)")
            labels = [label.value()[0] for label in label_result]

        return labels
    
    def get_relation_labels(self):
        with self.get_session() as session:
            label_result = session.run("MATCH ()-[r]->() RETURN DISTINCT TYPE(r)")
            labels = [label.value() for label in label_result]

        return labels
    
    def get_node_keys(self, node_label):
        with self.get_session() as session:
            key_result = session.run(
                """
                MATCH (n:{})
                WITH KEYS(n) AS keys
                UNWIND keys AS key
                WITH collect(DISTINCT key) AS sortedKeys
                RETURN sortedKeys
                """.format(node_label)
            )
            keys = key_result.value()[0]

            keys.remove("id")
        return keys

    def get_relation_keys(self, relation_label):
        with self.get_session() as session:
            key_result = session.run(
                """
                MATCH ()-[r:{}]->()
                WITH KEYS(r) AS keys
                UNWIND keys AS key
                WITH collect(DISTINCT key) AS sortedKeys
                RETURN sortedKeys
                """.format(relation_label)
            )
            keys = key_result.value()[0]

        return keys
    
    def create_vector_index(self, node_label, emb_size):
        with self.get_session() as session:
            session.run(
                """
                CREATE VECTOR INDEX {}Index
                FOR (n:{})
                ON n.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """.format(node_label, node_label, emb_size)
            )

    def query_node_index(self, node_label, query, n_query):
        similar_nodes = self.driver.execute_query(
            """
            CALL db.index.vector.queryNodes('{}Index', {}, {})
            YIELD node, score
            RETURN node.name, score
            """.format(node_label, n_query, query)
        )
        
        return similar_nodes

    def close(self):
        self.driver.close()