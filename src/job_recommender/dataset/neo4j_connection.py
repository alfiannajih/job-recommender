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
        label_result = self.driver.execute_query(
            "MATCH (n) RETURN DISTINCT LABELS(n)"
        )
        labels = [label.value()[0] for label in label_result.records]

        return labels
    
    def get_relation_labels(self):
        label_result = self.driver.execute_query(
            "MATCH ()-[r]->() RETURN DISTINCT TYPE(r)"
        )
        labels = [label.value() for label in label_result.records]

        return labels
    
    def get_node_keys(self, node_label):
        keys = self.driver.execute_query(
            """
            MATCH (n:{})
            WITH KEYS(n) AS keys
            UNWIND keys AS key
            WITH collect(DISTINCT key) AS sortedKeys
            RETURN sortedKeys
            """.format(node_label)
        ).records[0].values()[0]
            
        return keys

    def get_relation_keys(self, relation_label):
        keys = self.driver.execute_query(
            """
            MATCH ()-[r:{}]->()
            WITH KEYS(r) AS keys
            UNWIND keys AS key
            WITH collect(DISTINCT key) AS sortedKeys
            RETURN sortedKeys
            """.format(relation_label)
        ).records[0].values()[0]

        return keys
    
    def create_vector_index_nodes(self, node_label, emb_size):
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

    def create_vector_index_relations(self, relation_label, emb_size):
        with self.get_session() as session:
            session.run(
                """
                CREATE VECTOR INDEX {}Index
                FOR ()-[r:{}]->()
                ON r.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """.format(relation_label, relation_label, emb_size)
            )

    def get_node_label_from_relation(self, relation):
        node_result = self.driver.execute_query(
            """
            MATCH (a)-[r:{}]->(b)
            WITH COLLECT(DISTINCT labels(a)) + COLLECT(DISTINCT labels(b)) AS allLabels
            UNWIND allLabels AS labelsList
            UNWIND labelsList AS label
            RETURN label
            """.format(relation)
        )
        nodes = [node.value() for node in node_result.records]
        
        return nodes

    def query_relationship_from_node(self, emb_model, query, n_query):
        query_emb = list(emb_model.encode(query))

        similar_relations = self.driver.execute_query(
            """
            CALL db.index.vector.queryNodes('JobTitleIndex', {}, {})
            YIELD node, score
            MATCH p=(node)-[r:offered_by]->(connectedNode)
            RETURN elementId(r) AS id, r.job_qualification, r.job_responsibility, r.location
            """.format(n_query, query_emb)
        )
        
        relations = []
        for relation in similar_relations.records:
            _id = relation.get("id")
            text = "Job qualification: {}\nJob responsibility: {}\nJob location: {}".format(relation.get("r.job_qualification"), relation.get("r.job_responsibility"), relation.get("r.location"))

            relations.append({"rel_id": _id, "text": text})
        
        return relations
    
    def rerank_retrieved_relationship(self, documents, retrieved, query, rerank_model, top_k):
        results = rerank_model.rank(query, documents, top_k=top_k)
        rel_ids = [retrieved[i["corpus_id"]]["rel_id"] for i in results]

        return rel_ids

    def close(self):
        self.driver.close()