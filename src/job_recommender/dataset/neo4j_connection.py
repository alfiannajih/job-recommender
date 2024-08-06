from neo4j import GraphDatabase
import pathlib

from job_recommender.config.configuration import Neo4jConfig

ABSOLUTE_PATH = pathlib.Path(".").resolve()

class Neo4JConnection:
    def __init__(self, config: Neo4jConfig):
        self.driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        self.db = config.neo4j_db
        self.driver.verify_connectivity()

    def get_session(self):
        return self.driver.session(database=self.db)

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
    
    def get_all_nodes(self, session, node_label, node_keys):
        key_statement = ", ".join(["n.{}".format(key) for key in node_keys])

        result = session.run(
            """
            MATCH (n:{})
            RETURN {}
            """.format(node_label, key_statement)
        )

        return result

    def get_all_relations(self, session, relation_label, relation_keys):
        key_statement = ", ".join(["r.{}".format(key) for key in relation_keys])

        result = session.run(
            """
            MATCH ()-[r:{}]->()
            RETURN {}
            """.format(relation_label, key_statement)
        )

        return result

    def get_node_counts(self, node_label):
        node_counts = self.driver.execute_query(
            """
            MATCH (n:{})
            RETURN COUNT(n)
            """.format(node_label)
        ).records[0].value()

        return node_counts
    
    def get_relation_counts(self, relation_label):
        relation_counts = self.driver.execute_query(
            """
            MATCH ()-[r:{}]->()
            RETURN COUNT(r)
            """.format(relation_label)
        ).records[0].value()

        return relation_counts
    
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
    
    def delete_vector_index(self, label):
        with self.get_session() as session:
            session.run(
                """
                DROP INDEX {}Index IF EXISTS
                """.format(label)
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

    def get_head_node(self, relation_ids):
        node_ids = self.driver.execute_query(
            """
            MATCH (h)-[r]->()
            WHERE elementId(r) IN {}
            RETURN DISTINCT elementId(h) AS id
            """.format(relation_ids)
        )
        nodes = [node.value() for node in node_ids.records]
        
        return nodes
    
    def get_tail_node(self, relation_ids):
        node_ids = self.driver.execute_query(
            """
            MATCH ()-[r]->(t)
            WHERE elementId(r) IN {}
            RETURN DISTINCT elementId(t) AS id
            """.format(relation_ids)
        )
        nodes = [node.value() for node in node_ids.records]
        
        return nodes
    
    def get_tail_connection_from_head(self, head_ids):
        relation_ids = self.driver.execute_query(
            """
            MATCH (h)-[r]->()
            WHERE elementId(h) IN {}
            RETURN DISTINCT elementId(r) AS id LIMIT 50
            """.format(head_ids)
        )
        relations = [relation.value() for relation in relation_ids.records]

        return relations

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
        #query_emb = list(emb_model.encode(query))

        similar_relations = self.driver.execute_query(
            """
            CALL db.index.vector.queryNodes('JobTitleIndex', {}, {})
            YIELD node, score
            MATCH p=(node)-[r:offered_by]->(connectedNode)
            RETURN elementId(r) AS id, r.job_qualification, r.job_responsibility, r.location
            """.format(n_query, query)
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
    
    def query_node(self, node_label, emb_model, query, n_query):
        #query_emb = list(emb_model.encode(query))

        similar_nodes = self.driver.execute_query(
            """
            CALL db.index.vector.queryNodes('{}Index', {}, {})
            YIELD node
            RETURN elementId(node) AS id
            """.format(node_label, n_query, query)
        )

        ids = [r["id"] for r in similar_nodes.records]

        return ids

    def close(self):
        self.driver.close()