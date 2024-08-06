from job_recommender.config.configuration import KGRetrievalConfig
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.utils.common import get_emb_model, get_rerank_model

class KnowledgeGraphRetrieval:
    def __init__(self, config: KGRetrievalConfig, neo4j_connection: Neo4JConnection):
        self.config = config
        self.neo4j_connection = neo4j_connection

        self.embedding_model = get_emb_model(self.config.embedding_model)
        self.rerank_model = get_rerank_model(self.config.rerank_model)

    def query_relationship_from_node(self, query, n_query):
        similar_relations = self.neo4j_connection.driver.execute_query(
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
    
    def rerank_retrieved_relationship(self, retrieved, query, top_k):
        documents = [t["text"] for t in retrieved]
        
        results = self.rerank_model.rank(query, documents, top_k=top_k)
        rel_ids = [retrieved[i["corpus_id"]]["rel_id"] for i in results]

        return rel_ids