import torch
from torch_geometric.data import Data

from job_recommender.dataset.kg_retrieval import KnowledgeGraphRetrieval
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.config.configuration import KGRetrievalConfig

class KnowledgeGraphIndexingPipeline(KnowledgeGraphRetrieval):
    def __init__(
            self,
            config: KGRetrievalConfig,
            neo4j_connection: Neo4JConnection
        ):
        KnowledgeGraphRetrieval.__init__(self, config, neo4j_connection)

    def triples_retrieval(self, query, top_emb, top_rerank):
        query_emb = self.embedding_model.encode(query).tolist()

        relations = self.query_relationship_from_node(query_emb, top_emb)
        relation_ids = self.rerank_retrieved_relationship(relations, query, top_rerank)

        tail_ids = self.neo4j_connection.get_tail_node(relation_ids)
        tail_connection = self.neo4j_connection.get_tail_connection_from_head(tail_ids)
        
        head_ids = self.neo4j_connection.get_head_node(relation_ids)
        head_connection = self.neo4j_connection.get_tail_connection_from_head(head_ids)

        return relation_ids + tail_connection + head_connection
    
    def build_graph(self, triples):
        with self.neo4j_connection.get_session() as session:
            result = session.run(
                """
                MATCH (h)-[r]->(t)
                WHERE elementId(r) IN {}
                RETURN h.embedding AS h_embedding, elementId(h) AS h_id, r.embedding AS r_embedding, elementId(t) AS t_id, t.embedding AS t_embedding
                """.format(triples)
            )

            h_ids = []
            t_ids = []
            node_embedding = []
            node_mapping = {}
            edge_attr = []

            for rec in result:

                if rec.get("h_id") not in node_mapping:
                    node_embedding.append(rec.get("h_embedding"))
                    node_mapping[rec.get("h_id")] = len(node_mapping)

                if rec.get("t_id") not in node_mapping:
                    node_embedding.append(rec.get("t_embedding"))
                    node_mapping[rec.get("t_id")] = len(node_mapping)

                h_ids.append(rec.get("h_id"))
                t_ids.append(rec.get("t_id"))
                edge_attr.append(rec.get("r_embedding"))
            
            src = [node_mapping[index] for index in h_ids]
            dst = [node_mapping[index] for index in t_ids]

            edge_index = torch.tensor([src, dst])
            edge_attr = torch.tensor(edge_attr)

            graph = Data(x=torch.tensor(node_embedding), edge_index=edge_index, edge_attr=edge_attr)

            return graph
    
    def graph_retrieval_pipeline(self, query, top_emb, top_rerank):
        triples = self.triples_retrieval(query, top_emb, top_rerank)
        graph = self.build_graph(triples)
        return graph
