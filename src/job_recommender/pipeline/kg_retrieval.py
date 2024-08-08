import torch
import pandas as pd
from torch_geometric.data.data import Data

from job_recommender.utils.dataset import retrieval_via_pcst
from job_recommender.dataset.kg_retrieval import KnowledgeGraphRetrieval
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.config.configuration import KGRetrievalConfig

class KnowledgeGraphRetrievalPipeline(KnowledgeGraphRetrieval):
    def __init__(
            self,
            config: KGRetrievalConfig,
            neo4j_connection: Neo4JConnection
        ):
        KnowledgeGraphRetrieval.__init__(self, config, neo4j_connection)

    def triples_retrieval(self, resume, desc, top_emb=5, top_rerank=100):
        query = resume + [desc]
        query_emb = self.embedding_model.encode(query).mean(axis=0).tolist()

        relations = self.query_relationship_from_node(query_emb, top_emb)
        relation_ids = self.rerank_retrieved_relationship(relations, desc, top_rerank)

        tail_ids = self.neo4j_connection.get_tail_node(relation_ids)
        tail_connection = self.neo4j_connection.get_tail_connection_from_head(tail_ids)
        
        head_ids = self.neo4j_connection.get_head_node(relation_ids)
        head_connection = self.neo4j_connection.get_tail_connection_from_head(head_ids)

        return relation_ids + tail_connection + head_connection, torch.tensor(query_emb)
    
    def build_graph(self, triples, query_emb):
        with self.neo4j_connection.get_session() as session:
            result = session.run(
                """
                MATCH (h)-[r]->(t)
                WHERE elementId(r) IN {}
                RETURN h.name AS h_name, h.embedding AS h_embedding, TYPE(r) AS r_type, r.embedding AS r_embedding, r.job_qualification AS r_job_qualification, r.job_responsibility AS r_job_responsibility, t.embedding AS t_embedding, t.name AS t_name
                """.format(triples)
            )

            head_nodes = []
            tail_nodes = []
            node_embedding = []
            node_mapping = {}
            edge_attr = []
            edges = []
            nodes = {}

            for rec in result:
                if rec.get("h_name") not in node_mapping:
                    node_embedding.append(rec.get("h_embedding"))
                    nodes[len(node_mapping)] = rec.get("h_name")
                    node_mapping[rec.get("h_name")] = len(node_mapping)

                if rec.get("t_name") not in node_mapping:
                    node_embedding.append(rec.get("t_embedding"))
                    nodes[len(node_mapping)] = rec.get("t_name")
                    node_mapping[rec.get("t_name")] = len(node_mapping)

                head_nodes.append(rec.get("h_name"))
                tail_nodes.append(rec.get("t_name"))
                edge_attr.append(rec.get("r_embedding"))

                textualized_prop = "{}\nJob qualification: {}\nJob responsibility: {}".format(rec.get("r_type"), rec.get("r_job_qualification"), rec.get("r_job_responsibility"))

                edges.append({
                    "src": node_mapping[rec.get("h_name")],
                    "edge_attr": textualized_prop,
                    "dst": node_mapping[rec.get("t_name")]
                })
            
            src = [node_mapping[index] for index in head_nodes]
            dst = [node_mapping[index] for index in tail_nodes]

            edge_index = torch.tensor([src, dst])
            edge_attr = torch.tensor(edge_attr)

            graph = Data(x=torch.tensor(node_embedding), edge_index=edge_index, edge_attr=edge_attr)
            nodes = pd.DataFrame([{'node_id': k, 'node_attr': v} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
            edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])
            
            #torch.save(graph, 'dataset/samples/sample_1/graph.pt')
            #nodes.to_csv("dataset/samples/sample_1/node.csv", index=False)
            #edges.to_csv("dataset/samples/sample_1/edge.csv", index=False)
            
            subgraph, desc = retrieval_via_pcst(graph, query_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)

            return subgraph, desc
            #torch.save(subg, 'dataset/samples/sample_1/subg.pt')

            #with open('dataset/samples/sample_1/desc.txt', 'w') as f:
            #    f.write(desc)
    
    def graph_retrieval_pipeline(self, resume, desc, top_emb, top_rerank):
        triples, query_emb = self.triples_retrieval(resume, desc, top_emb, top_rerank)
        subgraph, desc = self.build_graph(triples, query_emb)
        
        return subgraph, desc
        #return graph, nodes, edges
