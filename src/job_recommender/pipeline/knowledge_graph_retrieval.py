import pandas as pd
import torch
from torch_geometric.data.data import Data
import numpy as np

from job_recommender.utils.dataset import retrieval_via_pcst
from job_recommender.config.configuration import KGRetrievalConfig
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.dataset.knowledge_graph_retrieval import KnowledgeGraphRetrieval

class KnowledgeGraphRetrievalPipeline(KnowledgeGraphRetrieval):
    def __init__(
            self,
            config: KGRetrievalConfig,
            neo4j_connection: Neo4JConnection
        ):
        KnowledgeGraphRetrieval.__init__(self, config, neo4j_connection)

    def triples_retrieval(self, resume, desc, node_labels, top_emb=20):
        query = resume + [desc]
        
        query_emb = self.embedding_model.encode(query, show_progress_bar=False)
        query_emb = np.average(query_emb, axis=0, weights=[3/20, 3/20, 3/20, 3/20, 2/5]).tolist()

        query_executions = []

        for node_label in node_labels:
            query_executions.append(
                """
                CALL db.index.vector.queryNodes('{}', {}, {})
                YIELD node, score
                MATCH (h)-[r]-()
                WHERE h = node AND score > 0.7
                RETURN elementId(r)
                """.format(node_label, top_emb, query_emb)
            )

        full_query = "UNION".join(query_executions)
        triples = self.neo4j_connection.driver.execute_query(full_query)

        triples_id = [triple.values()[0] for triple in triples.records]

        return triples_id, torch.tensor(query_emb)
        # query_emb = self.embedding_model.encode(query, show_progress_bar=False).mean(axis=0).tolist()

        # relations = self.query_relationship_from_node(query_emb, top_emb)
        # #relation_ids = self.rerank_retrieved_relationship(relations, desc, top_rerank)
        # relation_ids = [r["rel_id"] for r in relations]

        # tail_ids = self.neo4j_connection.get_tail_node(relation_ids)
        # tail_connection = self.neo4j_connection.get_tail_connection_from_head(tail_ids)
        
        # head_ids = self.neo4j_connection.get_head_node(relation_ids)
        # head_connection = self.neo4j_connection.get_tail_connection_from_head(head_ids)

        # required_ids = self.neo4j_connection.get_required_by(head_ids)

        # return required_ids + relation_ids + tail_connection + head_connection, torch.tensor(query_emb)
    
    def build_graph(self, triples, query_emb, topk, topk_e, cost_e):
        with self.neo4j_connection.get_session() as session:
            result = session.run(
                """
                MATCH (h)-[r]->(t)
                WHERE elementId(r) IN {}
                RETURN h.name AS h_name, h.embedding AS h_embedding, TYPE(r) AS r_type, r.embedding AS r_embedding, r.description AS job_description, t.embedding AS t_embedding, t.name AS t_name
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

                if rec.get("job_description") != None:
                    textualized_prop = "{}\nJob Description: {}".format(rec.get("r_type"), rec.get("job_description")[:1024])
                else:
                    textualized_prop = rec.get("r_type")

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
            
            subgraph, desc = retrieval_via_pcst(graph, query_emb, nodes, edges, topk=topk, topk_e=topk_e, cost_e=cost_e)

            return subgraph, desc
            #torch.save(subg, 'dataset/samples/sample_1/subg.pt')

            #with open('dataset/samples/sample_1/desc.txt', 'w') as f:
            #    f.write(desc)
    
    def graph_retrieval_pipeline(self, resume, desc, node_labels, top_emb):
        triples, query_emb = self.triples_retrieval(resume, desc, node_labels, top_emb)
        subgraph, textualize_graph = self.build_graph(triples, query_emb, topk=15, topk_e=10, cost_e=0.8)
        
        return subgraph, textualize_graph
        #return graph, nodes, edges
