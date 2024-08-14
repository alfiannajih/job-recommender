import pathlib
import torch
import pandas as pd
import os
from torch_geometric.data.data import Data

from job_recommender.utils.dataset import retrieval_via_pcst
from job_recommender.config.configuration import (
    KGConstructConfig,
    KGIndexingConfig,
    KGRetrievalConfig,
    RawDatasetConfig
)
from job_recommender import logger
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.utils.dataset import list_csv_files
from job_recommender.dataset.knowledge_graph import (
    KnowledgeGraphConstruction,
    KnowledgeGraphIndexing,
    KnowledgeGraphRetrieval,
    PrepareRawDataset
)

class PrepareRawDatasetPipeline(PrepareRawDataset):
    def __init__(self, config: RawDatasetConfig):
        PrepareRawDataset.__init__(self, config)
    
    def node_job_title_pipeline(self):
        posting_df = pd.read_csv(self.posting_path, usecols=[0, 2, 3, 6, 7], index_col=0)
        self.process_node_job_title(posting_df, 50)
    
    def node_company_pipeline(self):
        company_df = pd.read_csv(self.companies_profile_path, usecols=[0, 1, 3, 4, 5, 6], index_col=0)[["name"]]
        posting_df = pd.read_csv(self.posting_path, usecols=[0, 2, 3, 6, 7], index_col=0)
        self.process_node_company(company_df, posting_df, 25)

    def node_industry_pipeline(self):
        job_industry_df = pd.read_csv(self.mappings_industries_path, index_col=0).dropna()

        company_industries_df = pd.read_csv(self.company_industries_path, index_col=0)
        company_industry_df = company_industries_df["industry"].value_counts()

        self.process_node_industry(job_industry_df, company_industry_df, 20)

    def node_speciality_pipeline(self):
        company_specialities_df = pd.read_csv(self.company_specialities_path, index_col=0)
        company_specialization_df = company_specialities_df["speciality"].value_counts()
        
        self.process_node_company_speciality(company_specialization_df, 20)

    def node_size_pipeline(self):
        self.process_node_company_size()

    def rel_job_company_pipeline(self):
        job_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/JobTitle.csv"), index_col=0)
        company_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/Company.csv"), index_col=0)
        posting_df = pd.read_csv(self.posting_path, usecols=[0, 2, 3, 6, 7], index_col=0)

        self.process_rel_job_company(job_nodes, company_nodes, posting_df)

    def rel_job_industry_pipeline(self):
        job_industry_df = pd.read_csv(self.job_industries_path, index_col=0)
        industry_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/Industry.csv"), index_col=0)
        job_comp_rel = pd.read_csv(os.path.join(self.config.preprocessed_path, "relations/offered_by.csv"), index_col=0)

        self.process_rel_job_industry(job_industry_df, industry_nodes, job_comp_rel)

    def rel_company_industry_pipeline(self):
        industry_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/Industry.csv"), index_col=0)
        company_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/Company.csv"), index_col=0)
        company_industries_df = pd.read_csv(self.company_industries_path, index_col=0)

        self.process_rel_company_industry(industry_nodes, company_industries_df, company_nodes)

    def rel_company_speciality_pipeline(self):
        comp_specialization_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/Speciality.csv"), index_col=0)
        company_specialities_df = pd.read_csv(self.company_specialities_path, index_col=0)
        company_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/Company.csv"), index_col=0)
        
        self.process_rel_company_speciality(comp_specialization_nodes, company_specialities_df, company_nodes)

    def rel_company_size_pipeline(self):
        company_size_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/Size.csv"), index_col=0)
        companies_profile_df = pd.read_csv(self.companies_profile_path, usecols=[0, 1, 3, 4, 5, 6],  index_col=0)
        company_nodes = pd.read_csv(os.path.join(self.config.preprocessed_path, "nodes/Company.csv"), index_col=0)

        self.process_rel_company_size(company_size_nodes, companies_profile_df, company_nodes)
    
    def node_preprocess_pipeline(self):
        self.node_job_title_pipeline()
        self.node_company_pipeline()
        self.node_industry_pipeline()
        self.node_speciality_pipeline()
        self.node_size_pipeline()

    def relation_preprocess_pipeline(self):
        self.rel_job_company_pipeline()
        self.rel_job_industry_pipeline()
        self.rel_company_industry_pipeline()
        self.rel_company_speciality_pipeline()
        self.rel_company_size_pipeline()

class KnowledgeGraphConstructionPipeline(KnowledgeGraphConstruction):
    """
    A class to run the pipeline for constructing knowledge graph into neo4j database.

    This is inheritance of KnowledgeGrpahConstruction Class

    Args:
        config (KGConstructConfig): Configuration setting from the config module
        neo4j_connection (Neo4JConnection): Connection instance from the connection module

    Attributes:
        config (KGConstructConfig): Configuration setting to constructing knowledge graph.
        neo4j_connection (Neo4jConnection): Connection instance to Neo4j database.
        nodes_path (str): Directory that contains CSV files to construct nodes.
        relations_path (str): Directory that contains CSV files to construct relations.
        relations_map_path (str): Path of json files that mapping relations label with their connected nodes.
    """
    def __init__(
            self,
            config: KGConstructConfig,
            neo4j_connection: Neo4JConnection
        ):
        KnowledgeGraphConstruction.__init__(self, config, neo4j_connection)

    def node_construction_pipeline(self):
        """
        Construct nodes from CSV files iteratively for all listed nodes in the directory.
        """
        logger.info("Constructing nodes from {}".format(self.nodes_path))

        nodes_csv_files = list_csv_files(self.nodes_path)
        logger.info("Listed {} CSV files: {}".format(len(nodes_csv_files), nodes_csv_files))

        for nodes_csv_file in nodes_csv_files:
            path = pathlib.Path(self.nodes_path, nodes_csv_file)
            self.load_csv_to_nodes(path)
        
        logger.info("Construction from all listed nodes is finished")
    
    def relation_construction_pipeline(self):
        """
        Construct relations from CSV files iteratively for all listed relations in the directory.
        """
        logger.info("Constructing relations from {}".format(self.relations_path))

        rels_csv_files = list_csv_files(self.relations_path)
        logger.info("Listed {} CSV files: {}".format(len(rels_csv_files), rels_csv_files))
        
        for rels_csv_file in rels_csv_files:
            path = pathlib.Path(self.relations_path, rels_csv_file)
            self.load_csv_to_relations(path)
        
        logger.info("Construction from all listed relations is finished")
    
    def knowledge_graph_construction_pipeline(self):
        """
        Run both the node_construction_pipeline and relation_construction_pipeline to construct knowledge graph
        """
        logger.info("Constructing knowledge graph from {}".format(self.config.input_dir))
        self.node_construction_pipeline()
        self.relation_construction_pipeline()
        logger.info("Knowledge graph construction is finisher")

class KnowledgeGraphIndexingPipeline(KnowledgeGraphIndexing):
    def __init__(
            self,
            config: KGIndexingConfig,
            neo4j_connection: Neo4JConnection
        ):
        KnowledgeGraphIndexing.__init__(self, config, neo4j_connection)
    
    def single_node_label_indexing(self, session, node_label):
        self.neo4j_connection.delete_vector_index(node_label)
        
        node_keys = self.neo4j_connection.get_node_keys(node_label)
        results = self.neo4j_connection.get_all_nodes(session, node_label, node_keys)
        logger.info("Start processing node: [{}]".format(node_label))

        node_counts = self.neo4j_connection.get_node_counts(node_label)
        logger.info("Node [{}] contains {} nodes and keys: [{}]".format(node_label, node_counts, node_keys))

        node_with_embeddings = []
        i = 1
        batch_n = 1
        for record in results:
            property = {k: record.get("n.{}".format(k)) for k in node_keys}
            emb_property = self.embed_property(node_label, property)

            node_with_embeddings.append(emb_property)

            if i%self.config.batch_size == 0:
                self.import_batch_nodes(session, node_with_embeddings, node_label, node_keys, batch_n)
                node_with_embeddings = []
                batch_n += 1
            
            elif i == node_counts:
                self.import_batch_nodes(session, node_with_embeddings, node_label, node_keys, batch_n)
            
            i += 1
        self.neo4j_connection.create_vector_index_nodes(node_label, self.embedding_model.get_sentence_embedding_dimension())
        
    def nodes_indexing(self):
        node_labels = self.neo4j_connection.get_node_labels()

        for node_label in node_labels:
            with self.neo4j_connection.get_session() as session:
                self.single_node_label_indexing(session, node_label)
        
        self.neo4j_connection.close()

    def single_relation_label_indexing(self, session, relation_label):
        self.neo4j_connection.delete_vector_index(relation_label)
        relation_keys = self.neo4j_connection.get_relation_keys(relation_label)
        head, tail = self.neo4j_connection.get_node_label_from_relation(relation_label)
        textualized_relation = "{}.{}.{}".format(head, relation_label, tail)

        if relation_keys == []:
            self._relation_indexing_no_property(session, relation_label, textualized_relation)
        
        else:
            self._relation_indexing_with_property(session, relation_label, relation_keys)

        self.neo4j_connection.create_vector_index_relations(relation_label, self.embedding_model.get_sentence_embedding_dimension())

    def _relation_indexing_no_property(self, session, relation_label, textualized_relation):
        embedding = self.embedding_model.encode(textualized_relation, show_progress_bar=False)

        session.run(
            """
            MATCH ()-[r:{}]->()
            CALL db.create.setRelationshipVectorProperty(r, 'embedding', {})
            """.format(relation_label, list(embedding))
        )
    
    def _relation_indexing_with_property(self, session, relation_label, relation_keys):
        results = self.neo4j_connection.get_all_relations(session, relation_label, relation_keys)
        logger.info("Start processing relation: [{}]".format(relation_label))

        relation_counts = self.neo4j_connection.get_relation_counts(relation_label)
        logger.info("Relation [{}] contains {} relations and keys: [{}]".format(relation_label, relation_counts, relation_keys))

        relation_with_embeddings = []
        i = 1
        batch_n = 1
        for record in results:
            property = {k: record.get("r.{}".format(k)) for k in relation_keys}

            emb_property = self.embed_property(relation_label, property)

            relation_with_embeddings.append(emb_property)

            if i%self.config.batch_size == 0:
                self.import_batch_relations(session, relation_with_embeddings, relation_label, relation_keys, batch_n)
                relation_with_embeddings = []
                batch_n += 1
            
            elif i == relation_counts:
                self.import_batch_relations(session, relation_with_embeddings, relation_label, relation_keys, batch_n)
            
            i += 1
    
    def relations_indexing(self):
        relation_labels = self.neo4j_connection.get_relation_labels()

        for relation_label in relation_labels:
            with self.neo4j_connection.get_session() as session:
                self.single_relation_label_indexing(session, relation_label)
        
        self.neo4j_connection.close()
    
    def knowledge_graph_indexing_pipeline(self):
        self.nodes_indexing()
        self.relations_indexing()

class KnowledgeGraphRetrievalPipeline(KnowledgeGraphRetrieval):
    def __init__(
            self,
            config: KGRetrievalConfig,
            neo4j_connection: Neo4JConnection
        ):
        KnowledgeGraphRetrieval.__init__(self, config, neo4j_connection)

    def job_triples_retrieval(self, resume, desc, top_emb=5, top_rerank=3):
        query = ["{}: {}".format(k, v) for k, v in resume.items()] + [desc]
        query_emb = self.embedding_model.encode(query).mean(axis=0).tolist()

        relations = self.query_relationship_from_node(query_emb, top_emb)
        relation_ids = self.rerank_retrieved_relationship(relations, desc, top_rerank)

        with self.neo4j_connection.get_session() as session:
            result = session.run(
                """
                MATCH (h)-[r]->(t)
                WHERE elementId(r) IN {}
                RETURN h.name AS h_name, r.job_qualification AS job_desc, r.job_responsibility AS r_job_responsibility, t.name AS t_name
                """.format(relation_ids)
            )
            job_desc = []
            for rec in result:
                job_desc.append("Job position: {}\nCompany: {}\nJob description:\n{}".format(rec.get("h_name"), rec.get("t_name"), rec.get("job_desc")))
        
        return "\n\n".join(job_desc)

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
