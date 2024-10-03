import pathlib
import pandas as pd
import os
import kaggle
from tqdm import tqdm

from job_recommender.config.configuration import (
    KGConstructConfig,
    KGIndexingConfig,
    RawDatasetConfig
)
from job_recommender import logger
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.utils.dataset import list_csv_files
from job_recommender.dataset.knowledge_graph import (
    KnowledgeGraphConstruction,
    KnowledgeGraphIndexing,
    PrepareRawDataset
)

class PrepareRawDatasetPipeline(PrepareRawDataset):
    def __init__(self, config: RawDatasetConfig):
        PrepareRawDataset.__init__(self, config)
    
    def download_raw_dataset(self):
        kaggle.api.dataset_download_files(
            "arshkon/linkedin-job-postings",
            path=self.config.raw_path,
            unzip=True
        )

    def node_job_title_pipeline(self):
        posting_df = pd.read_csv(self.posting_path, usecols=[0, 2, 3, 6, 7], index_col=0)
        self.process_node_job_title(posting_df, 0.95)
    
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
            neo4j_connection: Neo4JConnection,
            local_import: bool
        ):
        KnowledgeGraphConstruction.__init__(self, config, neo4j_connection, local_import)

    def node_construction_pipeline(self):
        """
        Construct nodes from CSV files iteratively for all listed nodes in the directory.
        """
        logger.info("Constructing nodes from {}".format(self.nodes_path))

        nodes_csv_files = list_csv_files(pathlib.Path(self.nodes_path))
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

        rels_csv_files = list_csv_files(pathlib.Path(self.relations_path))
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
        logger.info("Knowledge graph construction is finished")

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
        node_label = self.neo4j_connection.get_node_label_from_relation(relation_label)
        
        if len(node_label) > 2:
            heads = node_label[:-1]
            tail = node_label[-1]

            for head in heads:
                textualized_relation = "{}.{}.{}".format(head, relation_label, tail)
                
                if relation_keys == []:
                    self._relation_indexing_no_property(session, relation_label, textualized_relation)
                
                else:
                    self._relation_indexing_with_property(session, relation_label, relation_keys)

        else:
            head, tail = node_label
        
            # head, tail = 
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
