import os

from job_recommender.config.configuration import KGConstructConfig
from job_recommender import logger
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.utils.dataset import list_csv_files
from job_recommender.dataset.kg_construct import KnowledgeGraphConstruction

class KnowledgeGraphConstructionPipeline(KnowledgeGraphConstruction):
    """
    A class to run the pipeline for constructing knowledge graph
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
            path = os.path.join(self.nodes_path, nodes_csv_file)
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
            path = os.path.join(self.relations_path, rels_csv_file)
            self.load_csv_to_relations(path)
        
        logger.info("Construction from all listed relations is finished")
    
    def knowledge_graph_construction_pipeline(self):
        """
        Run the pipeline to construct knowledge graph
        """
        logger.info("Constructing knowledge graph from {}".format(self.config.input_dir))
        self.node_construction_pipeline()
        self.relation_construction_pipeline()
        logger.info("Knowledge graph construction is finisher")