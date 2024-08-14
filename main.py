import argparse
import sys
import os
import pathlib

sys.path.append(str(pathlib.Path(os.path.dirname(os.path.realpath(__file__)), "src")))

from job_recommender import logger
from job_recommender.pipeline.knowledge_graph import (
    KnowledgeGraphConstructionPipeline,
    KnowledgeGraphIndexingPipeline,
    PrepareRawDatasetPipeline
)
from job_recommender.config.configuration import ConfigurationManager
from job_recommender.dataset.neo4j_connection import Neo4JConnection

def main(args):
    config = ConfigurationManager()
    
    neo4j_connection_config = config.get_neo4j_connection_config()
    neo4j_connection = Neo4JConnection(neo4j_connection_config)

    stage = 0
    if args.preprocess_raw_dataset:
        logger.info("-------Stage {}: Preprocess Raw Dataset-------".format(stage))
        raw_dataset_config = config.get_raw_dataset_config()
        
        raw_dataset_pipeline = PrepareRawDatasetPipeline(raw_dataset_config)
        raw_dataset_pipeline.node_preprocess_pipeline()
        raw_dataset_pipeline.relation_preprocess_pipeline()
        stage += 1

    if args.construct_kg:
        logger.info("-------Stage {}: Constructing Knowledge Graph-------".format(stage))
        kg_construct_config = config.get_kg_construct_config()
        
        kg_construct_pipeline = KnowledgeGraphConstructionPipeline(kg_construct_config, neo4j_connection)
        kg_construct_pipeline.knowledge_graph_construction_pipeline()
        stage += 1
    
    if args.index_kg:
        logger.info("-------Stage {}: Indexing Knowledge Graph-------".format(stage))
        kg_indexing_config = config.get_kg_indexing_config()
        
        kg_indexing_pipeline = KnowledgeGraphIndexingPipeline(kg_indexing_config, neo4j_connection)
        kg_indexing_pipeline.knowledge_graph_indexing_pipeline()
        stage += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--preprocess_raw_dataset", action="store_true", help="")
    parser.add_argument("--construct_kg", action="store_true", help="")
    parser.add_argument("--index_kg", action="store_true", help="")

    args = parser.parse_args()

    main(args)