import argparse
from job_recommender.pipeline.kg_construct import KnowledgeGraphConstructionPipeline
from job_recommender.config.configuration import ConfigurationManager
from job_recommender.dataset.neo4j_connection import Neo4JConnection

def main(args):
    config = ConfigurationManager()
    kg_construct_config = config.get_kg_construct_config()
    
    neo4j_connection_config = config.get_neo4j_connection_config()
    neo4j_connection = Neo4JConnection(neo4j_connection_config)
    
    kg_construct_pipeline = KnowledgeGraphConstructionPipeline(kg_construct_config, neo4j_connection)
    kg_construct_pipeline.knowledge_graph_construction_pipeline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)