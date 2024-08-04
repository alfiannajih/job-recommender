from job_recommender import logger
from job_recommender.config.configuration import KGIndexingConfig
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.utils.common import (
    get_emb_model
)
from job_recommender.utils.dataset import textualize_property

class KnowledgeGraphIndexing:
    def __init__(
            self,
            config: KGIndexingConfig,
            neo4j_connection: Neo4JConnection
        ):
        self.config = config
        self.neo4j_connection = neo4j_connection

        self.embedding_model = get_emb_model(self.config.embedding_model)
    
    def import_batch_nodes(self, session, nodes_with_embeddings, node_label, node_keys, batch_n):
        # Add embeddings to nodes
        match_statement = ", ".join(["{}: node.{}".format(k, k) for k in node_keys])

        session.run("""
        UNWIND $node_dict as node
        MATCH (n:{} {{{}}})
        CALL db.create.setNodeVectorProperty(n, 'embedding', node.embedding)
        """.format(node_label, match_statement),
            node_dict=nodes_with_embeddings,
            database_=self.neo4j_connection.db
        )
        
        logger.info("[{}] Processed batch {}".format(node_label, batch_n))

    def embed_property(self, node_label, property):
        textualized_property = textualize_property(property)
        
        property.update(
            {"embedding": self.embedding_model.encode(
                "{}\n{}".format(node_label, textualized_property),
                show_progress_bar=False
            )}
        )

        return property