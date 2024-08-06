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

    def import_batch_relations(self, session, rel_with_embeddings, relation_label, relation_keys, batch_n):
        # Add embeddings to nodes
        match_statement = ", ".join(["{}: relation.{}".format(k, k) for k in relation_keys])
        
        session.run("""
            UNWIND $relation_dict as relation
            MATCH ()-[r:{} {{{}}}]->()
            CALL db.create.setRelationshipVectorProperty(r, 'embedding', relation.embedding)
            """.format(relation_label, match_statement),
            relation_dict=rel_with_embeddings,
            database_=self.neo4j_connection.db
        )

        logger.info("[{}] Processed batch {}".format(relation_label, batch_n))

    def embed_property(self, label, property):
        textualized_property = textualize_property(property)
        
        property.update(
            {"embedding": self.embedding_model.encode(
                "{}\n{}".format(label, textualized_property),
                show_progress_bar=False
            )}
        )

        return property