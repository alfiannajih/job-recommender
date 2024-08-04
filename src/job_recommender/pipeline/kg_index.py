from job_recommender import logger
from job_recommender.dataset.kg_index import KnowledgeGraphIndexing
from job_recommender.config.configuration import KGIndexingConfig
from job_recommender.dataset.neo4j_connection import Neo4JConnection

class KnowledgeGraphIndexingPipeline(KnowledgeGraphIndexing):
    def __init__(
            self,
            config: KGIndexingConfig,
            neo4j_connection: Neo4JConnection
        ):
        KnowledgeGraphIndexing.__init__(self, config, neo4j_connection)
    
    def single_node_label_indexing(self, session, node_label):
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