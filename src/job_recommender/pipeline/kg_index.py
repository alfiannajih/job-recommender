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