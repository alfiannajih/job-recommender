import argparse

from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.utils.dataset import get_emb_model
from job_recommender import logger

def processing_nodes(neo4j_conn, node_labels, batch_size, emb_model):
    with neo4j_conn.get_session() as session:
        batch_n = 1
        node_with_embeddings = []
        for node in node_labels:
            node_keys = neo4j_conn.get_node_keys(node)
            key_statement = ", ".join(["n.{}".format(key) for key in node_keys])
            
            logger.info("Start processing node: [{}]".format(node))

            result = session.run(
                """
                MATCH (n:{})
                RETURN {}
                """.format(node, key_statement)
            )

            node_counts = session.run(
                """
                MATCH (n:{})
                RETURN COUNT(n)
                """.format(node)
            ).value()[0]
            
            logger.info("Node [{}] contains {} nodes and keys: [{}]".format(node, node_counts, key_statement))
            
            for i, record in enumerate(result):
                rec = {k: record.get("n.{}".format(k)) for k in node_keys}
                textualized_prop = "\n".join(["{}: {}".format(k, v) for k, v in rec.items()])
                
                rec.update(
                    {"embedding": emb_model.encode(
                        "{}\n{}".format(node, textualized_prop),
                        show_progress_bar=False
                    )}
                )

                node_with_embeddings.append(rec)
                
                if i%batch_size == 0:
                    import_batch(session, node_with_embeddings, node, node_keys)
                    logger.info("[{}] Processed batch {}".format(node, batch_n))
                    
                    node_with_embeddings = []
                    batch_n += 1
                
                elif i == node_counts-1:
                    import_batch(session, node_with_embeddings, node, node_keys)
                    logger.info("[{}] Processed batch {}".format(node, batch_n))

            batch_n = 1
            
            neo4j_conn.create_vector_index(node, 384)

            logger.info("Processing node [{}] is finished".format(node))

def main(args):
    neo4j_conn = Neo4JConnection(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        db=args.neo4j_db
    )
    
    emb_model = get_emb_model(args.embedding_model_path)

    node_labels = neo4j_conn.get_node_labels()
    rel_labels = neo4j_conn.get_relation_labels()

    processing_nodes(
        neo4j_conn=neo4j_conn,
        node_labels=node_labels,
        batch_size=args.batch_size,
        emb_model=emb_model
    )
                
    '''result = session.run('MATCH (j:JobTitle) RETURN j.name AS name')
        for record in result:
            name = record.get('name')

            # Create embedding for title and plot
            if name is not None:
                job_title_with_embeddings.append({
                    'name': name,
                    'embedding': model.encode(f"Job Title: {name}"),
                })

            # Import when a batch of movies has embeddings ready; flush buffer
            if len(job_title_with_embeddings) == batch_size:
                import_batch(driver, job_title_with_embeddings, batch_n)
                job_title_with_embeddings = []
                batch_n += 1

    # Import complete, show counters
    records, _, _ = driver.execute_query('''
    #MATCH (j:JobTitle WHERE j.embedding IS NOT NULL)
    #RETURN count(*) AS countJobTitleWithEmbeddings, size(j.embedding) AS embeddingSize
    ''', database_=DB_NAME)

    driver.execute_query("""
    CREATE VECTOR INDEX jobTitles
    FOR (j:JobTitle)
    ON j.embedding
    OPTIONS {indexConfig: {
        `vector.dimensions`: 1024,
        `vector.similarity_function`: 'cosine'
    }}
    """)

    print(f"""
Embeddings generated and attached to nodes.
Job Title nodes with embeddings: {records[0].get('countJobTitleWithEmbeddings')}.
Embedding size: {records[0].get('embeddingSize')}.
    """)'''


def import_batch(session, nodes_with_embeddings, node_label, node_keys):
    # Add embeddings to Movie nodes
    match_statement = ", ".join(["{}: node.{}".format(k, k) for k in node_keys])

    session.run('''
    UNWIND $node_dict as node
    MATCH (n:{} {{{}}})
    CALL db.create.setNodeVectorProperty(n, 'embedding', node.embedding)
    '''.format(node_label, match_statement), node_dict=nodes_with_embeddings, database_=args.neo4j_db)
    #print(f'Processed batch {batch_n}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_model_path", type=str, help="Path of embedding model.")
    parser.add_argument("--batch_size", type=int, help="Size of data per batch")
    parser.add_argument("--neo4j_uri", type=str, help="Neo4j URI.")
    parser.add_argument("--neo4j_db", type=str, help="Neo4j database.")
    parser.add_argument("--neo4j_user", type=str, help="Neo4j user.")
    parser.add_argument("--neo4j_password", type=str, help="Neo4j password.")

    args = parser.parse_args()

    main(args)