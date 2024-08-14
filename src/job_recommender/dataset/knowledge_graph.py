import pathlib
import os
import pandas as pd

from job_recommender.config.configuration import (
    KGConstructConfig,
    KGIndexingConfig,
    KGRetrievalConfig,
    RawDatasetConfig
)
from job_recommender import logger
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.utils.dataset import (
    count_csv_rows,
    get_label_from_path,
    get_csv_header,
    textualize_property
)
from job_recommender.utils.common import (
    read_json,
    get_emb_model,
    get_rerank_model
)

# Absolute path of the project folder
ABSOLUTE_PATH = pathlib.Path(".").resolve()

class PrepareRawDataset:
    def __init__(
            self,
            config: RawDatasetConfig
        ):
        self.config = config
        
        self.posting_path = os.path.join(self.config.raw_path, "postings.csv")
        self.job_industries_path = os.path.join(self.config.raw_path, "jobs/job_industries.csv")

        self.companies_profile_path = os.path.join(self.config.raw_path, "companies/companies.csv")
        self.company_industries_path = os.path.join(self.config.raw_path, "companies/company_industries.csv")
        self.company_specialities_path = os.path.join(self.config.raw_path, "companies/company_specialities.csv")

        self.mappings_industries_path = os.path.join(self.config.raw_path, "mappings/industries.csv")

    def process_node_job_title(self, posting_df, threshold=50):
        # Filter job title
        job_filter = (posting_df["title"].value_counts() > threshold).to_dict()
        job_mask = posting_df["title"].apply(lambda x: job_filter[x])

        # Lower case job title
        job_nodes = posting_df[job_mask][["title"]]
        job_nodes["title"] = job_nodes["title"].str.lower()
        job_nodes = job_nodes.drop_duplicates().reset_index(drop=True).rename({"title": "main_prop"}, axis=1)

        # Save job title nodes and ids
        job_nodes_path = os.path.join(self.config.preprocessed_path, "nodes/JobTitle.csv")
        job_nodes.to_csv(job_nodes_path, index_label="prop_id")

    def process_node_industry(self, job_industry_df, company_industry_df, threshold=20):
        # Filter industry for company
        comp_industry = company_industry_df[company_industry_df > threshold].index
        comp_industry_nodes = pd.DataFrame(comp_industry)
        comp_industry_nodes["industry"] = comp_industry_nodes["industry"].str.lower()
        comp_industry_nodes.drop_duplicates().reset_index(drop=True)

        # Lower case job_industry_df
        job_industry_df["industry_name"] = job_industry_df["industry_name"].str.lower()
        job_industry_nodes = job_industry_df.drop_duplicates().reset_index(drop=True)

        # Combine company industry and job industry
        concat_industry = pd.concat([comp_industry_nodes, job_industry_nodes.rename({"industry_name": "industry"}, axis=1)], axis=0).drop_duplicates().reset_index(drop=True).rename({"industry": "main_prop"}, axis=1)

        # Save industry nodes and ids
        industry_nodes_path = os.path.join(self.config.preprocessed_path, "nodes/Industry.csv")
        concat_industry.to_csv(industry_nodes_path, index_label="prop_id")

    def process_node_company(self, company_df, posting_df, threshold=20):
        # Filter company
        company_nodes = company_df.copy().reset_index()

        company_filter = posting_df["company_id"].value_counts()
        company_mask = company_filter[company_filter > threshold].index

        company_nodes["company_id"] = company_nodes["company_id"].apply(lambda x: x if x in company_mask else None)
        company_nodes = company_nodes.dropna().astype({"company_id": int})
        company_nodes = company_nodes.set_index("company_id").rename({"name": "main_prop"}, axis=1)

        # Save company nodes and ids
        company_nodes_path = os.path.join(self.config.preprocessed_path, "nodes/Company.csv")
        company_nodes.to_csv(company_nodes_path, index_label="prop_id")

    def process_node_company_speciality(self, company_specialization_df, threshold=20):
        # Filter company specialization
        comp_specialization = company_specialization_df[company_specialization_df > threshold].index
        comp_specialization_nodes = pd.DataFrame(comp_specialization)

        # Lower case specialization
        comp_specialization_nodes["speciality"] = comp_specialization_nodes["speciality"].str.lower()
        comp_specialization_nodes = comp_specialization_nodes.drop_duplicates().reset_index(drop=True).rename({"speciality": "main_prop"}, axis=1)

        # Save company specialization and ids
        company_speciality_nodes_path = os.path.join(self.config.preprocessed_path, "nodes/Speciality.csv")
        comp_specialization_nodes.to_csv(company_speciality_nodes_path, index_label="prop_id")

    def process_node_company_size(self):
        # Categorized company size
        company_size_nodes = pd.DataFrame(["Small Company", "Medium Company", "Large Company"], columns=["main_prop"])

        # Save company size and ids
        company_size_path = os.path.join(self.config.preprocessed_path, "nodes/Size.csv")
        company_size_nodes.to_csv(company_size_path, index_label="prop_id")

    def process_rel_job_company(self, job_nodes, company_nodes, posting_df):
        # Create id: job map
        job_map = {v: k for k, v in job_nodes["main_prop"].to_dict().items()}
        
        # Create relations
        job_comp_rel = posting_df.copy()
        breakpoint()
        job_comp_rel["title"] = job_comp_rel["title"].apply(lambda x: job_map.get(x.lower(), None))
        job_comp_rel["company_id"] = job_comp_rel["company_id"].apply(lambda x: x if x in company_nodes.index else None)
        job_comp_rel = job_comp_rel.dropna(subset=["title", "company_id"]).astype({"title": int, "company_id": int})
        job_comp_rel = job_comp_rel.rename({"title": "h_id", "company_id": "t_id"}, axis=1)[["h_id", "location", "description", "t_id"]]
        job_comp_rel["description"] = job_comp_rel["description"].apply(lambda x: x[:2000])
        
        # Keep 5 first duplicated rows
        mask = job_comp_rel.duplicated(subset=["h_id", "t_id"], keep=False)
        duplicated_rows = job_comp_rel[mask]
        unique_rows = job_comp_rel[~mask]
        selective_duplicate_rows = duplicated_rows.groupby(["h_id", "t_id"]).head(5)
        job_comp_rel = pd.concat([unique_rows, selective_duplicate_rows])

        # Save job-company relations
        job_comp_path = os.path.join(self.config.preprocessed_path, "relations/offered_by.csv")
        job_comp_rel.to_csv(job_comp_path)
    
    def process_rel_job_industry(self, job_industry_df, industry_nodes, job_comp_rel):
        # Map industry to id and vice versa
        ind_to_id = {k: v for v, k in industry_nodes["main_prop"].to_dict().items()}
        id_to_ind = industry_nodes["main_prop"].to_dict()

        job_ind = job_industry_df.copy()
        job_ind["industry_id"] = job_ind["industry_id"].apply(lambda x: id_to_ind.get(x, None))
        job_to_industry = job_ind["industry_id"].apply(lambda x: ind_to_id.get(x, None)).to_dict()

        # Create relations
        job_industry_rel = job_comp_rel.copy()[["h_id"]]

        job_industry_rel.h_id = job_comp_rel.index
        job_industry_rel.index = job_comp_rel.h_id

        job_industry_rel["t_id"] = job_industry_rel.h_id.apply(lambda x: job_to_industry.get(x, None))
        job_industry_rel = job_industry_rel.dropna().astype({"t_id": int})

        job_industry_rel = job_industry_rel[["t_id"]].reset_index()

        # Save job-industry relation
        job_industry_path = os.path.join(self.config.preprocessed_path, "relations/belongs_to_industry.csv")
        job_industry_rel.to_csv(job_industry_path, index=False)

        job_comp_rel_path = os.path.join(self.config.preprocessed_path, "relations/offered_by.csv")
        job_comp_rel.to_csv(job_comp_rel_path, index=False)
    
    def process_rel_company_industry(self, industry_nodes, company_industries_df, company_nodes):
        # Map industry to id and vice versa
        ind_to_id = {k: v for v, k in industry_nodes["main_prop"].to_dict().items()}

        # Create relations
        comp_industry_rel = company_industries_df.copy().reset_index()

        comp_industry_rel["company_id"] = comp_industry_rel["company_id"].apply(lambda x: x if x in company_nodes.index else None)
        comp_industry_rel["industry"] = comp_industry_rel["industry"].apply(lambda x: ind_to_id.get(x.lower(), None))

        comp_industry_rel = comp_industry_rel.dropna().rename({"company_id": "h_id", "industry": "t_id"}, axis=1).astype(int)
        
        # Save company-industry relation
        company_industry_path = os.path.join(self.config.preprocessed_path, "relations/part_of_industry.csv")
        comp_industry_rel.to_csv(company_industry_path, index=False)

    def process_rel_company_speciality(self, comp_specialization_nodes, company_specialities_df, company_nodes):
        # Map speciality to id
        spec_to_id = {k: v for v, k in comp_specialization_nodes["main_prop"].to_dict().items()}

        # Create relations
        comp_spec_rel = company_specialities_df.copy().reset_index()

        comp_spec_rel["company_id"] = comp_spec_rel["company_id"].apply(lambda x: x if x in company_nodes["main_prop"].index else None)
        comp_spec_rel = comp_spec_rel.dropna()

        comp_spec_rel["speciality"] = comp_spec_rel["speciality"].apply(lambda x: spec_to_id.get(x.lower(), None))
        comp_spec_rel = comp_spec_rel.dropna()

        comp_spec_rel = comp_spec_rel.astype(int).rename({"company_id": "h_id", "speciality": "t_id"}, axis=1)

        # Save company-speciality relations
        company_spec_path = os.path.join(self.config.preprocessed_path, "relations/specialized_in.csv")
        comp_spec_rel.to_csv(company_spec_path, index=False)
    
    def process_rel_company_size(self, company_size_nodes, companies_profile_df, company_nodes):
        # Map company size
        map_size = {
            1: "Small Company",
            2: "Small Company",
            3: "Medium Company",
            4: "Medium Company",
            5: "Medium Company",
            6: "Large Company",
            7: "Large Company"
        }
        size_to_id  = {k: v for v, k in company_size_nodes["main_prop"].to_dict().items()}

        # Create relation
        comp_size_rel = companies_profile_df.reset_index()[["company_id", "company_size"]]
        comp_id = company_nodes.index
        comp_size_rel["company_id"] = comp_size_rel["company_id"].apply(lambda x: x if x in comp_id else None)
        comp_size_rel["company_size"] = comp_size_rel["company_size"].apply(lambda x: map_size.get(x, None))

        comp_size_rel = comp_size_rel.dropna()

        comp_size_rel["company_size"] = comp_size_rel["company_size"].apply(lambda x: size_to_id[x])
        comp_size_rel = comp_size_rel.astype(int).rename({"company_id": "h_id", "company_size": "t_id"}, axis=1)

        # Save company-size relation
        company_size_path = os.path.join(self.config.preprocessed_path, "relations/categorized_as.csv")
        comp_size_rel.to_csv(company_size_path, index=False)

class KnowledgeGraphConstruction:
    """
    A class to construct knowledge graphs from CSV files.

    Args:
        config (KGConstructConfig): Configuration setting to constructing knowledge graph.
        neo4j_connection (Neo4jConnection): Connection instance to Neo4j database.

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
        self.config = config
        self.neo4j_connection = neo4j_connection

        self.nodes_path = pathlib.Path(self.config.input_dir, "nodes")
        self.relations_path = pathlib.Path(self.config.input_dir, "relations")
        self.relations_map_path = pathlib.Path(self.relations_path, "rel_mapping.json")

    def load_csv_to_nodes(self, path: str):
        """
        Load CSV files to construct nodes in Neo4j database.

        Args:
            path (str): Path of the nodes CSV file.
        """
        # Get node label from the filepath
        label = get_label_from_path(path)
        logger.info("Constructing node: [{}]".format(label))
        logger.info("{}.csv contains {} rows".format(label, count_csv_rows(path)))

        # Create MERGE statement to construct nodes in Neo4j
        merge_statement = self._create_merge_node_statement(path, label)
        
        # Get absolute path of the csv file
        csv_path = pathlib.Path(ABSOLUTE_PATH, path)
        csv_path = str(pathlib.PurePosixPath(csv_path)).replace(" ", "%20")

        # Load the CSV files to create nodes in Neo4j Database
        with self.neo4j_connection.get_session() as session:
            session.run(
                """
                LOAD CSV WITH HEADERS FROM 'file:///{}' AS row
                {}
                """.format(csv_path, merge_statement))
        logger.info("Construction nodes [{}] is finished".format(label))

    def load_csv_to_relations(self, path: str):
        """
        Load CSV files to construct relations in Neo4j database.

        Args:
            path (str): Path of the relations CSV file.
        """
        # Get relation label and relation map from the filepath
        label = get_label_from_path(path)
        relations_map = self._get_relations_map()
        logger.info("Constructing relation: [{}]".format(label))
        logger.info("{}.csv contains {} rows".format(label, count_csv_rows(path)))

        # Create MATCH and MERGE statement to construct relations in Neo4j
        match_statement = self._create_match_statement(relations_map[label])
        merge_statement = self._create_merge_relation_statement(path, label)

        # Get absolute path of the csv file
        csv_path = pathlib.Path(ABSOLUTE_PATH, path)
        csv_path = str(pathlib.PurePosixPath(csv_path)).replace(" ", "%20")

        # Load the CSV files to create relations in Neo4j Database
        with self.neo4j_connection.get_session() as session:
            session.run(
                """
                LOAD CSV WITH HEADERS FROM 'file:///{}' AS row
                {}
                {}
                """.format(csv_path, match_statement, merge_statement))
        
        logger.info("Construction relations [{}] is finished".format(label))

    def _get_relations_map(self) -> dict:
        """
        Load relation map that map relation label with their connected nodes

        Args:
            dict: Dictionary that contains keys: relation label and their values: head and tail nodes.
        """
        return read_json(self.relations_map_path)
    
    def _create_merge_node_statement(
            self,
            path: str,
            node_label: str
        ) -> str:
        """
        Creates a Cypher MERGE statement for creating nodes in Neo4j if not exists.

        Args:
            path (str): The path to the CSV file containing node properties.
            node_label (str): The label to assign to the node in Neo4j.

        Returns:
            str: A Cypher MERGE statement for creating nodes with the specified properties if not exists.
        """
        # Get the sub property of nodes from CSV header
        headers = get_csv_header(path)
        headers.remove("prop_id")
        headers.remove("main_prop")

        # Create a statement to construct the sub property if it exists
        sub_prop = ""
        if len(headers) > 0:
            sub_prop += ", ".join(["{}: row.{}".format(x, x) for x in headers])

        # Concat main property and sub property statement
        main_prop = "id: row.prop_id, name: row.main_prop"
        full_prop = main_prop + sub_prop

        statement = "MERGE (n: {} {{{}}})"\
            .format(node_label, full_prop)
        
        return statement

    def _create_match_statement(
            self,
            relation_map: dict
        ) -> str:
        """
        Creates a Cypher MATCH statement for searching head (h) and tail (t) nodes in Neo4j based on the its label and id.

        Args:
            relation_map (dict): Dictionary that consist of head and tail node label connected with its relation.

        Returns:
            str: A Cypher MATCH statement for searching nodes in Neo4j database.
        """
        # Get the head and tail node label
        head_label = relation_map["head"]
        tail_label = relation_map["tail"]

        # Create match statement to match the head and tail node label based on their properties
        statement = "OPTIONAL MATCH (h: {} {{id: row.h_id}}), (t: {} {{id: row.t_id}})".format(head_label, tail_label)

        return statement
    
    def _create_merge_relation_statement(
            self,
            path: str,
            rel_label: str
        ) -> str:
        """
        Creates a Cypher MERGE statement for creating relations in Neo4j if not exists.

        Args:
            path (str): The path to the CSV file containing relation properties.
            rel_label (str): The label to assign to the relation in Neo4j.

        Returns:
            str: A Cypher MERGE statement for creating relations with the specified properties if not exists.
        """
        # Get the sub property of relations from CSV header
        headers = get_csv_header(path)
        headers.remove("h_id")
        headers.remove("t_id")

        # Create a statement to construct the sub property if it exists
        sub_prop = ""
        if len(headers) > 0:
            sub_prop += ", ".join(["{}: row.{}".format(x, x) for x in headers])

        statement = "MERGE (h)-[r: {} {{{}}}]->(t)"\
            .format(rel_label, sub_prop)
        
        return statement
    
class KnowledgeGraphIndexing:
    """
    A class for indexing the property of nodes and relations in Neo4j database.

    Args:
        config (KGConstructConfig): Configuration setting to constructing knowledge graph.
        neo4j_connection (Neo4jConnection): Connection instance to Neo4j database.

    Attributes:
        config (KGConstructConfig): Configuration setting to constructing knowledge graph.
        neo4j_connection (Neo4jConnection): Connection instance to Neo4j database.
        embedding_model (SentenceTransformer): Embedding model from sentence_transformers library
    """
    def __init__(
            self,
            config: KGIndexingConfig,
            neo4j_connection: Neo4JConnection
        ):
        self.config = config
        self.neo4j_connection = neo4j_connection

        self.embedding_model = get_emb_model(self.config.embedding_model)
    
    def import_batch_nodes(self, session, nodes_with_embeddings, node_label, node_keys, batch_n):
        """
        Insert vector representation of the node's property in a batch.

        Args:
            session (Session): Neo4j session for inserting 
            nodes_with_embeddings (List):
            node_label (str):
            node_keys (List):
            batch_n (int):
        """
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

    def import_batch_relations(self, session, relation_with_embeddings, relation_label, relation_keys, batch_n):
        """
        Insert vector representation of the relation's property in a batch.

        Args:
            session (Session)
            relation_with_embeddings (List)
            relation_label (str)
            relation_keys (List)
            batch_n (int)
        """
        # Add embeddings to nodes
        match_statement = ", ".join(["{}: relation.{}".format(k, k) for k in relation_keys])
        
        session.run("""
            UNWIND $relation_dict as relation
            MATCH ()-[r:{} {{{}}}]->()
            CALL db.create.setRelationshipVectorProperty(r, 'embedding', relation.embedding)
            """.format(relation_label, match_statement),
            relation_dict=relation_with_embeddings,
            database_=self.neo4j_connection.db
        )

        logger.info("[{}] Processed batch {}".format(relation_label, batch_n))

    def embed_property(self, label, property):
        """
        Args:
            label (str):
            property (dict): 
        """
        textualized_property = textualize_property(property)
        
        property.update(
            {"embedding": self.embedding_model.encode(
                "{}\n{}".format(label, textualized_property),
                show_progress_bar=False
            )}
        )

        return property

class KnowledgeGraphRetrieval:
    def __init__(self, config: KGRetrievalConfig, neo4j_connection: Neo4JConnection):
        logger.info("Initialize knowledge graph retriever")
        self.config = config
        self.neo4j_connection = neo4j_connection
        
        self.embedding_model = get_emb_model(self.config.embedding_model)
        self.rerank_model = get_rerank_model(self.config.rerank_model)

    def query_relationship_from_node(self, query, n_query):
        similar_relations = self.neo4j_connection.driver.execute_query(
            """
            CALL db.index.vector.queryNodes('JobTitleIndex', {}, {})
            YIELD node, score
            MATCH p=(node)-[r:offered_by]->(connectedNode)
            RETURN elementId(r) AS id, r.job_qualification, r.job_responsibility, r.location
            """.format(n_query, query)
        )
        
        relations = []
        for relation in similar_relations.records:
            _id = relation.get("id")
            text = "Job qualification: {}\nJob responsibility: {}\nJob location: {}".format(relation.get("r.job_qualification"), relation.get("r.job_responsibility"), relation.get("r.location"))

            relations.append({"rel_id": _id, "text": text})
        
        return relations
    
    def rerank_retrieved_relationship(self, retrieved, query, top_k):
        documents = [t["text"] for t in retrieved]
        
        results = self.rerank_model.rank(query, documents, top_k=top_k)
        rel_ids = [retrieved[i["corpus_id"]]["rel_id"] for i in results]

        return rel_ids