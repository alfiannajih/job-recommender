import pathlib

from job_recommender.config.configuration import KGConstructConfig
from job_recommender import logger
from job_recommender.dataset.neo4j_connection import Neo4JConnection
from job_recommender.utils.dataset import (
    count_csv_rows,
    get_label_from_path,
    get_csv_header
)
from job_recommender.utils.common import (
    read_json
)

# Absolute path of the project folder
ABSOLUTE_PATH = pathlib.Path(".").resolve()

class KnowledgeGraphConstruction:
    """
    A class to construct knowledge graphs from CSV files.

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
        """
        Initializes the instance with its configuration and Neo4j Connection.

        Args:
            config (KGConstructConfig): Configuration setting to constructing knowledge graph.
            neo4j_connection (Neo4jConnection): Connection instance to Neo4j database.
        """
        self.config = config
        self.neo4j_connection = neo4j_connection

        self.nodes_path = pathlib.Path(self.config.input_dir, "nodes")
        self.relations_path = pathlib.Path(self.config.input_dir, "relations")
        self.relations_map_path = pathlib.Path(self.relations_path, "rel_mapping.json")

    def load_csv_to_nodes(self, path: str):
        """
        Load CSV files to construct nodes in Neo4j database.

        Args:
            path (str): Path of the CSV file.
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
            path (str): Path of the CSV file.
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