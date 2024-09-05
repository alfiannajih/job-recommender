import os
from dotenv import load_dotenv
from dataclasses import dataclass, asdict

from job_recommender.utils.common import read_yaml

CONFIG_PATH = "config/config.yaml"
HYPERPARAMS_PATH = "config/hyperparameters.yaml"

load_dotenv()

@dataclass(frozen=True)
class Neo4jConfig:
    """
    A class to store configuration for connecting to neo4j database.

    Attributes:
        neo4j_uri (str): The URI for the neo4j connection.
        neo4j_user (str): The username for the neo4j connection.
        neo4j_password (str): The password for the neo4j connection.
        neo4j_db (str): The database name for the neo4j connection.
    """
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_db: str

@dataclass(frozen=True)
class KGConstructConfig(Neo4jConfig):
    """
    A class to store configuration for constructing knowledge graphs.

        Attributes:
            neo4j_uri (str): The URI for the neo4j connection.
            neo4j_user (str): The username for the neo4j connection.
            neo4j_password (str): The password for the neo4j connection.
            neo4j_db (str): The database name for the neo4j connection.
            input_dir (str): The directory containing input files for constructing the knowledge graph.
    """
    input_dir: str

@dataclass(frozen=True)
class KGIndexingConfig(Neo4jConfig):
    """
    A class to store configuration for indexing nodes and relations of knowledge graph.

        Attributes:
            neo4j_uri (str): The URI for the neo4j connection.
            neo4j_user (str): The username for the neo4j connection.
            neo4j_password (str): The password for the neo4j connection.
            neo4j_db (str): The database name for the neo4j connection.
            input_dir (str): The directory containing input files for constructing the knowledge graph.
            embedding_model (str): Path of embedding model (could be local path or huggingface path).
            batch_size (int): Size of the data to be indexed for each batch.
    """
    embedding_model: str
    batch_size: int

@dataclass(frozen=True)
class KGRetrievalConfig(Neo4jConfig):
    """
    A class to store configuration to retrieve nodes and relations of knowledge graph.

        Attributes:
            neo4j_uri (str): The URI for the neo4j connection.
            neo4j_user (str): The username for the neo4j connection.
            neo4j_password (str): The password for the neo4j connection.
            neo4j_db (str): The database name for the neo4j connection.
            input_dir (str): The directory containing input files for constructing the knowledge graph.
            embedding_model (str): Path of embedding model (could be local path or huggingface path).
            rerank_model (str): Path of rerank model (could be local path or huggingface path).
    """
    embedding_model: str
    rerank_model: int

@dataclass(frozen=True)
class RawDatasetConfig:
    raw_path: str
    preprocessed_path: str

@dataclass(frozen=True)
class ResumeDatasetConfig:
    input_dir: str
    resume_dir: str

@dataclass(frozen=True)
class SyntheticDatasetConfig:
    desc_prompt_system: str
    desc_prompt_user: str
    desc_path: str
    desc_request_count: int
    desc_model: str
    resume_prompt_system: str
    resume_path: str
    resume_model: str
    feedback_prompt_system: str
    feedback_path: str
    feedback_model: str

class ConfigurationManager:
    """
    A class to manage configuration settings.

    Attributes:
        config (dict): The configuration settings loaded from a YAML file.
    """
    def __init__(
            self,
            config_path: str = CONFIG_PATH,
            hyperparams_path: str = HYPERPARAMS_PATH
        ):
        """
        Initializes the instance with config.yaml.
        
        Args:
            config_path (str): Path of config.yaml
        """
        self.config = read_yaml(config_path)
        self.hp = read_yaml(hyperparams_path)

    def get_neo4j_connection_config(self) -> Neo4jConfig:
        """
        Retrieves the configuration of Neo4j connection.

        Returns:
            Neo4jConfig: An instance of Neo4jConfig with the connection details.
        """

        connection_config = Neo4jConfig(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_db=os.getenv("NEO4J_DB")
        )

        return connection_config
    
    def get_kg_construct_config(self) -> KGConstructConfig:
        """
        Retrieves the KGConstructConfig for constructing knowledge graphs.

        Returns:
            KGConstructConfig: An instance of KGConstructConfig with the specified configuration.
        """
        config = self.config.kg_construct
        connection_config = self.get_neo4j_connection_config()

        kg_construct_config = KGConstructConfig(
            neo4j_uri=connection_config.neo4j_uri,
            neo4j_user=connection_config.neo4j_user,
            neo4j_password=connection_config.neo4j_password,
            neo4j_db=connection_config.neo4j_db,
            input_dir=config.input_dir
        )

        return kg_construct_config
    
    def get_kg_indexing_config(self) -> KGIndexingConfig:
        """
        Retrieves the KGIndexingConfig for indexing nodes and relationships of knowledge graphs.

        Returns:
            KGIndexingConfig: An instance of KGIndexingConfig with the specified configuration.
        """
        config = self.config.kg_indexing
        connection_config = self.get_neo4j_connection_config()

        kg_indexing_config = KGIndexingConfig(
            neo4j_uri=connection_config.neo4j_uri,
            neo4j_user=connection_config.neo4j_user,
            neo4j_password=connection_config.neo4j_password,
            neo4j_db=connection_config.neo4j_db,
            embedding_model=config.embedding_model,
            batch_size=config.batch_size
        )

        return kg_indexing_config
    
    def get_kg_retrieval_config(self) -> KGRetrievalConfig:
        """
        Retrieves the KGRetrievalConfig for indexing nodes and relationships of knowledge graphs.

        Returns:
            KGRetrievalConfig: An instance of KGRetrievalConfig with the specified configuration.
        """
        config = self.config.kg_retrieval
        connection_config = self.get_neo4j_connection_config()

        kg_retrieval_config = KGRetrievalConfig(
            neo4j_uri=connection_config.neo4j_uri,
            neo4j_user=connection_config.neo4j_user,
            neo4j_password=connection_config.neo4j_password,
            neo4j_db=connection_config.neo4j_db,
            embedding_model=config.embedding_model,
            rerank_model=config.rerank_model
        )

        return kg_retrieval_config
    
    def get_raw_dataset_config(self) -> RawDatasetConfig:
        config = self.config.raw_dataset

        raw_dataset_config = RawDatasetConfig(
            raw_path=config.raw_path,
            preprocessed_path=config.preprocessed_path
        )

        return raw_dataset_config

    def get_resume_dataset_config(self) -> ResumeDatasetConfig:
        config = self.config.resume_dataset

        resume_dataset_config = ResumeDatasetConfig(
            input_dir=config.input_dir,
            resume_dir=config.resume_dir
        )

        return resume_dataset_config

    def get_synthetic_dataset_config(self) -> SyntheticDatasetConfig:
        config = self.config.synthetic_dataset

        synthetic_dataset_config = SyntheticDatasetConfig(
            desc_prompt_system=config.desc_prompt_system,
            desc_prompt_user=config.desc_prompt_user,
            desc_path=config.desc_path,
            desc_request_count=config.desc_request_count,
            desc_model=config.desc_model,
            resume_prompt_system=config.resume_prompt_system,
            resume_path=config.resume_path,
            resume_model=config.resume_model,
            feedback_prompt_system=config.feedback_prompt_system,
            feedback_path=config.feedback_path,
            feedback_model=config.feedback_model        
        )
            
        return synthetic_dataset_config