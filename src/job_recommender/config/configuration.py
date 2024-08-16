from dataclasses import dataclass

from job_recommender.utils.common import read_yaml

CONFIG_PATH = "config/config.yaml"
HYPERPARAMS_PATH = "config/hyperparameters.yaml"

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

@dataclass(frozen=True)
class HyperparametersConfig:
    input_dir: str
    #Training
    seed: int
    learning_rate: float
    weight_decay: float
    patience: int
    batch_size: int
    grad_steps: int

    # Learning Rate Scheduler
    num_epochs: int
    warmup_epochs: int

    # Validation
    eval_batch_size: int

    # LLM related
    llm_model_name: str
    llm_model_path: str
    llm_frozen: bool
    llm_num_virtual_tokens: int
    output_dir: str
    max_txt_len: int
    max_new_tokens: int

    # GNN related
    gnn_model_name: str
    gnn_num_layers: int
    gnn_in_dim: int
    gnn_hidden_dim: int
    gnn_num_heads: int
    gnn_dropout: float

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
        config = self.config.neo4j_connection

        connection_config = Neo4jConfig(
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_user,
            neo4j_password=config.neo4j_password,
            neo4j_db=config.neo4j_db
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
    
    def get_hyperparameters(self) -> HyperparametersConfig:
        hp = self.hp

        hp = HyperparametersConfig(
            input_dir=hp.input_dir,
            seed=hp.seed,
            learning_rate=hp.learning_rate,
            weight_decay=hp.weight_decay,
            patience=hp.patience,
            batch_size=hp.batch_size,
            grad_steps=hp.grad_steps,
            num_epochs=hp.num_epochs,
            warmup_epochs=hp.warmup_epochs,
            eval_batch_size=hp.eval_batch_size,
            llm_model_name=hp.llm_model_name,
            llm_model_path=hp.llm_model_path,
            llm_frozen=hp.llm_frozen,
            llm_num_virtual_tokens=hp.llm_num_virtual_tokens,
            output_dir=hp.output_dir,
            max_txt_len=hp.max_txt_len,
            max_new_tokens=hp.max_new_tokens,
            gnn_model_name=hp.gnn_model_name,
            gnn_num_layers=hp.gnn_num_layers,
            gnn_in_dim=hp.gnn_in_dim,
            gnn_hidden_dim=hp.gnn_hidden_dim,
            gnn_num_heads=hp.gnn_num_heads,
            gnn_dropout=hp.gnn_dropout,
        )

        return hp