from transformers import PretrainedConfig, LlamaConfig
from typing import List

class GRetrieverConfig(PretrainedConfig):
    def __init__(
        self,
        llm_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        bos_id: List = [128000, 128000, 128006, 882, 128007],
        max_txt_len: int = 1024,
        max_new_tokens: int = 256,
        gnn_num_layers: int = 4,
        gnn_in_dim: int = 768,
        gnn_hidden_dim: int = 1024,
        gnn_num_heads: int = 4,
        gnn_dropout: int = 0,
        **kwargs
    ):
        # LLM Related
        self.llama_config = LlamaConfig.from_pretrained(llm_path)
        self.llama_config.pad_token_id = self.llama_config.eos_token_id[-1]

        self.bos_id = bos_id
        self.max_txt_len = max_txt_len
        self.max_new_tokens = max_new_tokens
        # GNN Related
        self.gnn_num_layers = gnn_num_layers
        self.gnn_in_dim = gnn_in_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_heads = gnn_num_heads
        self.gnn_dropout = gnn_dropout
        
        super().__init__(**kwargs)