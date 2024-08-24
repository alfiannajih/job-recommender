from transformers import LlamaConfig

class GRetrieverConfig(LlamaConfig):
    model_type = "llama"
    
    def __init__(
        self,
        max_txt_len: int = 1024,
        max_new_tokens: int = 256,
        gnn_num_layers: int = 4,
        gnn_in_dim: int = 768,
        gnn_hidden_dim: int = 1024,
        gnn_num_heads: int = 4,
        gnn_dropout: int = 0,
        bos_id: list = [128000, 128006, 882, 128007],
        **kwargs
    ):
        pretrained_config = LlamaConfig.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")
        pretrained_config.update(kwargs)
        
        self.max_txt_len = max_txt_len
        self.max_new_tokens = max_new_tokens
        self.gnn_num_layers = gnn_num_layers
        self.gnn_in_dim = gnn_in_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_heads = gnn_num_heads
        self.gnn_dropout = gnn_dropout
        self.bos_id = bos_id
        
        super().__init__(**pretrained_config.to_dict())
        self.pad_token_id = pretrained_config.eos_token_id