from transformers import PreTrainedModel, LlamaForCausalLM, LlamaConfig
import torch
from torch import nn
from torch_geometric.nn.pool import global_mean_pool

from .g_retriever_config import GRetrieverConfig
from .gnn import GAT

class GRetrieverModel(PreTrainedModel):
    config_class = GRetrieverConfig

    def __init__(
        self,
        config: GRetrieverConfig,
        **kwargs
    ):
        super().__init__(config)
        self.config = config
        
        if isinstance(self.config.llama_config, dict):
            self.config.llama_config = LlamaConfig.from_dict(self.config.llama_config)

        self.llm_model = LlamaForCausalLM(self.config.llama_config)
        self.word_embedding = self.llm_model.get_input_embeddings()

        self.graph_encoder = GAT(
            in_channels=self.config.gnn_in_dim,
            out_channels=self.config.gnn_hidden_dim,
            hidden_channels=self.config.gnn_hidden_dim,
            num_layers=self.config.gnn_num_layers,
            dropout=self.config.gnn_dropout,
            num_heads=self.config.gnn_num_heads,
        )

        self.projector = nn.Sequential(
            nn.Linear(self.config.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.word_embedding.embedding_dim),
        )
    
    def encode_graphs(self, graph):
        n_embeds, _ = self.graph_encoder(graph.x, graph.edge_index.long(), graph.edge_attr)

        # mean pooling
        g_embeds = global_mean_pool(n_embeds, graph.batch)

        return g_embeds

    def forward(
        self,
        inputs,
        attention_mask,
        graph,
        labels=None
    ):
        bos_embeds = self.word_embedding(self.config.bos_id)
        
        graph_embeds = self.encode_graphs(graph)
        graph_embeds = self.projector(graph_embeds)
        
        non_tokenized_ids = (inputs == -1).nonzero()
        non_tokenized_shape = non_tokenized_ids[:, 0], non_tokenized_ids[:, 1]
        
        inputs[non_tokenized_shape] = self.config.llama_config.pad_token_id
        
        inputs_embeds = self.word_embedding(inputs)
        non_tokenized_embeds = torch.cat([bos_embeds.repeat(len(inputs), 1, 1), graph_embeds.unsqueeze(1)], dim=1)
        
        inputs_embeds[non_tokenized_shape] = non_tokenized_embeds.view(len(non_tokenized_ids), -1)
        
        if labels is not None:
            return self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        return inputs_embeds
    
    def generate(
        self,
        inputs,
        attention_mask,
        graph,
        **kwargs
    ):
        #inputs_embeds = self(inputs, attention_mask, graph)
        
        #return self.llm_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=20)
        inputs_embeds = self(inputs, attention_mask, graph)
        
        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            max_new_tokens=20,
            **kwargs)