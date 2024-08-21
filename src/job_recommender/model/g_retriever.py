import torch
from torch import nn
import torch.nn.functional as F

from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import StaticCache
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position
from .g_retriever_config import GRetrieverConfig
from .gnn import GAT

from functools import wraps
from torch_geometric.nn.pool import global_mean_pool

class GRetrieverModel(LlamaForCausalLM):
    config_class = GRetrieverConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.graph_encoder = GAT(
            in_channels=config.gnn_in_dim,
            out_channels=config.gnn_hidden_dim,
            hidden_channels=config.gnn_hidden_dim,
            num_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout,
            num_heads=config.gnn_num_heads,
        )

        self.projector = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.get_input_embeddings().embedding_dim),
        )
    
    def encode_graphs(self, graph):
        n_embeds, _ = self.graph_encoder(graph.x, graph.edge_index.long(), graph.edge_attr)

        # mean pooling
        g_embeds = global_mean_pool(n_embeds, graph.batch)

        return g_embeds
    
    @wraps(LlamaForCausalLM.forward)
    def forward(
        self,
        input_ids=None,
        graph=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None
    ):  
        inputs = input_ids.clone()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs.shape != torch.Size([1, 1]):
            # embed bos prompt
            bos_embeds = self.get_input_embeddings()(torch.tensor(self.config.bos_id))
            
            # encode graph
            graph_embeds = self.encode_graphs(graph)
            graph_embeds = self.projector(graph_embeds)
            
            # prepare for reserved ids (bos+graph)
            non_tokenized_ids = (inputs == -1).nonzero()
            non_tokenized_shape = non_tokenized_ids[:, 0], non_tokenized_ids[:, 1]
            
            # embed inputs
            inputs[non_tokenized_shape] = self.config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(inputs)
            non_tokenized_embeds = torch.cat([bos_embeds.repeat(len(inputs), 1, 1), graph_embeds.unsqueeze(1)], dim=1)
            
            # replace reserved ids with bos+graph
            inputs_embeds[non_tokenized_shape] = non_tokenized_embeds.view(len(non_tokenized_ids), -1)
        
        else:
            inputs_embeds = self.get_input_embeddings()(inputs)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        graph=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "graph": graph,
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs