from transformers import Pipeline, AutoTokenizer
from torch_geometric.data import Batch
import torch

class GRetrieverPipeline(Pipeline):
    def __init__(self, **kwargs):
        Pipeline.__init__(self, **kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        self.eos_user = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        self.max_txt_len = self.model.config.max_txt_len
        self.bos_length = len(self.model.config.bos_id)
        
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "textualized_graph" in kwargs:
            preprocess_kwargs["textualized_graph"] = kwargs["textualized_graph"]

        if "graph" in kwargs:
            preprocess_kwargs["graph"] = kwargs["graph"]
            
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, textualized_graph, graph):
        textualized_graph_ids = self.tokenizer(textualized_graph, add_special_tokens=False)["input_ids"][:self.max_txt_len]
        question_ids = self.tokenizer(inputs, add_special_tokens=False)["input_ids"]
        eos_user_ids = self.tokenizer(self.eos_user, add_special_tokens=False)["input_ids"]
        
        input_ids = torch.tensor([
            [-1]*(self.bos_length + 1)
            + textualized_graph_ids
            + question_ids
            + eos_user_ids
        ])
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids)
        }
        model_inputs.update({
            "graph": Batch.from_data_list([graph])
        })
        
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model.generate(**model_inputs)
            
        return model_outputs

    def postprocess(self, model_outputs):
        return model_outputs