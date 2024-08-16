import yaml
import json
from box import ConfigBox
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)
import torch
import os
import random
import numpy as np
import math

def read_json(path):
    with open(path, "r") as fp:
        return json.load(fp)

def read_yaml(path):
    with open(path) as fp:
        return ConfigBox(yaml.safe_load(fp))
    
def get_emb_model(path):
    model = SentenceTransformer(model_name_or_path=path, trust_remote_code=True)

    return model

def get_rerank_model(path):
    model = CrossEncoder(model_name=path, trust_remote_code=True)

    return model

def _save_checkpoint(model, optimizer, cur_epoch, config, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    os.makedirs(f'{config.output_dir}', exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": cur_epoch,
    }
    #path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_{"best" if is_best else cur_epoch}.pth'
    path = "{}/{}_model.pth".format(config.output_dir, "best" if is_best else cur_epoch)
    torch.save(save_obj, path)


def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    #checkpoint_path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}_checkpoint_best.pth'

    #print("Loading checkpoint from {}.".format(checkpoint_path))
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def adjust_learning_rate(param_group, LR, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 5e-6
    if epoch < args.warmup_epochs:
        lr = LR * epoch / args.warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group["lr"] = lr
    return lr