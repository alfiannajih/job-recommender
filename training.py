import sys
import os
import argparse
import pathlib
from torch.utils.data import random_split, DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
import mlflow
from dotenv import load_dotenv

sys.path.append(str(pathlib.Path(os.path.dirname(os.path.realpath(__file__)), "src")))

from job_recommender.dataset.resume_dataset import ResumeDataset, collate_fn
from job_recommender.model.original_model import GraphLLM

load_dotenv()

def main(args):
    model = GraphLLM(
        max_txt_len=args.max_txt_len,
        max_new_tokens=args.max_new_tokens,
        llm_model_path=args.llm_path,
        gnn_in_dim=args.gnn_in_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        gnn_num_heads=args.gnn_num_heads,
        gnn_dropout=args.gnn_dropout,
    )

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.weight_decay}, ],
        betas=(0.9, 0.95)
    )

    num_epochs = 2
    log_every_n_steps = 20
    checkpoint_every_n_steps = 500

    best_val_loss = float('inf')

    parameter_log = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_txt_len": args.max_txt_len,
        "max_new_tokens": args.max_new_tokens,
        "llm_model_path": args.llm_path,
        "gnn_in_dim": args.gnn_in_dim,
        "gnn_hidden_dim": args.gnn_hidden_dim,
        "gnn_num_layers": args.gnn_num_layers,
        "gnn_num_heads": args.gnn_num_heads,
        "gnn_dropout": args.gnn_dropout,
    }

    mlflow.log_params(parameter_log)
    dataset = ResumeDataset(pathlib.Path("dataset/resume_dataset"))
    train_set, val_set, test_set = random_split(dataset, [0.85, 0.1, 0.05])

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        collate_fn=collate_fn
    )
    
    global_train_steps = num_epochs * len(train_loader)
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.
        loss_step = 0.
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            optimizer.step()
            epoch_loss += loss.item()
            loss_step += loss.item()
            
            global_step += 1
            if (step + 1) % log_every_n_steps == 0:
                mlflow.log_metrics({"train_loss - step": loss_step / log_every_n_steps}, step=global_step)
                print(f"[Training] Epoch: {epoch}|{num_epochs}\tStep: {global_step}|{global_train_steps}\tTrain Loss (Step): {loss_step / log_every_n_steps}")
                loss_step = 0.
            
            if global_step % checkpoint_every_n_steps == 0:
                torch.save(model.state_dict(), "output/model_steps_{}.pth".format(global_step))

        mlflow.log_metrics({"train_loss - epoch": epoch_loss / len(train_loader)}, step=epoch)
        print(f"[Training] Epoch: {epoch}|{num_epochs}\tStep: {global_step}|{global_train_steps}\tTrain Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        
        model.eval()    
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss/len(val_loader)
            
            mlflow.log_metrics({"val_loss - epoch": val_loss}, step=epoch)
            print(f"[Validation] Epoch: {epoch}|{num_epochs}\tVal Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "output/model_best.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_path", type=str, default="NousResearch/Hermes-3-Llama-3.1-8B")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_txt_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gnn_in_dim", type=int, default=768)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("Compfest: Job Recommender")
    
    with mlflow.start_run():
        args = parser.parse_args()
        main(args)