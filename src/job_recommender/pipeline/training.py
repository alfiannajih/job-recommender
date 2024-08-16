import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split, DataLoader

from job_recommender.dataset.resume_dataset import ResumeDataset
from job_recommender.config.configuration import HyperparametersConfig
from job_recommender.utils.common import (
    seed_everything,
    adjust_learning_rate,
    _save_checkpoint
)
from job_recommender.utils.dataset import collate_fn
from job_recommender.model.graph_llm import GraphLLM
from job_recommender import logger

class TrainingPipeline:
    def __init__(
        self,
        config: HyperparametersConfig,
        dataset: ResumeDataset
    ):
        self.config = config
        seed_everything(self.config.seed)

        train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])
        
        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_set, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_set, batch_size=self.config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

        self.model = GraphLLM(self.config)

        params = [p for _, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [{'params': params, 'lr': float(self.config.learning_rate), 'weight_decay': float(self.config.weight_decay)}, ],
            betas=(0.9, 0.95)
        )

        trainable_params, all_param = self.model.print_trainable_params()
        logger.info(f"Trainable params: {trainable_params}\tAll params: {all_param}\tTrainable ratio: {100 * trainable_params / all_param}")

    def train(self):
        global_train_steps = self.config.num_epochs * len(self.train_loader)
        best_val_loss = float('inf')

        global_step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss, accum_loss = 0., 0.

            for step, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                loss = self.model(batch)
                loss.backward()

                clip_grad_norm_(self.optimizer.param_groups[0]['params'], 0.1)

                if (step + 1) % self.config.grad_steps == 0:
                    adjust_learning_rate(self.optimizer.param_groups[0], float(self.config.learning_rate), step / len(self.train_loader) + epoch, self.config)

                self.optimizer.step()
                epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

                global_step += 1
                if (step + 1) % self.config.grad_steps == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info("Step: {}|{}\tLr: {}\tAccum Loss: {}".format(global_step, global_train_steps, lr, accum_loss / self.config.grad_steps))
                    accum_loss = 0.

            logger.info(f"Epoch: {epoch}|{self.config.num_epochs}\tStep: {global_step}|{global_train_steps}\tTrain Loss (Epoch Mean): {epoch_loss / len(self.train_loader)}")
            
            best_val_loss = self._validation(epoch, best_val_loss)
    
    def _validation(self, epoch, best_val_loss):
        val_loss = 0
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.model(batch)
                val_loss += loss.item()
            val_loss = val_loss/len(self.val_loader)
            logger.info(f"Epoch: {epoch}|{self.config.num_epochs}: Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(self.model, self.optimizer, epoch, self.config, is_best=True)
            best_epoch = epoch

        logger.info(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        return best_val_loss