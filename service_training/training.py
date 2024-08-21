import os
import sys
import pathlib
import mlflow
import torch

sys.path.append(str(pathlib.Path(os.path.dirname(os.path.realpath(__file__)), "../src")))
from job_recommender.config.configuration import ConfigurationManager
from job_recommender.pipeline.training import TrainingPipeline
from job_recommender import logger
from job_recommender.dataset.resume_dataset import ResumeDataset


if __name__ == "__main__":
    config = ConfigurationManager()

    train_config = config.get_hyperparameters()

    dataset = ResumeDataset(train_config.input_dir)

    mlflow.set_experiment("Compfest: Job Recommender")
    trainer = TrainingPipeline(train_config, dataset)

    with mlflow.start_run() as run:
        trainer.train()
        trainer.evaluation()