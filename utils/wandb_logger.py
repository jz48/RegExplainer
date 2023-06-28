import wandb
import random
import time
from datetime import datetime


class WandbLogger:
    def __init__(self, project_name, config, if_log=True):
        self.run = None
        self.project_name = project_name
        self.config = config
        self.if_log = if_log
        now = datetime.now()  # current date and time
        self.date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        self.config = config

    def start_log(self, seed, idx):
        if self.if_log:
            self.run = wandb.init(
                # set the wandb project where this run will be logged
                project=self.project_name+self.date_time,
                name='seed_'+str(seed)+'_idx_'+str(idx),
                # track hyper-parameters and run metadata
                config=self.config
            )

    def log_step(self, data, step):
        if self.if_log:
            self.run.log(data, step=step)

    def close(self):
        if self.if_log:
            self.run.finish()


if __name__ == '__main__':
    logger = WandbLogger('1', {1: 1})
