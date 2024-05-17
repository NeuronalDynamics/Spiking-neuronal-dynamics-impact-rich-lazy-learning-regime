import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torchvision import datasets, transforms
import neurogym as ngym
from torch.utils.data import DataLoader, IterableDataset
import numpy as np


class NeurogymDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.dataset = create_neurogym_dataset()

    def train_dataloader(self):
        return tud.DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return tud.DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return tud.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers)

def create_neurogym_dataset():
    task = 'PerceptualDecisionMaking-v0'
    timing = {'fixation': 100,
              'stimulus': 1000, #1000
              'delay': 0,
              'decision': 100}
    kwargs = {'dt': 5, 'timing': timing} #'dt': 20
    seq_len = 1200//5                     # timing/dt     #1200/5

    # Make supervised dataset
    dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                           seq_len=seq_len)

    inputs, targets = dataset()
    # Convert into dataset
    nsteps, n, input_size = inputs.shape

    inputs = np.moveaxis(inputs, 0,1)
    targets = np.moveaxis(targets, 0,1)

    inputs = inputs.reshape((n, nsteps, input_size))
    targets = targets.reshape((n, nsteps, -1))

    return tud.TensorDataset(torch.tensor(inputs).float(), torch.tensor(targets).float())