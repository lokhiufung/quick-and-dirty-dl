import json
from functools import namedtuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig

from cpc.dataset import AudioRawDataset


class ConvNetEncoder(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=10, stride=5, padding=3)
        self.bnorm1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=8, stride=4, padding=2)
        self.bnorm2 = nn.BatchNorm1d(hidden_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm1d(hidden_size)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm1d(hidden_size)
        self.conv5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1)
        self.bnorm5 = nn.BatchNorm1d(hidden_size)
    
    @property
    def input_port(self):
        return (
            ('audio_signal', ('B', 'C', 'T')),
        )
    @property
    def output_port(self):
        return (
            ('encoder_embedding', ('B', 'T', 'C')),
        )

    def forward(self, x):
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = F.relu(self.bnorm3(self.conv3(x)))
        x = F.relu(self.bnorm4(self.conv4(x)))
        x = F.relu(self.bnorm5(self.conv5(x)))
        x = x.transpose(1, 2)  # Reminder: make the channel last
        return x


class GRUAutoRegressiveModel(nn.Module):
    def __init__(self, embedding_size=512, hidden_size=256, keep_hidden=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.keep_hidden = keep_hidden
        self.hidden = None

        self.rnn = nn.GRU(self.embedding_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

    def input_port(self):
        return (
            ('encoder_embedding', ('B', 'T', 'C')),
        )

    def output_port(self):
        return (
            ('ar_embedding', ('B', 'T', 'C')),
        )

    def forward(self, x):
        x, h = self.rnn(x, self.hidden)  # (batch, seq_len, hidden_size)
        if self.keep_hidden:
            self.hidden = h.detach()
        return x


class LinearPredictionModel(nn.Module):
    def __init__(self, ar_embedding_size=256, enc_embedding_size=512):
        super().__init__()
        self.ar_embedding_size = ar_embedding_size
        self.linear = nn.Linear(ar_embedding_size, enc_embedding_size, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


class CPCCriterion(nn.Module):
    def __init__(self, ar_embedding_size=256, enc_embedding_size=512, n_predictions=12, n_negs=8):
        super().__init__()
        self.ar_embedding_size = ar_embedding_size
        self.enc_embedding_size = enc_embedding_size
        self.n_predictions = n_predictions  # max number of steps for prediction horizon
        self.n_negs = n_negs  # number of negative samples to be sampled
        self.loss_function = nn.CrossEntropyLoss()  # reminder: mean reduction by defualt
        self.predictors = []
        for _ in range(self.n_predictions):
            self.predictors.append(
                LinearPredictionModel(self.ar_embedding_size, self.enc_embedding_size)
            )

    @property
    def input_port(self):
        return (
            ('encoder_embedding', ('B', 'T', 'C')),
            ('ar_embedding', ('B', 'T', 'C')),
        )

    @property
    def output_port(self):
        return (
            ('loss', ('N',)),
            ('acc', ('N',))
        )

    def get_random_samples(self, z_features, window_size):
        samples = []
        batch_size, steps, z_dim = z_features.size()

        # randomly sample n_negs * batch_size for each step
        z_neg = z_features.contiguous().view(-1, z_dim)
        sample_idx = torch.randint(low=0, high=batch_size*steps, size=(batch_size*self.n_negs*window_size,))
        z_neg = z_neg[sample_idx].view(batch_size, self.n_negs, window_size, z_dim)
        
        labels = torch.zeros(size=(batch_size*window_size,), dtype=torch.long)
        for k in range(1, self.n_predictions + 1):
            z_pos = z_features[:, k:k+window_size].unsqueeze(1) 
            sample = torch.cat([z_pos, z_neg], dim=1)
            samples.append(sample)
        return samples, labels

    def forward(self, c_features, z_features, window_size):
        c_features = c_features[:, :window_size]
        samples, labels = self.get_random_samples(z_features, window_size)
        losses = []
        for k in range(self.n_predictions):
            z_pred = self.predictors[k](c_features)
            z_pred = z_pred.unsqueeze(1)
            prediction = (z_pred * samples[k]).sum(dim=3)

            prediction = prediction.permute(0, 2, 1)
            prediction = prediction.contiguous().view(-1, prediction.size(2))
            loss = self.loss_function(prediction, labels)
            losses.append(loss.view(1, -1))
        return torch.cat(losses, dim=1)
            

class CPCAudioRawModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.window_size = cfg.window_size // cfg.downsampling  # number of steps in encoded space  
        # self.enc_embedding_size = cfg.enc_embedding_size
        # self.ar_embedding_size = cfg.ar_embedding_size
        # self.n_predictions = cfg.n_predictions
        
        self.encoder = ConvNetEncoder(**cfg.encoder)
        self.ar = GRUAutoRegressiveModel(**cfg.ar)
        self.cpc_criterion = CPCCriterion(**cfg.cpc_criterion)

        self._train_dataset = None
        self._validation_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._optimizers = None
        
        self.setup_train_dataloader(cfg.train_data)
        self.setup_val_dataloader(cfg.validation_data)
        self.setup_optimizers(cfg.optim)

    def setup_optimizers(self, optim_cfg: DictConfig):
        self._optimizers = Adam(self.parameters(), **optim_cfg)

    def setup_train_dataloader(self, train_data_cfg: DictConfig):
        self._train_dataset = AudioRawDataset(
            **train_data_cfg.dataset
        )
        self._train_dataloader = DataLoader(
            self._train_dataset,
            collate_fn=self._train_dataset.collate_fn,
            **train_data_cfg.dataloader
        )
    
    def setup_val_dataloader(self, validation_data_cfg: DictConfig):
        self._validation_dataset = AudioRawDataset(
            **validation_data_cfg.dataset
        )
        self._val_dataloader = DataLoader(
            self._validation_dataset,
            collate_fn=self._validation_dataset.collate_fn,
            **validation_data_cfg.dataloader
        )

    def forward(self, audio_signal):
        z_features = self.encoder(audio_signal)
        c_features = self.ar(z_features)
        return z_features, c_features

    def training_step(self, batch, batch_idx):
        """
        batch: audio_signals; tensor (B, C, L)
        """
        z_features, c_features = self(batch)
        # c_features = c_features[:, :self.window_size]
        # random_samples = self.get_random_samples(z_features)
        losses = self.cpc_criterion(c_features, z_features, self.window_size)
        total_loss = losses.sum(dim=1)
    
        self.log('train_loss', total_loss, on_step=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        batch: audio_signals; tensor (B, C, L)
        """
        z_features, c_features = self(batch)
        # c_features = c_features[:, :self.window_size]
        # random_samples = self.get_random_samples(z_features)
        losses = self.cpc_criterion(c_features, z_features, self.window_size)
        total_loss = losses.sum(dim=1)
    
        self.log('val_loss', total_loss, on_step=True, prog_bar=True, logger=True)
        return total_loss

    def train_dataloader(self):
        if self._train_dataloader:
            return self._train_dataloader
        else:
            raise AttributeError('Please setup_train_dataloader() first')

    def val_dataloader(self):
        if self._val_dataloader:
            return self._val_dataloader
        else:
            raise AttributeError('Please setup_val_dataloader() first')
    
    def configure_optimizers(self):
        if self._optimizers:
            return self._optimizers
        else:
            raise AttributeError('Please setup_optimizers() first')
