import json
from functools import namedtuple
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


        

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
            ('ar_embedding', ('B', 'T', 'C'))
        )

    def forward(self, x):
        x, h = self.rnn(x, self.hidden)  # (batch, seq_len, hidden_size)
        if self.keep_hidden:
            self.hidden = h.detach()
        return x



class CPCCriterion(nn.Module):
    def __init__(self, ar_embedding_size, enc_embedding_size, n_predictions):
        self.ar_embedding_size = ar_embedding_size
        self.enc_embedding_size = enc_embedding_size
        self.loss_function = nn.CrossEntropyLoss()
        self.prediction_models = []
        for _ in self.n_prediction:
            self.prediction_models.append(
                LinearPredictionModel(self.ar_embedding_size, self.enc_embedding_size)
            )

    def forward(self, c_t, samples, labels):
        samples, labels = self.get_samples(z_features)
        predictions = []
        for k in range(1, self.n_prediction + 1):
            pred = samples[:, -k, :] * self.prediction_models[-k](c_t)  # scalar product
            predictions.append(pred)

        loss = torch.mean(losses)
        return loss

class LinearPredictionModel(nn.Module):
    def __init__(self, ar_embedding_size=256, enc_embedding_size=512):
        super().__init__()
        self.ar_embedding_size = ar_embedding_size
        self.linear = nn.Linear(ar_embedding_size, enc_embedding_size, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x


class HyperParameters(object):
    def __init__(self, enc_embedding_size: int=512, ar_embedding_size: int=256,
        lr: float=2e-4, batch_size: int=16, n_prediction: int=12
        ):
        self.enc_embedding_size = enc_embedding_size
        self.ar_embedding_size = ar_embedding_size
        self.lr = lr
        self.batch_size = batch_size
        self.n_prediction = n_prediction


################################
# ENV VARIABLES
TRAIN_MANIFEST = ''
VALIDATION_MANIFEST = ''
SAMPLE_RATE = 16000
DOWNSAMPLING = 160
MAX_DURATION = 16.7
SAMPLE_LEN = 20480
N_PREDICTION = 12
MIN_DURATION = (SAMPLE_LEN + N_PREDICTION * DOWNSAMPLING + 1) / SAMPLE_RATE  # prepare n_prediction + 1 steps
################################ 


class CPCAudioRawModel(pl.LightningModule):
    def __init__(self, batch_size: int=16, lr: float=2e-4, enc_embedding_size: int=512, ar_embedding_size: int=256, n_prediction: int=12):
        self.batch_size = batch_size
        self.lr = lr
        self.enc_embedding_size = enc_embedding_size
        self.ar_embedding_size = ar_embedding_size
        self.n_prediction = n_prediction
        
        self.encoder = ConvNetEncoder(self.enc_embedding_size)
        self.ar = GRUAutoRegressiveModel(self.enc_embedding_size, ar_embedding_size)
        self.cpc_criterion = CPCCriterion(self.ar_embedding_size, self.enc_embedding_size, self.n_predictions) 
        self.train_dataset = None
        self.validation_dataset = None
        
        self.window_size = SAMPLE_LEN / DOWNSAMPLING  # context size of ar 

    def setup(self):
        self.train_dataset = AudioRawDataset(TRAIN_MANIFEST, sample_len=SAMPLE_LEN, min_duration=MIN_DURATION, max_duration=MAX_DURATION, trim=True)
        self.validation_dataset = AudioRawDataset(VALIDATION_MANIFEST, sample_len=SAMPLE_LEN, min_duration=MIN_DURATION, max_duration=MAX_DURATION, trim=True)

    def get_samples(self, z_features):
        pass

    def forward(self, audio_signal, hidden):
        enc_embedding = self.encoder(audio_signal)
        ar_embedding = self.ar(enc_embedding)
        return enc_embedding, ar_embedding

    def training_step(self, batch, batch_idx):
        audio_signal, audio_len = batch
        
        z_features, c_features = self(audio_signal)
        z_features = z_features[:, self.window_size:, :]  
        c_t = c_features[:, self.window_size + 1, :]

        samples = self.get_samples(z_features)
        
        loss = self.cpc_criterion(c_t, samples)
        return loss


    def _collate_fn(self, batch):
        audio_signal, audio_len = batch
        audio_signal = torch.from_numpy(audio_signal.astype(np.float32))
        audio_len = torch.from_numpy(audio_len.astype(np.int8))
        return audio_signal, audio_len

    def train_dataloader(self):
        self.train_dataset = AudioRawDataset(
            manifest_file=TRAIN_MANIFEST,
            sample_rate=SAMPLE_RATE,
            min_duration=MIN_DURATION,
            max_duration=MAX_DURATION,
            trim=True
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self._collate_fn)

    def validation_dataloader(self):
        self.validation_dataset = AudioRawDataset(
            manifest_file=VALIDATION_MANIFEST,
            sample_rate=SAMPLE_RATE,
            min_duration=MIN_DURATION,
            max_duration=MAX_DURATION,
            trim=True
        )
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self._collate_fn)
