import os

import tqdm
import torch
from torch.utils.data import DataLoader

from cpc.model import ConvNetEncoder, GRUAutoRegressiveModel, LinearPredictionModel, CPCAudioRawModel
from cpc.dataset import AudioRawDataset


USER = os.environ['USER']
# encoder = ConvNetEncoder()
# ar = GRUAutoRegressiveModel()
# predictors = [LinearPredictionModel() for _ in range(12)]

dataset = AudioRawDataset(
    manifest_file=f'/home/{USER}/data/english/LibriSpeech/train-clean-100-validation.json',
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)

batch = next(iter(dataloader))
cpc_model = CPCAudioRawModel()
loss = cpc_model.training_step(batch, 1)
print(loss)
