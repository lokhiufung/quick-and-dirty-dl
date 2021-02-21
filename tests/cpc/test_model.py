import os

import tqdm
import torch
from torch.utils.data import DataLoader

from cpc.model import ConvNetEncoder, GRUAutoRegressiveModel, LinearPredictionModel
from cpc.dataset import AudioRawDataset


USER = os.environ['USER']
encoder = ConvNetEncoder()
ar = GRUAutoRegressiveModel()
predictors = [LinearPredictionModel() for _ in range(12)]

dataset = AudioRawDataset(
    manifest_file=f'/home/{USER}/data/english/LibriSpeech/train-clean-100-validation.json',
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)

window_size = 128
batch =  next(iter(dataloader))

embed = encoder(batch)
context = ar(embed)

batch_size, steps, ar_dim = context.size()  

context = context[:, :window_size]

z_negs = torch.normal(0, 1, size=(batch_size, 10, window_size, 512))

for k in range(1, 13):
    z_pred = predictors[k](context)
    # print(z_pred.size())
    z_pred = z_pred.unsqueeze(1)
    print('z_pred: ', z_pred.size())
    z_pos = embed[:, k:k+window_size].unsqueeze(1)
    print('z_pos: ', z_pos.size())
    z_samples = torch.cat([z_pos, z_negs], dim=1) # k step forward for k ar output, prepare e.g 10 neg samples + 1 pos sample
    print('z_samples: ', z_samples.size())
    logit = (z_pred * z_samples).sum(dim=3)
    logit = logit.permute(0, 2, 1)
    logit = logit.contiguous().view(-1, logit.size(2))
    print('logit: ', logit.size())
    break
# print(context.size())


    


   
#    print(embed.shape)

# print(embed)