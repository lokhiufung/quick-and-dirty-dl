import os

import tqdm
from torch.utils.data import DataLoader

from cpc.model import ConvNetEncoder
from cpc.dataset import AudioRawDataset


USER = os.environ['USER']
encoder = ConvNetEncoder()
print(encoder)

dataset = AudioRawDataset(
    manifest_file=f'/home/{USER}/data/english/LibriSpeech/train-clean-100-validation.json',
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)

for batch in tqdm.tqdm(dataloader):
   embed = encoder(batch)
#    print(embed.shape)

# print(embed)