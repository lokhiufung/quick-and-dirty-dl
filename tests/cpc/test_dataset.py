import os

from torch.utils.data import DataLoader

from cpc.dataset import AudioRawDataset


USER = os.environ['USER']

sample_len = 20480 + int(12 * 160)
print(f'sample_len: {sample_len}')
dataset = AudioRawDataset(
    manifest_file=f'/home/{USER}/data/english/LibriSpeech/train-clean-100-validation.json',
    sample_len=sample_len,
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)
for batch in dataloader:
    # print(len(batch))
    assert batch.size(2) == sample_len, f'{batch.size(2)}'
    # break

