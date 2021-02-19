from torch.utils.data import DataLoader

from cpc.dataset import AudioRawDataset


dataset = AudioRawDataset(
    manifest_file='/home/data/english/LibriSpeech/train-clean-100-validation.json',
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)
for batch in dataloader:
    print(len(batch))
    print(batch.size())
    break
