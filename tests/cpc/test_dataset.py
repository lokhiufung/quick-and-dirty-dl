from torch.utils.data import DataLoader

from cpc.dataset import AudioRawDataset


def test_audio_raw_dataset(manifest_file):
    sample_len = 20480 + int(12 * 160)
    dataset = AudioRawDataset(
        manifest_file=manifest_file,
        sample_len=sample_len,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)

    for batch in dataloader:
        batch_audio_len = batch.size(2)
        assert batch_audio_len == sample_len
        