import yaml

from omegaconf import DictConfig
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import torch
from torch.utils.data import DataLoader
import plotly.express as px

from cpc.model import CPCAudioRawModel
from cpc.dataset import AudioRawDataset


cpc_model = CPCAudioRawModel.load_from_checkpoint(checkpoint_path='outputs/2021-02-25/18-18-53/lightning_logs/version_0/checkpoints/epoch=1432-step=179061.ckpt', cfg=cfg.model)
cpc_model.eval()

dataset = AudioRawDataset(
    manifest_file='',
    sample_len=20480,
)
dataloader = DataLoader(dataset, shuffle=False, batch_size=16, collate_fn=dataset.collate_fn)


embeddings = []
with torch.no_grad():
    for audio_signal in dataloader:
        _, c = cpc_model(audio_signal)
        embeddings.append(c[..., -1])

labels = [sample.audio_filepath.split('/')[-3] for sample in dataset.audio_files]
embeddings = np.concatenate(embeddings, axis=0) 
tsne = TSNE(
    n_components=2,
    perplexity=10.0,
    verbose=1
)

projections = tsne.fit_transform(embeddings)

fig = px.scatter(x=projections[:, 0], y=projections[:, 1], color=labels)
fig.show()
