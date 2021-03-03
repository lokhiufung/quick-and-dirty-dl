import os
import argparse

import tqdm
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import torch
from torch.utils.data import DataLoader

from cpc.model import CPCAudioRawModel
from cpc.dataset import AudioRawDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='ckpt of cpc model')
    parser.add_argument('--manifest_file', type=str, required=True, help='manifest file')
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()

    cpc_model = CPCAudioRawModel.load_from_checkpoint(checkpoint_path=args.ckpt)
    cpc_model.eval()

    dataset = AudioRawDataset(
        manifest_file=args.manifest_file,
        sample_len=22400,
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=16, collate_fn=dataset.collate_fn)

    if args.device == 'gpu':
        cpc_model.to('cuda:0')
    embeddings = []
    with torch.no_grad():
        for audio_signal in tqdm.tqdm(dataloader):
            if args.device == 'gpu':
                audio_signal = audio_signal.cuda()
            z, _ = cpc_model(audio_signal)
            embeddings.append(z.mean(dim=1).cpu().numpy())  # average over timesteps
            # _, c = cpc_model(audio_signal)
            # embeddings.append(c[..., -1].cpu().numpy())

    labels = [os.path.basename(sample.audio_filepath).split('-')[0] for sample in dataset.audio_files]
    embeddings = np.concatenate(embeddings, axis=0) 
    tsne = TSNE(
        n_components=2,
        perplexity=20.0,
        learning_rate=200.0,
        n_iter=5000,
        n_jobs=-1,
        verbose=1
    )

    projections = tsne.fit_transform(embeddings)

    fig = px.scatter(x=projections[:, 0], y=projections[:, 1], color=labels)
    fig.show()


if __name__ == '__main__':
    main()
    