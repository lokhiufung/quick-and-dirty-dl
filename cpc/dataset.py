import json
from functools import namedtuple

import librosa
import torch
import numpy as np
from torch.utils.data import Dataset


class AudioRawDataset(Dataset):
    def __init__(self, manifest_file: str, sample_len: int=20480, sample_rate: int=16000, min_duration: float=0.1, max_duration: float=16.7, trim: bool=True):
        """
        Train on sampled audio windows of length 20480
        LibriSpeech train-100
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sample_len = sample_len
        self.sample_rate = sample_rate
        self.trim = trim
        self.audio_files = self._prepare_audio_text(manifest_file)

    def _prepare_audio_text(self, manifest_file: str) -> list:
        AudioSample = namedtuple(
            'AudioSample',
            ('audio_filepath', 'duration')
        )
        audio_files = []
        with open(manifest_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # filter audio with window_size + prediction_step * downsampling factor
                if (item['duration'] >= self.min_duration) and (item['duration'] <= self.max_duration):
                    audio_files.append(
                        AudioSample(audio_filepath=item['audio_filepath'], duration=item['duration'])
                    )
        audio_files = sorted(audio_files, key=lambda x: x.duration)
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index: int):
        audio_sample = self.audio_files[index]
        audio_signal, _ = librosa.load(
            audio_sample.audio_filepath,
            sr=self.sample_rate,
            mono=True,
        )
        if self.trim:
            audio_signal, _ = librosa.effects.trim(audio_signal, top_db=60)
        # audio_signal = audio_signal[:self.sample_len] 
        return audio_signal, len(audio_signal)

    def collate_fn(self, batch):
        audio_lengths = [sample[1] for sample in batch]
        min_length = min(audio_lengths)
        audio_signals = [
            torch.tensor(sample[0][:min_length]) for sample in batch
        ]  # cut off the parts beyond min_length 
        audio_signals = torch.stack(audio_signals, dim=0)
        audio_signals = audio_signals.unsqueeze(1)  # to ensure the format for convolution encoder (B, C, L) 
        return audio_signals

    @property
    def output_port(self):
        #for safety checking
        return (
            ('audio_signal', ('B', 'C', 'L')),
        )