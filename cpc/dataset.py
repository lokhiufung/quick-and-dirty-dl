import random
import json
import math
from functools import namedtuple

import librosa
import torch
import numpy as np
from torch.utils.data import Dataset


class AudioRawDataset(Dataset):
    def __init__(self, manifest_file: str, sample_len: int, sample_rate: int=16000, trim: bool=True, buffer_len: int=1):
        """
        Train on sampled audio windows of length 20480
        LibriSpeech train-100
        manifest_file: str, manifest file
        sample_len: int, number of steps to be sampled
        sample_rate: int, sample rate of the audio
        trim: bool, to trim leadning/trailing silence
        buffer_len: int, 
        """
        self.sample_len = sample_len 
        self.sample_rate = sample_rate
        self.min_duration = (self.sample_len + buffer_len) / sample_rate  # for stability; make sure the audio contain enough steps
        self.trim = trim
        self.audio_files = self._prepare_audio_text(manifest_file)
        # self.min_signal_length = math.floor(self.min_duration * self.sample_rate)

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
                if item['duration'] > self.min_duration:
                    audio_files.append(
                        AudioSample(audio_filepath=item['audio_filepath'], duration=item['duration'])
                    )
        audio_files = sorted(audio_files, key=lambda x: x.duration)
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index: int):
        # try:
        audio_sample = self.audio_files[index]
        audio_signal, _ = librosa.load(
            audio_sample.audio_filepath,
            sr=self.sample_rate,
            mono=True,
        )
        # FIXME: trimmed audio may be shorter than self.sample_len. This may induce error in random index.
        # if self.trim:
        #     audio_signal, _ = librosa.effects.trim(audio_signal, top_db=60)

        sample_index = random.randrange(start=0, stop=len(audio_signal) - self.sample_len)
        audio_signal = audio_signal[sample_index:sample_index+self.sample_len]
        # audio_signal = audio_signal[:self.sample_len] 
        return audio_signal, len(audio_signal)
        # except:
        #     print(audio_sample)
        #     raise Exception('dataset error')

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