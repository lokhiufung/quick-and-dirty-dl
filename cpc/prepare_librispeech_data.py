import argparse
import glob
import os
import json

import soundfile as sf
import pandas as pd


DATASETS = ['train-clean-100']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', '-D', type=str, help='data root of librispeech datasets')
    # parser.add_argument('--wav', action='store_true', default=True)
    args = parser.parse_args()
    return args


def _extract_transcripts(filepath, ext='.trans.txt'):
    tran_filepaths = list(glob.glob(f'{filepath}/*/*/*{ext}'))
    # filename2transcript = {}
    filename_transcript = []
    for tran_filepath in tran_filepaths:
        with open(tran_filepath, 'r') as f:
            for line in f:
                filename, transcript = line.strip().split(' ', 1)  # split by the first space
                # filename2transcript[filename] = transcript
                filename_transcript.append((filename, transcript))
    transcript_df = pd.DataFrame(filename_transcript, columns=['filename', 'transcript'])
    return transcript_df


def _extract_audio(filepath, ext='.flac'):
    audio_filepaths = list(glob.glob(f'{filepath}/*/*/*{ext}'))
    # filename2audio = {
    #     os.path.basename(audio_filepath): audio_filepath for audio_filepath in audio_filepaths
    # }
    filename_audio = [
        (os.path.basename(audio_filepath).replace(ext, ''), audio_filepath) for audio_filepath in audio_filepaths
    ]
    audio_df = pd.DataFrame(filename_audio, columns=['filename', 'audio_filepath'])
    return audio_df


@DeprecationWarning
def _flac_to_wav(audio_filepath, output_filepath):
    output_filepath = audio_filepath.replace('.flac', '.wav')
    os.system(f'ffmpeg -i {audio_filepath} {output_filepath}')
    # return output_filepath


def _get_duration(audio_filepath):
    audio, sample_rate = sf.read(audio_filepath)
    return len(audio) / sample_rate


def _prepare_sample(audio_filepath, duration, text, sample_id, spk_id):
    sample = dict()
    sample['audio_filepath'] = audio_filepath
    sample['duration'] = duration
    sample['text'] = text
    sample['sample_id'] = sample_id
    sample['spk_id'] = spk_id
    return sample


def write_df_to_manifest(df, output_filepath):
    with open(output_filepath, 'w') as f:
        for row in df.itertuples():
            sample = _prepare_sample(
                audio_filepath=row.audio_filepath,
                duration=row.duration,
                text=row.transcript.lower(),
                sample_id=row.filename,
                spk_id=row.filename.split('-', 1)[0]
            )
            f.write(json.dumps(sample) + '\n')


def main():
    args = parse_args()
    data_root = args.data_root
    # to_wav = args.wav

    for dataset in DATASETS:
        # ds_filepath = os.path.join(data_root, dataset)
        # transcript_df = _extract_transcripts(ds_filepath)
        # audio_df = _extract_audio(ds_filepath)
        # df = transcript_df.merge(audio_df, how='inner', on='filename')
        # # soundFile can open .flac file
        # # if to_wav:
        # #     print('convert .flac to .wav ...')
        # #     df['audio_filepath'].apply(_flac_to_wav)
        # df['duration'] = df['audio_filepath'].apply(_get_duration)
        df = pd.read_csv(f'{data_root}/{dataset}.csv', sep='\t', header=None, names=['filename', 'transcript', 'audio_filepath', 'duration'])
        write_df_to_manifest(
            df[:-1000],
            output_filepath=os.path.join(data_root, f'{dataset}-train.json')
        )
        write_df_to_manifest(
            df[-1000:],
            output_filepath=os.path.join(data_root, f'{dataset}-validation.json')
        )
        # output_filepath = os.path.join(data_root, f'{dataset}.csv')
        # df.to_csv(
        #     output_filepath,
        #     sep='\t',
        #     header=False,
        #     index=False,
        # )
        # print(f'wrote {dataset} to {output_filepath}')


if __name__ == '__main__':
    main()