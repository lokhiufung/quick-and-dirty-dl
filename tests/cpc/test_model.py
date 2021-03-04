from omegaconf import DictConfig 

from cpc.model import CPCAudioRawModel


def test_cpc_audio_raw_model(manifest_file):
    sample_len = 20480 + int(12 * 160)
    
    model_cfg = {
        'window_size': 20480,
        'downsampling': 160,
        'train_data': {
            'dataset': {
                'manifest_file': manifest_file,
                'sample_len': sample_len,
                'sample_rate': 16000
            },
            'dataloader': {
                'batch_size': 8,
                'shuffle': True,
                'num_workers': 2
            }
        },
        'validation_data': {
            'dataset': {
                'manifest_file': manifest_file,
                'sample_len': sample_len,
                'sample_rate': 16000
            },
            'dataloader': {
                'batch_size': 8,
                'shuffle': False,
                'num_workers': 2
            }
        },
        
        'encoder': {
            'hidden_size': 512
        },
        'ar': {
            'embedding_size': 512,
            'hidden_size': 256
        },
        'cpc_criterion': {
            'ar_embedding_size': 256,
            'enc_embedding_size': 512,
            'n_predictions': 12,
            'n_negs': 128
        },
        'optim': {
            'lr': 2e-4,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 0.0
        }
    }

    cpc_model = CPCAudioRawModel(cfg=DictConfig(model_cfg))
    batch = next(iter(cpc_model.train_dataloader()))
    loss = cpc_model.training_step(batch, 1)

    assert len(loss.size()) == 1
 
