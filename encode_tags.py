"""
This script is used to compute neural network embeddings.
"""
import torch
import numpy as np
import sklearn
import pickle
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import librosa
import pandas as pd

from utils import compute_spectrogram, return_spectrogram_max_nrg_frame
from models_t1000_att import AudioEncoder, TagMeanEncoder, TagSelfAttentionEncoder


scaler = pickle.load(open('./scaler_top_1000.pkl', 'rb'))
id2tag = json.load(open('./json/id2token_top_1000.json', 'rb'))
tag2id = {tag: id for id, tag in id2tag.items()}


def return_loaded_model(Model, checkpoint):
    model = Model()
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    model.eval()
    return model


def extract_audio_embedding(model, filename):
    with torch.no_grad():
        try:
            x = compute_spectrogram(filename)[:96, :96]
            x = scaler.transform(x)
            x = torch.unsqueeze(torch.unsqueeze(torch.tensor(x), 0), 0).float()
            embedding, embedding_d = model(x)
            return embedding, embedding_d
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e, filename)


def extract_audio_embedding_chunks(model, filename):
    with torch.no_grad():
        try:
            x = compute_spectrogram(filename)
            x_chunks = np.array([scaler.transform(chunk.T) for chunk in 
                    librosa.util.frame(np.asfortranarray(x), frame_length=96, hop_length=96, axis=-1).T])
            x_chunks = torch.unsqueeze(torch.tensor(x_chunks), 1)
            embedding_chunks, embedding_d_chunks = model(x_chunks)
            return embedding_chunks, embedding_d_chunks
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e, filename)


def extract_tag_embedding(model, tag):
    with torch.no_grad():
        try:
            tag_vector = torch.tensor(np.zeros(1000)).view(1, 1000).float()
            tag_vector[0, int(tag2id[tag])] = 1
            embedding, _ = model(tag_vector)
            return embedding
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e, tag)


if __name__ == "__main__":

    tag_vectors = json.load(open('./json/US8K_tag_vectors.json', 'r'))
    us8k_metadata = pd.read_csv('./data/UrbanSound8K/metadata/UrbanSound8K.csv')

    filenames_tag_vectors = [
        ('fold{}/{}'.format(row['fold'], row['slice_file_name']), tag_vectors[str(row['fsID'])])
        for idx, row in us8k_metadata.iterrows() 
        if str(row['fsID']) in tag_vectors.keys()
        and sum(tag_vectors[str(row['fsID'])]) > 0
    ]

    BASE_PATH = './data/UrbanSound8K/audio/'

    for MODEL_NAME in [
        'ae_w2v_selfatt_c_1h/tag_encoder_att_epoch_200',
        'ae_w2v_128_selfatt_c_1h/tag_encoder_att_epoch_200',
        'ae_w2v_selfatt_c_4h/tag_encoder_att_epoch_200',
        'ae_w2v_128_selfatt_c_4h/tag_encoder_att_epoch_200',
        'ae_w2v_mean_c/tag_encoder_att_epoch_200',
        'ae_w2v_128_mean_c/tag_encoder_att_epoch_200',
    ]:
        MODEL_PATH = f'./saved_models/{MODEL_NAME}.pt'

        if 'ae_w2v_att_c' in MODEL_NAME:
            audio_model = return_loaded_model(
                AudioEncoder, './saved_models/ae_w2v_att_c_2/audio_encoder_epoch_200.pt'
            )
            audio_model.eval()
            model = return_loaded_model(
                lambda: TagAttentionEncoder(1001, 1152, 1, 1152, 1152, 1152),
                MODEL_PATH
            )
        elif 'ae_w2v_mean_c' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagMeanEncoder(1001, 1152, 1152),
                MODEL_PATH
            )
        elif 'ae_w2v_128_mean_c' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagMeanEncoder(1001, 128, 128),
                MODEL_PATH
            )
        elif 'ae_w2v_selfatt_c_1h' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagSelfAttentionEncoder(1001, 1152, 1, 1152, 1152, 1152),
                MODEL_PATH
            )
        elif 'ae_w2v_128_selfatt_c_1h' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagSelfAttentionEncoder(1001, 128, 1, 128, 128, 128),
                MODEL_PATH
            )
        elif 'ae_w2v_selfatt_c_4h' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagSelfAttentionEncoder(1001, 1152, 4, 1152, 1152, 1152),
                MODEL_PATH
            )
        elif 'ae_w2v_128_selfatt_c_4h' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagSelfAttentionEncoder(1001, 128, 4, 128, 128, 128),
                MODEL_PATH
            )

        model.eval()
        model_name = MODEL_NAME.split('/')[0] + '_' + MODEL_NAME.split('_epoch_')[-1]
        folder = f'./data/tag_embeddings/us8k/embeddings_{model_name}'
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f'\n {MODEL_NAME}')

        for filename, tag_vector in tqdm(filenames_tag_vectors):
            with torch.no_grad():
                filename_stem = filename.split('/')[-1].split('.')[0]
                path = os.path.join(BASE_PATH, filename)

                if 'ae_w2v_att_c' in MODEL_NAME:
                    spec = compute_spectrogram(path)
                    spec_cut = return_spectrogram_max_nrg_frame(spec)
                    x = scaler.transform(spec_cut)
                    x = torch.unsqueeze(torch.unsqueeze(torch.tensor(x), 0), 0).float()
                    z_audio, _ = audio_model(x)
                else:
                    z_audio = None

                tag_idxs = [
                    ([idx+1 for idx, val in enumerate(tag_vector) if val]
                    + 10*[0])[:10]
                ]
                tags_input = torch.tensor(tag_idxs, dtype=torch.long)
                z_tags, _, _ = model(tags_input, z_audio, mask=tags_input.unsqueeze(1))
                np.save(os.path.join(folder, filename_stem + '.npy'), z_tags)
