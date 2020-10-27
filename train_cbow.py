import pandas as pd
import json
from gensim.models import Word2Vec
import numpy as np


SIZE_W2V = 128


sound_tags_data = pd.read_csv('tags/tags_ppc_top_1000.csv', error_bad_lines=False)
num_sounds = len(sound_tags_data)
sound_tags = [[t for t in sound_tags_data.iloc[idx].tolist()[1:] if isinstance(t, str)] for idx in range(num_sounds)]
model = Word2Vec(sound_tags, size=SIZE_W2V, window=100, min_count=1, workers=4)

id2tag = json.load(open('json/id2token_top_1000.json', 'rb'))

embedding_matrix = np.zeros((len(model.wv.vocab)+1, SIZE_W2V))
for i, v in id2tag.items():
    embedding_vector = model.wv[v]
    embedding_matrix[int(i)+1] = embedding_vector


np.save(f'embedding_matrix_{SIZE_W2V}.npy', embedding_matrix)
