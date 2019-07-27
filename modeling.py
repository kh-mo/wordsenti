import os
import pickle
import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":
    vector_dimension = 100
    with open(os.path.join(os.getcwd(), 'raw_data/tokenized'), "rb") as f:
        input_data = pickle.load(f)
    print("Load tokenized data done.")

    # word2vec modeling & save
    model = Word2Vec(input_data, size=vector_dimension, window=5, min_count=1, workers=4)
    print("Word2vec modeling done.")
    model.save(os.path.join(os.getcwd(), "word2vec.model"))

    # word vector L2 normalization
    model.init_sims(True)

    # get word name list
    words = list(model.wv.vocab.keys())

    # get euclidean distance matrix
    word_dim_mat = np.zeros((len(words), vector_dimension))
    for idx, word in enumerate(words):
        word_dim_mat[idx] = model.__getitem__(word)
    euc_mat = euclidean_distances(word_dim_mat)
    print("Generate euclidean distance matrix done.")

    euc_mat[:5,:5]
    np.max(euc_mat)