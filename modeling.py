import os
import json
import time
import pickle
import numpy as np

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":
    start_time = time.time()
    vector_dimension = 100
    with open(os.path.join(os.getcwd(), 'raw_data/tokenized'), "rb") as f:
        input_data = pickle.load(f)
    print("Load tokenized data done.")

    # word2vec modeling & save
    model = Word2Vec(input_data, size=vector_dimension, window=5, min_count=10)
    # model = Word2Vec.load(os.path.join(os.getcwd(), "word2vec.model"))
    print("Word2vec modeling done.")

    # word vector L2 normalization
    model.init_sims(True)

    # get word name list
    words = list(model.wv.vocab.keys())

    # get euclidean distance matrix
    word_dim_mat = np.zeros((len(words), vector_dimension))
    for idx, word in enumerate(words):
        word_dim_mat[idx] = model.wv.__getitem__(word)
    euc_mat = euclidean_distances(word_dim_mat)
    print("Generate euclidean distance matrix done.")

    # disconnect useless euclidean distance in matrix
    min_distance = np.min(euc_mat)
    max_distance = np.max(euc_mat)
    # epsilon_value = [(max_distance - min_distance) * round(rate, 1) for rate in np.arange(0, 1, 0.1)]
    epsilon_value = (max_distance - min_distance) * 0.9
    # euc_mat_discon_by_eps = np.where(euc_mat > epsilon_value[8], euc_mat, 0)
    euc_mat_discon_by_eps = np.where(euc_mat > epsilon_value, euc_mat, 0)

    # check if all connection disconnected
    zero_row = []
    for row in range(len(euc_mat)):
        if np.any(euc_mat_discon_by_eps[row])==False:
            zero_row.append(row)

    # reconnect minimum value
    if zero_row:
        for row in zero_row:
            euc_row = euc_mat[row]
            min_value = np.min(euc_row[np.nonzero(euc_row)]) # get minimum nonzero value
            non_zero_idx = np.where(euc_row==min_value)
            replaced_row = np.zeros_like(euc_row)
            replaced_row[non_zero_idx] = min_value
            euc_mat_discon_by_eps[row] = replaced_row

    euc_mat = euc_mat_discon_by_eps
    print("Make distance matrix done.")

    # label propagation
    from label_propagation_algorithm import LabelPropagation
    # from sklearn.semi_supervised import LabelPropagation

    # pos_words = ["사랑","대박나고","소름","찡했어요","천재","온기","눈물","찡","굳","대박"]
    pos_words = ["사랑","소름","천재","눈물","찡","굳","대박"]
    # neg_words = ["과로사","틀렸어","개뿔","빌런","시끄럽게","과도","거짓말","실망하신듯","어색한","걱정"]
    neg_words = ["개뿔","빌런","시끄럽게","과도","거짓말","어색한","걱정"]
    pos_idx = [words.index(word) for word in pos_words]
    neg_idx = [words.index(word) for word in neg_words]
    euc_label = np.zeros_like(words, dtype=float)
    euc_label[pos_idx] = 1
    euc_label[neg_idx] = -1

    labelprop_model = LabelPropagation()
    labelprop_model.affinity_matrix = euc_mat
    labelprop_model.fit(word_dim_mat, euc_label)
    print("Label propagation done.")
    print("total modeling time : %s seconds" % (time.time() - start_time))

    # make saved directory
    try:
        os.mkdir(os.path.join(os.getcwd(), "saved_model"))
    except FileExistsError as e:
        pass

    # save word2vec model
    model.save(os.path.join(os.getcwd(), "saved_model/word2vec.model"))

    # save label propagation model
    pickle.dump(labelprop_model, open(os.path.join(os.getcwd(), "saved_model/labelprop_model"), 'wb'))

    # save result
    with open(os.path.join(os.getcwd(), 'saved_model/result.json'), "w", encoding='utf-8') as w:
        for idx, word in enumerate(words):
            w.write(json.dumps({"word": word, "pred_score": labelprop_model.label_distributions_[idx].tolist()}, ensure_ascii=False))
            w.write("\n")
