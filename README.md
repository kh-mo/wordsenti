# wordsenti

## papers
- project paper : [Word Sentiment Score Evaluation based on Graph-Based Semi-Supervised Learning and Word Embedding](http://jkiie.snu.ac.kr/index.php/journal/article/download/336/pdf), accepted JKIIE 2017
- word2vec : [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- label propagation : [Learning from Labeled and Unlabeled Data with Label Propagation](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf)

## Getting Start
### Crawling data
- Download in current space
- Output : raw_data folder, json file
```shell
python crawling.py
```

### Preprocessing
- Input : raw_data folder, json file
- Output : preprocessed json file, tokenized pickle file
```shell
python preprocessing.py
```

### Modeling
- input : preprocessed json file, tokenized pickle file
- output : word2vec model, label propagation model, result file
```shell
python modeling.py
```
