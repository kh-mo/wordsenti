# wordsenti

[Paper](http://jkiie.snu.ac.kr/index.php/journal/article/download/336/pdf)

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
