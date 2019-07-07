import os
import json
from konlpy.tag import Okt

if __name__ == "__main__":
    okt = Okt()

    with open(os.path.join(os.getcwd(), 'raw_data/data.json'), encoding='utf-8') as r:
        with open(os.path.join(os.getcwd(), 'raw_data/preprocessed_data.json'), "w", encoding='utf-8') as w:
            for idx, line in enumerate(r):
                file = json.loads(line)
                review = file['review'].strip()
                # get tokenized file &
                tokenize_review = okt.morphs(review)

                # write to json file
                w.write(json.dumps({"score": file['score'],
                                    "review": review,
                                    "tokenized_review": tokenize_review}, ensure_ascii=False))
                w.write("\n")

                # show processing
                if idx % 1000 == 0:
                    print("preprocessing :", idx)