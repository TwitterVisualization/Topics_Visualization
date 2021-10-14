import tqdm
import numpy as np
np.set_printoptions(suppress=True)
from gensim.models import Word2Vec

w2v_path = "/mlo-container-scratch/hartley/twitter_covid_insights/insights_All/word2vec_300d.model"
model = Word2Vec.load(w2v_path)

vocab = model.wv.index_to_key
vectors = model.wv.vectors

with open("/mlo-container-scratch/hartley/twitter_covid_insights/insights_All/word2vec_prakhar.txt","w", encoding="utf8") as f :
    for i in tqdm.tqdm(range(len(vocab)+1)) :
        if i==0 :
            f.write(f"{len(vocab)} {len(vectors[0])}\n")
        else :
            endstr = "" if i == len(vocab) else "\n"
            vector_str = " ".join(vectors[i-1].astype(str))
            f.write(f"{vocab[i-1]} {vector_str}{endstr}")