import nltk
from nltk import word_tokenize
import pandas as pd
from tqdm import tqdm_notebook, tqdm
import multiprocessing as mp
import os

tweets_piped_path = "/mlo-container-scratch/hartley/tweets_piped"
texts_path = "/mlo-container-scratch/hartley/old_tweets_text.txt"

def tokenize_text(text) :  
    return word_tokenize(text.lower())

# Files containing tweets
nb_files = 429
files = [os.path.join(tweets_piped_path,f) for f in sorted(os.listdir(tweets_piped_path)) if '.txt' not in f]
files = files[:nb_files] if nb_files is not None else files        

pool = mp.Pool(20)
for f_idx, f in tqdm(enumerate(files), total=len(files)) :
    #print('processing', f)
    df = pd.read_parquet(f, columns=["cleaned_text", "hashtags"]).astype('unicode')

    if len(df) == 0 :
        continue
        
    # Tokenize cleaned_texts and append hashtags
    texts = pool.map(tokenize_text, df.cleaned_text.values)
    for e, t in enumerate(texts) :
        texts[e] += [x for x in df.hashtags.values[e].split(',')]

    # Save tweets in a file for Word2Vec
    mode = 'w' if f_idx == 0 else 'a'
    with open(texts_path, mode, encoding='utf8') as filehandle:
        filehandle.writelines(' '.join(t) + '\n' for t in texts)