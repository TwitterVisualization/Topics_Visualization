import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
import langdetect
import shutil
import random
import string
import emoji
import multiprocessing as mp

os.environ["LASER"] = "/home/massemin/LASER"

def random_string(len_string=20) :
    letters = string.ascii_lowercase
    return  ''.join(random.choice(letters) for i in range(len_string))

def preprocess_tweet(tweet, no_emoji=True):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
    tweet = re.sub('(\@[^\s]+)','<user>',tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    
    if no_emoji :
        tweet = "".join([c for c in tweet if c not in emoji.UNICODE_EMOJI])
        
    return tweet

def laser_embed_file(src, dst, lang) :
    os.chdir("/home/massemin/LASER/tasks/embed/")
    command = "bash ./embed.sh {} {} {}".format(src, lang, dst)
    os.system(command)
    
def load_laser_embs(file, count=-1, offset=0, dim=1024) :
    embeddings = np.fromfile(file, dtype=np.float32, offset=offset*4, count=count*dim)
    embeddings.resize(embeddings.shape[0] // dim, dim)
    return embeddings

def laser_embed_texts(texts, lang, preprocess=True) :
    
    ### Write texts in file ###
    
    file_id = random_string()
    file_path = f"/mlo-container-scratch/massemin/{file_id}_tmp_laser.txt"
    if os.path.isfile(file_path) :
        os.remove(file_path)
        
    try :
        if preprocess :
            texts = [preprocess_tweet(t.replace("\n", "")) for t in texts if t !="\n"]

        else :
            texts = [t.replace("\n", "") for t in texts if t !="\n"]

        with open(file_path, 'w',encoding="utf8") as f:
            for e, entry in enumerate(texts) :

                if e != 0 :
                    f.write("\n" + entry)
                else :
                    f.write(entry)

        # Embed the file
        output_file = f"/mlo-container-scratch/massemin/{file_id}_tmp_laser.raw"
        laser_embed_file(file_path, output_file, lang)
        
        # Load its embeddings
        embeddings = load_laser_embs(output_file)
        
        # Remove output file and input file
        os.remove(file_path)
        os.remove(output_file)
        
        return embeddings
    
    except :
        os.remove(file_path)
    
    
