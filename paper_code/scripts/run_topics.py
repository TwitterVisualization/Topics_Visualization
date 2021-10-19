import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import sys
from scipy.stats import pearsonr
from hash_utils import *

########### CONFIG ############

# Ignore existing files
force_compute=False

# Limit computation to few files (useful for testing)
nb_files=None

# ISO-2 country code (read only tweets from this country, None for all countries)
country_code = None
country_string = country_code if country_code is not None else "All"

# Word2Vec params
w2v_dim        = 300
w2v_window     = 100
w2v_min_counts = 10
w2v_workers    = 30

# General files and folders
user_path =   "/scratch/hartley"
analysis_path = os.path.join(user_path, "twitter_covid_insights") # Where results will be stored
tweets_path = os.path.join(user_path, "tweets")                      # Where raw df tweets are 
tweets_piped_path = tweets_path + '_piped'                           # Where clean processed tweets will be stored 
sent_classifier_path = os.path.join(user_path, "sentiment140/sentiment_classifier.pkl")
country_path = os.path.join(analysis_path, f"insights_{country_string}")
lang_text_path    = os.path.join(country_path, "lang_files")

if not os.path.isdir(country_path) :
    os.makedirs(country_path)

# Computation Backup files
texts_path           = os.path.join(country_path,  "texts.txt")
word_counts_path     = os.path.join(country_path,  "word_counts.pkl")
w2v_path             = os.path.join(country_path,  f"word2vec_{w2v_dim}d.model")
topics_path          = os.path.join(country_path,  "topics_700_occ400.pkl")
growth_path          = os.path.join(country_path,  "topics_700__occ400_growth.pkl") 
raw_trends_path      = os.path.join(country_path,  "trends_raw.pkl")
weighted_trends_path = os.path.join(country_path,  "trends.pkl")
what_country_path    = os.path.join(analysis_path, "what_country.pkl")
epi_path             = os.path.join(analysis_path, "model_data_owid.csv")
report_path          = os.path.join(country_path,  "correlations_report.txt")
day_flux_path        = os.path.join(country_path,  "day_flux.pkl")

# What was computed already
counts_computed    = os.path.isfile(word_counts_path)
word2vec_computed = os.path.isfile(w2v_path)
topics_computed   = os.path.isfile(topics_path)
sent_computed     = os.path.isdir(lang_text_path)
what_country_computed = os.path.isfile(what_country_path)
raw_trends_computed = os.path.isfile(raw_trends_path)
weighted_trends_computed = os.path.isfile(weighted_trends_path) and os.path.isfile(day_flux_path)


########### Dataframe cleaning and word counts ############
if counts_computed and not force_compute:    
    print("Step 1/5 : Dataframe was already prepared")
    if not word2vec_computed or not topics_computed or force_compute :
        word_counts = np.array(pkl.load(open(word_counts_path, "rb")))
    
else :
    print(f"Step 1/5 : Preparing df and counting words, dumping in {word_counts_path}")
    word_counts = np.array(prepare_df(tweets_path, texts_path, tweets_piped_path, country_code=country_code, nb_files=nb_files))
    pkl.dump(word_counts, open(word_counts_path, "wb"), protocol=4)
    
#print('word_counts')
#print(word_counts)


########### Compute Hashtags Embeddings ############
if word2vec_computed and not force_compute:
    print("Step 2/5 : Word2Vec embeddings were already computed")
    if True or not topics_computed or force_compute :
        pass
        #model = Word2Vec.load(w2v_path)

else :
    print(f"Step 2/5 : Computing Word2Vec embeddings, dumping in {w2v_path}")
    sentences = LineSentence(texts_path)
    model = Word2Vec(sentences, vector_size=w2v_dim, window=w2v_window, min_count=w2v_min_counts, workers=w2v_workers)
    model.save(w2v_path)

model = pkl.load(open('/scratch/hartley/twitter_covid_insights/insights_All/s2v_dict_700.pkl', 'rb'))

########### Topics creation ############
if topics_computed and not force_compute :
    print("Step 3/5 : Topics were already computed")
else :
    print(f"Step 3/5 : Computing topics, dumping in {topics_path}")
    find_topics(model, word_counts, topics_path, max_absorption=100, min_clust_size=3, growth_path=growth_path, s2v=True)