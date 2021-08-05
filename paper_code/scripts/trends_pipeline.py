import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import sys
import seaborn as sns
from scipy.stats import pearsonr

from hash_utils import *

########### CONFIG ############

# Limit computation to few files (useful for testing)
nb_files=30 #None

# ISO-2 country code (read only tweets from this country, None for all countries)
country_code = None
country_string = country_code if country_code is not None else "All"

# Word2Vec params
w2v_dim        = 150
w2v_window     = 100
w2v_min_counts = 10
w2v_workers    = 30

# General files and folders
user_path =   "/mlo-container-scratch/massemin"
analysis_path = os.path.join(user_path, "twitter_covid_insights_v4") # Where results will be stored
tweets_path = os.path.join(user_path, "tweets")                      # Where raw df tweets are 
tweets_piped_path = tweets_path + '_piped'                           # Where clean processed tweets will be stored 
sent_classifier_path = os.path.join(user_path, "sentiment_140.pkl")
country_path = os.path.join(analysis_path, f"insights_{country_string}")
lang_text_path    = os.path.join(country_path, "lang_files")

if not os.path.isdir(country_path) :
    os.makedirs(country_path)

# Computation Backup files
texts_path        = os.path.join(country_path,  "texts.txt")
word_counts_path  = os.path.join(country_path,  "word_counts.pkl")
w2v_path          = os.path.join(country_path,  f"word2vec_{w2v_dim}d.model")
topics_path       = os.path.join(country_path,  "topics.pkl")
growth_path       = os.path.join(country_path, "topics_growth.pkl") 
raw_trends_path   = os.path.join(country_path,  "trends_raw.pkl")
what_country_path = os.path.join(analysis_path, "what_country.pkl")
epi_path          = os.path.join(analysis_path, "model_data_owid.csv")
report_path       = os.path.join(country_path,  "correlations_report.txt")

# What was computed already
counts_computed    = os.path.isfile(word_counts_path)
word2vec_computed = os.path.isfile(w2v_path)
topics_computed   = os.path.isfile(topics_path)
sent_computed     = os.path.isdir(lang_text_path)
what_country_computed = os.path.isfile(what_country_path)
trends_computed = os.path.isfile(raw_trends_path)


########### Dataframe cleaning and word counts ############
if counts_computed :    
    print("Step 1/5 : Dataframe was already prepared, loading word counts.")
    word_counts = pkl.load(open(word_counts_path, "rb"))
    
else :
    print(f"Step 1/5 : Preparing df and counting words, dumping in {word_counts_path}")
    word_counts = prepare_df(tweets_path, texts_path, tweets_piped_path, country_code=country_code, nb_files=nb_files)
    pkl.dump(word_counts, open(word_counts_path, "wb"))


########### Compute Hashtags Embeddings ############
if word2vec_computed :
    print("Step 2/5 : Word2Vec embeddings were already computed, loading them.")
    model = Word2Vec.load(w2v_path)

else :
    print(f"Step 2/5 : Computing Word2Vec embeddings, dumping in {w2v_path}")
    sentences = LineSentence(texts_path)
    model = Word2Vec(sentences, size=w2v_dim, window=w2v_window, min_count=w2v_min_counts, workers=w2v_workers)
    model.save(w2v_path)

########### Topics creation ############
if topics_computed :
    print("Step 3/5 : Topics were already computed")
else :
    print(f"Step 3/5 : Computing topics, dumping in {topics_path}")
    find_topics(model, word_counts, topics_path, max_absorption=100, min_clust_size=5, growth_path=growth_path)
    
topics = pkl.load(open(topics_path, 'rb'))
    

########### Sentiment labelling ############

if sent_computed :
    print("Step 4/5 : Sentiments were already computed")
else :
    print(f"Step 4/5 : Computing sentiment labels, dumping lang files in {lang_text_path}")
    sent_labels = label_sentiments(tweets_piped_path, sent_classifier_path, lang_text_path)

########### Topic trends derivation ############    
    
if trends_computed :
    print("Step 5/5 : Topic trends were already computed")
else :
    print(f"Step 5/5 : Computing topic trends, dumping in {raw_trends_path}")
    
    sub_trends, higher_trends = topic_trends(tweets_piped_path, topics, country_code=None)
    pkl.dump((sub_trends, higher_trends), open(raw_trends_path, 'wb'))