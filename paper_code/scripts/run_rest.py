import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import sys
from hash_utils import *

########### CONFIG ############

# Ignore existing files
force_compute=False

# Limit computation to few files (useful for testing)
nb_files=None

# ISO-2 country code (read only tweets from this country, None for all countries)
country_code = None
country_string = country_code if country_code is not None else "All"

# Sent2vec path
s2v_dim = 300
s2v_ngrams = 2
s2v_voc_size = 5_000_000
s2v_min_count = 20

# Word2Vec params
w2v_dim        = 300
w2v_window     = 100
w2v_min_counts = 10
w2v_workers    = 30

# Whether to use s2v or w2v
use_s2v = True 

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
w2v_path             = os.path.join(country_path,  f"word2vec_{w2v_dim}d_{w2v_window}w_{w2v_min_count}mc.model")
s2v_out_path         = os.path.join(country_path,  f"sent2vec_{s2v_dim}d_{s2v_ngrams}ng_{s2v_voc_size}vs_{s2v_min_count}mc.txt")
#s2v_model_path       =  os.path.join(country_path,  f"sent2vec_{s2v_dim}d_{s2v_ngrams}ng_{s2v_voc_size}vs_{s2v_min_count}mc.pkl")
s2v_model_path       =  os.path.join(country_path, "s2v_dict_700.pkl")
topics_path          = os.path.join(country_path,  "topics_700.pkl")
growth_path          = os.path.join(country_path,  "topics_700_growth.pkl") 
raw_trends_path      = os.path.join(country_path,  "trends_raw.pkl")
weighted_trends_path = os.path.join(country_path,  "trends.pkl")
what_country_path    = os.path.join(analysis_path, "what_country.pkl")
epi_path             = os.path.join(analysis_path, "model_data_owid.csv")
report_path          = os.path.join(country_path,  "correlations_report.txt")
day_flux_path        = os.path.join(country_path,  "day_flux.pkl")

# What was computed already
counts_computed    = os.path.isfile(word_counts_path)
sent2vec_computed = os.path.isfile(s2v_model_path)
word2vec_computed = os.path.isfile(w2v_path)
topics_computed   = os.path.isfile(topics_path)
sent_computed     = os.path.isdir(lang_text_path)
what_country_computed = os.path.isfile(what_country_path)
raw_trends_computed = os.path.isfile(raw_trends_path)
weighted_trends_computed = os.path.isfile(weighted_trends_path) and os.path.isfile(day_flux_path)


########### Dataframe cleaning and word counts ############
if counts_computed and not force_compute:    
    print("Step 1/5 : Dataframe was already prepared")
    if not word2vec_computed or force_compute :
        word_counts = np.array(pkl.load(open(word_counts_path, "rb")))
    
else :
    raise Exception('yo1')
    print(f"Step 1/5 : Preparing df and counting words, dumping in {word_counts_path}")
    word_counts = np.array(prepare_df(tweets_path, texts_path, tweets_piped_path, country_code=country_code, nb_files=nb_files))
    pkl.dump(word_counts, open(word_counts_path, "wb"), protocol=4)
    
#print('word_counts')
#print(word_counts)


########### Compute Hashtags Embeddings ############

if use_s2v and not sent2vec_computed:
    if not sent2vec_computed :
        raise Exception('yo2')
        cmd = f"./fasttext sent2vec -input {texts_path} -output {s2v_out_path} -wordNgrams {s2v_ngrams} -dim {s2v_dim} -maxVocabSize {s2v_voc_size} -minCount {s2v_min_count} -lr 0.2 -epoch 1 -t 0.000005 -bucket 4000000 -dropoutK 4 -neg 10 -thread 24"
        os.system(cmd)
        s2v_txt_to_dict(s2v_out_path, s2v_model_path, s2v_dim)
    else :
        model = pkl.load(open(s2v_model_path, 'rb'))

else :    
    if word2vec_computed and not force_compute:
        print("Step 2/5 : Word2Vec embeddings were already computed")
        if not topics_computed or force_compute :
            model = Word2Vec.load(w2v_path)

    else :
        raise Exception('yo3')
        print(f"Step 2/5 : Computing Word2Vec embeddings, dumping in {w2v_path}")
        sentences = LineSentence(texts_path)
        model = Word2Vec(sentences, vector_size=w2v_dim, window=w2v_window, min_count=w2v_min_counts, workers=w2v_workers)
        model.save(w2v_path)

########### Topics creation ############
if topics_computed and not force_compute :
    print("Step 3/5 : Topics were already computed")
else :
    raise Exception('yo4')
    print(f"Step 3/5 : Computing topics, dumping in {topics_path}")
    find_topics(model, word_counts, topics_path, max_absorption=100, min_clust_size=5, growth_path=growth_path)


########### Sentiment labelling ############

if sent_computed and not force_compute :
    print("Step 4/5 : Sentiments were already computed")
else :
    raise Exception('yo5')
    print(f"Step 4/5 : Computing sentiment labels, dumping lang files in {lang_text_path}")
    sent_labels = label_sentiments(tweets_piped_path, sent_classifier_path, lang_text_path)

########### Topic trends derivation ############    
    
if raw_trends_computed and not force_compute :
    print("Step 5/5 part A : Raw topic trends were already computed")
else :
    print(f"Step 5/5 part A : Computing raw topic trends, dumping in {raw_trends_path}")
    
    sub_trends, higher_trends = topic_trends(tweets_piped_path, topics_path, country_code=country_code)
    pkl.dump((sub_trends, higher_trends), open(raw_trends_path, 'wb'), protocol=4)
    
if weighted_trends_computed and not force_recompute :
    print("Step 5/5 part B : weighted topic trends were already computed")
else :
    print(f"Step 5/5 part B : Computing weighted topic trends, dumping in {weighted_trends_path}")
    weighted_topic_trends(tweets_path, tweets_piped_path, day_flux_path, topics_path, weighted_trends_path)

