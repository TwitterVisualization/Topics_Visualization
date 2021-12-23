import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ipywidgets import interact, fixed
import warnings
from tqdm import tqdm
from collections import defaultdict, Counter


TRENDS_PATH = "/mlodata1/prakhar/twitter_covid_insights/insights_All/trends.pkl"
TOPICS_PATH = "/mlodata1/prakhar/twitter_covid_insights/insights_All/topics.pkl"
TWEETS_PIPED_PATH = "/mlodata1/prakhar/all_available_tweets_piped"
CLUSTERED_TWEETS_PATH = "/mlodata1/raileanu/clustered_sampled_tweets"

topics = pkl.load(open(TOPICS_PATH, 'rb'))[0]

analyzed_hashtags = ['#vaccine']
hashtag_topic_id = {hashtag: -1  for hashtag in analyzed_hashtags}

for topic_id in topics.keys():
    for hashtag in analyzed_hashtags:
        if hashtag in topics[topic_id]:
            hashtag_topic_id[hashtag] = topic_id
            
for hashtag in analyzed_hashtags:
    if hashtag_topic_id[hashtag] != -1:
        topic_hashtags = topics[hashtag_topic_id[hashtag]]
        
counts = {hashtag: {} for hashtag in analyzed_hashtags}

for file in tqdm(os.listdir(TWEETS_PIPED_PATH)):
    piped_tweets = pd.read_parquet(os.path.join(TWEETS_PIPED_PATH, file))
    piped_tweets['tokenized_hashtags'] = piped_tweets['hashtags'].apply(lambda x: x.split(','))
    country_hashtags = piped_tweets.groupby('country_code', as_index=False)['tokenized_hashtags'].agg(sum)
    country_hashtags['tokenized_hashtags'] = country_hashtags['tokenized_hashtags'].apply(lambda x: [elem for elem in x if elem != ''])
    date = file.replace('parsed_', '').replace('.parquet', '')
    for index, row in country_hashtags.iterrows():
        country = row['country_code']
        if country not in counts[hashtag]:
            counts[hashtag][country] = {}
        counts[hashtag][country][date] = 0
        tokenized_hashtags = Counter(row['tokenized_hashtags'])
        for hashtag in analyzed_hashtags:
            for topic_hashtag in topic_hashtags:
                counts[hashtag][country][date] += tokenized_hashtags[topic_hashtag]

with open('/mlodata1/raileanu/country_trends.pkl', 'wb') as f:
    pkl.dump(counts, f)