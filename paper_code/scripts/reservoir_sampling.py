import os.path
from tqdm import tqdm
import random

def get_tweets_random_sample(file, n_lines):
    it = iter(file)
    try:
        result = [next(it) for _ in range(n_lines)]
    except StopIteration:
        raise ValueError("Sample larger than population")

    for i, item in tqdm(enumerate(it, start=n_lines)):
        s = random.randint(0, i)
        if s < n_lines:
            result[s] = item
    return result

TWEETS_SAMPLE = '/mlodata1/raileanu/tweets_sample_10M.txt'
TEXTS_PATH = '/mlodata1/prakhar/twitter_covid_insights/insights_All/texts.txt'

if not os.path.isfile(TWEETS_SAMPLE):
    tweets_sample_file = open(TWEETS_SAMPLE, 'w')
    with open(TEXTS_PATH) as file:
        for line in get_tweets_random_sample(file, 10000000):
            tweets_sample_file.write(line)