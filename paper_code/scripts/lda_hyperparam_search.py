# Base and Cleaning
import tqdm
import pandas as pd
import numpy as np

# Natural Language Processing (NLP)
import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


def compute_coherence_values(corpus, dictionary, k, p, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=p,
                                           alpha=a,
                                           eta=b,
                                           workers=8)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=sample_tweets_df['lemma_token'],
                                         dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()


TWEETS_PROCESSED = '/mlodata1/raileanu/tweets_sample_lda.csv'
MODEL_RESULTS = '/mlodata1/raileanu/lda_model_results.csv'
MODEL_RESULTS_TEMP = '/mlodata1/raileanu/lda_model_results_temp.csv'

if __name__ == "__main__":
    sample_tweets_df = pd.read_csv(TWEETS_PROCESSED,
                                   dtype={"tweet": str, "string_token": str, "string_lemma": str,
                                          "lemma_token": str})
    sample_tweets_df["lemma_token"] = sample_tweets_df["lemma_token"].apply(
        lambda x: x[1:-1].split(','))
    print("Sample tweets loaded!")
    sample_tweets_df.head()
    # Create an id2word dictionary
    id2word = Dictionary(sample_tweets_df['lemma_token'])
    # Filtering Extremes
    id2word.filter_extremes(no_below=2, no_above=.99)
    # Creating a corpus
    corpus = [id2word.doc2bow(d) for d in sample_tweets_df['lemma_token']]

    grid = {'Validation_Set': {}}
    # Topics range
    min_topics = 5
    max_topics = 100
    step_size = 5
    # topics_range = range(min_topics, max_topics, step_size)
    topics_range = [10, 25, 50, 75, 100]
    # Num of Passes
    passes = range(10, 20, 5)
    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')
    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
        corpus]
    corpus_title = [  # '75% Corpus',
        '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Passes': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': [],
                     }

    pbar_total = len(corpus_sets) * len(topics_range) * len(passes) * len(alpha) * len(beta)
    pbar = tqdm.tqdm(total=pbar_total)
    index = 0
    # iterate through validation corpus
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through passes
            for p in passes:
                # iterate through alpha values
                for a in alpha:
                    # iterate through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        print(f"Running model with parameters:\n"
                              f"{corpus_title[i]}\n"
                              f"\tTopics: {k}\n\tPasses: {p}\n\tAlpha: {a}\n\tBeta: {b}\n")
                        cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                      k=k, p=p, a=a, b=b)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Passes'].append(p)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)
                        pbar.update(1)
                        temp_results = pd.DataFrame({
                            'Validation_Set': corpus_title[i],
                            'Topics': k,
                            'Passes': p,
                            'Alpha': a,
                            'Beta': b,
                            'Coherence': cv
                        }, index=[index])
                        pd.DataFrame(temp_results).to_csv(MODEL_RESULTS_TEMP, mode='a', header=False)
                        index += 1
    pd.DataFrame(model_results).to_csv(MODEL_RESULTS, index=False)
    pbar.close()
    print("Model results saved!")
