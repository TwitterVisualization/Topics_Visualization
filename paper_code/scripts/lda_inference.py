# Base and Cleaning
import os
import pandas as pd
import re
import string
import logging

# Visualizations
import pyLDAvis.gensim_models

# Natural Language Processing (NLP)
import spacy
import advertools as adv
import stopwordsiso
from spacy.tokenizer import Tokenizer
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

TWEETS_SAMPLE_1M = '/mlodata1/raileanu/tweets_sample_1M.txt'
TWEETS_PROCESSED_1M = '/mlodata1/raileanu/tweets_sample_1M_lda.csv'
LDA_MODEL_OUTPUT = '/mlodata1/raileanu/lda_results/models'
LDA_HTML_OUTPUT = '/mlodata1/raileanu/lda_results'
LDA_MODEL_FNAME = "index-266__n_topics-100__coherence-5358.bin"
LDA_HTML_FNAME = "index-266__n_topics-100__coherence-5358__1M.html"


def remove_emojis(text):
    return str(text).encode('ascii', 'ignore').decode('ascii')


def get_lemmas(text):
    lemmas = []
    doc = nlp(text)
    for token in doc:
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON'):
            lemmas.append(token.lemma_)
    return lemmas


def tokenize(text):
    tokens = re.sub(r'[^a-zA-Z 0-9]', '',
                    text)  # Remove text that doesn't contain letters or numbers
    tokens = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    tokens = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    tokens = tokens.lower().split()  # Make text lowercase and split it
    return tokens


if __name__ == "__main__":
    if os.path.isfile(TWEETS_PROCESSED_1M):
        sample_tweets_df = pd.read_csv(TWEETS_PROCESSED_1M,
                                       dtype={"tweet": str, "string_token": str,
                                              "string_lemma": str,
                                              "lemma_token": str})
        sample_tweets_df["lemma_token"] = sample_tweets_df["lemma_token"].apply(
            lambda x: x[1:-1].split(','))
        print("Sample tweets loaded!")
        sample_tweets_df.head()
    else:
        lines = []
        with open(TWEETS_SAMPLE_1M) as file:
            lines = file.read().splitlines()
        sample_tweets_df = pd.DataFrame(lines, columns=['tweet'])
        # Remove emojis
        sample_tweets_df['tweet'] = sample_tweets_df['tweet'].apply(lambda x: remove_emojis(x))
        print("Emojis removed!")
        # Tokenization
        nlp = spacy.load('en_core_web_lg')
        multilingual_nlp = spacy.load('xx_ent_wiki_sm')
        # Tokenizer
        tokenizer = Tokenizer(nlp.vocab)

        # Custom stopwords can be added
        custom_stopwords = ['\n', '\n\n', '&amp;', ' ', '.', '-', '$', '@', '!', '?', '..',
                            '\'', '+', '=', '-', '~', '//', '/', ':', '+.', 'de', 'la', 'el',
                            'y', 'en', 'se', 'es', 'las', 'un', 'lo', 'si', 'ms',
                            'por', 'los', 'con', 'para', 'del', 'una', 'pero', 'todo',
                            'le', 'les', 'et', 'pas', 'des', 'pour', 'que', 'est',
                            'die', 'und', 'der', 'ist', 'das', 'nicht', 'ich', 'zu', 'den',
                            'al', 'q', 'o', 'sin', 'este', 'te', 'son', 'nos', 'c', 'ser',
                            'qu', 'esta', 'mi', 'hay', 'e', 'da', 'com', 'um', 'pra',
                            'na', 'em', 'uma', 'os', 'mais', 's', 'mas', 'tem', 't', 'j',
                            'ser', 'vai', 'ele', 'isso', 'sem', 'meu', 'foi', 'l', 'qui',
                            'd', 'ce', 'une', 'au', 'il', 'vous', 'sur', 'dans', 'par', 'p',
                            'n', 'avec', 's', ',', '.user_placeholder', 'vai', 'r', 'z',
                            'como', 'su', 'ya', 'user_placeholder', 'url_placeholder']

        # Customize stop words by adding to the default list
        STOP_WORDS = nlp.Defaults.stop_words.union(custom_stopwords).union(
            multilingual_nlp.Defaults.stop_words)
        # ALL_STOP_WORDS = spacy (EN + XX) + gensim + wordcloud + advtools + stopwordsiso
        ALL_STOP_WORDS = STOP_WORDS.union(SW).union(stopwords)
        for language, swords in adv.stopwords.items():
            ALL_STOP_WORDS.union(swords)
        for lang in stopwordsiso.langs():
            ALL_STOP_WORDS.union(stopwordsiso.stopwords(lang))

        tokens = []
        for doc in tokenizer.pipe(sample_tweets_df['tweet'], batch_size=512):
            doc_tokens = []
            for token in doc:
                if token.text.lower() not in STOP_WORDS:
                    doc_tokens.append(token.text.lower())
            # Make tokens a string again
            tokens.append(doc_tokens)

        # Makes tokens column
        sample_tweets_df['string_token'] = [' '.join(map(str, l)) for l in tokens]
        print("Tokens made!")
        # Lemmatization
        sample_tweets_df['string_lemma'] = [' '.join(map(str, l)) for l in
                                            sample_tweets_df['string_token'].apply(get_lemmas)]
        sample_tweets_df['lemma_token'] = sample_tweets_df['string_lemma'].apply(tokenize)
        print("Lemmas made!")
        sample_tweets_df.to_csv(TWEETS_PROCESSED_1M, index=False)
        print("Sample tweets saved!")

    # Modeling
    id2word = Dictionary(sample_tweets_df['lemma_token'])
    id2word.filter_extremes(no_below=2, no_above=.99)
    corpus = [id2word.doc2bow(d) for d in sample_tweets_df['lemma_token']]
    # lda_model = LdaMulticore.load(os.path.join(LDA_MODEL_OUTPUT, LDA_MODEL_FNAME))
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=id2word,
                             num_topics=100,
                             random_state=100,
                             chunksize=100,
                             passes=10,
                             alpha='asymmetric',
                             eta=0.31,
                             workers=8)

    words = [re.findall(r'"([^"]*)"', t[1]) for t in lda_model.print_topics()]
    # Create Topics
    topics = [' '.join(t[0:10]) for t in words]
    # Getting the topics
    for id, t in enumerate(topics):
        print(f"------ Topic {id} ------")
        print(t, end="\n\n")
    base_perplexity = lda_model.log_perplexity(corpus)
    print('\nPerplexity: ', base_perplexity)

    # Compute Coherence Score
    coherence_model = CoherenceModel(model=lda_model, texts=sample_tweets_df['lemma_token'],
                                     dictionary=id2word, coherence='c_v')
    coherence_lda_model_base = coherence_model.get_coherence()
    print('\nCoherence Score: ', coherence_lda_model_base)
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(lda_vis, os.path.join(LDA_HTML_OUTPUT, LDA_HTML_FNAME))
