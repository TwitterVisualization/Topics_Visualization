import numpy as np
import emoji
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pandas as pd
from tqdm import tqdm_notebook, tqdm
import multiprocessing as mp
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr
import os
import pickle as pkl
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import multiprocessing as mp
import io
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1" # GPU id

from laser_embedder import *

stop_words = stopwords.words('english')
porter = PorterStemmer()

def clean_text(text) :
    
    if text is None :
        return ''
    
    # No url
    t = re.sub(r'https?:\/\/[^\s]*(\r|\n|\s|$)', '', text, flags=re.MULTILINE).strip()
    
    # Remove hashtags and user mentions
    t = re.sub(r'[@#][^\s]*(?:\r|\n|\s|$)', '', t).strip()
    
    # remove emojis
    t = "".join([c for c in t if c not in emoji.UNICODE_EMOJI])
    
    # remove new lines
    t = t.replace('\n', '')
    
    return t

def tokenize_text(text) :
    
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    stemmed = [porter.stem(word) for word in filtered_words]
    return stemmed

# Used only in read_hash, defined here for multiprocessed serialization
def find_hash(t) : return ['#'+x.lower() for x in re.findall(r"#(\w+)", t)] if not pd.isna(t) else []

def prepare_df(tweets_path, texts_path, tweets_piped_path, country_code=None, nb_files=None) :
    
    # Files containing tweets
    files = [os.path.join(tweets_path,f) for f in  sorted(os.listdir(tweets_path))]
    files = files[:nb_files] if nb_files is not None else files        
    
    c = Counter()
    pool = mp.Pool(20)
    for f_idx, f in tqdm(enumerate(files), total=len(files)) :
        
        df = pd.read_parquet(f, columns=["id", "country_code", "text", "lang"]).astype('unicode')
        
        if country_code is not None :            
            df = df[df.country_code == country_code]        
            
        if len(df) == 0 :
            continue
        
        # Spot hashtags, save them in column
        cur_hashtags = pool.map(find_hash, df.text.values)
        df['hashtags'] = np.array([",".join(h) for h in cur_hashtags])
        #df = df[df['hashtags'] != ''] => DON'T, hurts Word2Vec perf
        
        # Clean text -> Allows filtering        
        df['cleaned_text'] = pool.map(clean_text, df.text.values)
        df = df.drop_duplicates("cleaned_text")
        df = df[df.cleaned_text != '']
        df = df.drop('text', axis=1)
        df.reset_index(drop=True, inplace=True)
        
        # Save cleaned dataframe
        if not os.path.isdir(tweets_piped_path) : os.mkdir(tweets_piped_path)
        df_name = os.path.basename(f)
        save_path = os.path.join(tweets_piped_path, df_name)
        df.to_parquet(save_path)
        
        # Tokenize cleaned_texts and append hashtags
        texts = pool.map(tokenize_text, df.cleaned_text.values)
        for e, t in enumerate(texts) :
            texts[e] += [x for x in cur_hashtags[e]]
        
        # Save tweets in a file for Word2Vec
        mode = 'w' if f_idx == 0 else 'a'
        with open(texts_path, mode, encoding='utf8') as filehandle:
            filehandle.writelines(' '.join(t) + '\n' for t in texts)
        
        
        # Count hashtags
        c.update([h for x in df.hashtags.values for h in x.split(',') if h != ''])        
        
    return c.most_common()

def dbscan(model, counts_per_word, embeddings=None, sim_thresh=0.8, min_samples=5, min_occs=1000, verbose=False) :
    
    
    if embeddings is None :
        
        #print('COUNTS PER WORD:', counts_per_word[:, 1])
        
        # Keep only hashtags with more than min_occs occurences
        nb_to_keep = np.argmax(counts_per_word[:, 1].astype(int) < min_occs)
        if nb_to_keep == 0 :
            raise Exception(f'dbscan : No word with more than {min_occs} occurences')

        # Create fit data
        #model_words = set(model.wv.vocab.keys())
        model_words = set(model.wv.index_to_key)
        words_kept = np.array([word for word, count in counts_per_word[:nb_to_keep] if word in model_words])
        X = np.array([model.wv[w] for w in words_kept])
        
    else :
        X = embeddings
        #print('X2 :', X)
        words_kept = np.arange(len(X)).astype(str)

    # cosine DBScan
    clustering = DBSCAN(eps=1-sim_thresh, min_samples=min_samples, metric='cosine').fit(X)
    clust_labels = clustering.labels_
    
    if verbose :
        
        print(np.bincount(clust_labels+1)[1:])

        for e in range(clust_labels.max() + 1) :
            print(f"Topic {e} :")
            tags = np.array(counts_per_word)[:len(clust_labels)][clust_labels == e]
            for tag in tags :
                print(f"\t{tag}")
                
    return clust_labels, words_kept

def find_topics(model, word_counts, topics_path, max_absorption=150, min_clust_size=5, growth_path=None) :
    
    # First find sub topics
    topics = multiscale_dbscan(model, word_counts, None, max_absorption, min_clust_size, growth_path)
    if len(topics) == 0 :
        raise 'Not enough data to find significant topics'
    #print('TOPICS :', topics)
    embs = np.array([np.mean(model.wv[t], axis=0) for t in topics])
    
    # Consider higher topics
    meta_topics = multiscale_dbscan(None, None, embs, max_absorption, min_clust_size)
    
    # Turn topics into dictionnaries
    topics = dict(zip(np.arange(len(topics)), topics))
    meta_topics = dict(zip(np.arange(len(meta_topics)), meta_topics))
    
    # Save everything in memory
    pkl.dump((topics, meta_topics), open(topics_path, 'wb'))

def multiscale_dbscan(model, word_counts, embeddings=None, max_absorption=150, min_clust_size=5, growth_path=None) :
    
    clusters = []
    
    # Run iterations of dbscan
    val_to_try = np.arange(0.5, 1, 0.001)
    save_eps = (embeddings is None)
    
    for epsilon in tqdm(val_to_try) :
        if embeddings is None :
            clust_labels, words_kept = dbscan(model, 
                                              word_counts, 
                                              sim_thresh=epsilon, 
                                              min_samples=1, 
                                              min_occs=50, 
                                              verbose=False)
        else :
            clust_labels, words_kept = dbscan(None, 
                                              None, 
                                              embeddings=embeddings, 
                                              sim_thresh=epsilon, 
                                              min_samples=1, 
                                              min_occs=100, 
                                              verbose=False)
            
        clusters.append(clust_labels)

    clusters = np.array(clusters[::-1])
    wkept = words_kept
    
    if save_eps and (growth_path is not None):
        pkl.dump((val_to_try, clusters, wkept), open(growth_path, 'wb'))
    
    final_clusts = []
    invalid = set()
    for iteration_idx in range(len(clusters)) :

        if iteration_idx == 0 :
            # indices and sizes for first iteration of clusters
            cur_values, cur_indices, cur_counts = np.unique(clusters[iteration_idx], return_index=True, return_counts=True)
        else :
            cur_values, cur_indices, cur_counts = next_values, next_indices, next_counts

        if iteration_idx == len(clusters) - 1 :        
            # Termination, register labels that did not explode yet
            still_valid = [l for l in cur_values if l not in invalid]
            final_clusts += zip(still_valid, [iteration_idx]*len(still_valid))
            break

        # Check next iteration
        next_values, next_indices, next_counts = np.unique(clusters[iteration_idx+1], return_index=True, return_counts=True)

        # How many hashtags were absorbed
        new_sizes = next_counts[clusters[iteration_idx+1][cur_indices]]
        absorptions = new_sizes - cur_counts

        # Spot extreme expansions and remember correct scales
        exploded_labels = [x for x in cur_values[absorptions > max_absorption] if x not in invalid]
        final_clusts += zip(exploded_labels, [iteration_idx]*len(exploded_labels))

        # invalid labels at next iteration
        bad_labels = np.append(exploded_labels, [i for i in invalid]).astype(np.uint32)
        invalid = set(np.unique(clusters[iteration_idx+1][cur_indices[bad_labels]]))

    topics = [wkept[clusters[it_idx] == cl_idx] for cl_idx, it_idx in final_clusts]
    topics = [t for t in topics if len(t) > min_clust_size]
    
    return topics
    

def counts_cross_corr(topic_counts, nb_corrs=100) :
    
    counts_normalized = dict(topic_counts)

    # Normalize all counts
    for k, v in counts_normalized.items() :    
        topic_count = np.array(v)
        counts_normalized[k] = topic_count / topic_count.sum()
        
    # Compute cross correlation
    counts_df = pd.DataFrame(counts_normalized)
    cross_corr = counts_df.corr()

    # Plot it
    plt.matshow(cross_corr)
    plt.show()

    # Get rid of diagonal and half of matrix (symmetry)
    all_corrs = np.array(cross_corr.values)
    for i in range(len(all_corrs)) :
        all_corrs[i, :i+1] = 0    

    # Find Strongest correlations and their associated topics
    flat_to_2d = lambda x : [x//len(all_corrs), x % len(all_corrs)]
    argsrt = np.argsort(-all_corrs.flatten())
    
    strong_arg = np.append([argsrt][:nb_corrs], [argsrt][-nb_corrs:])    
    strong_corrs = all_corrs.flatten()[strong_arg]
    strong_corrs_idx = np.array([flat_to_2d(x) for x in strong_arg])
    
    corrs = list(zip(counts_df.columns[strong_corrs_idx[:, 0]], counts_df.columns[strong_corrs_idx[:, 1]], strong_corrs))

    return corrs, cross_corr.values
"""
def topn_corr(corr_matrix, corr_names, topn=10) :
    
    # Get rid of diagonal and half of matrix (symmetry)
    all_corrs = np.array(corr_matrix)
    for i in range(len(all_corrs)) :
        all_corrs[i, :i+1] = 0    

    # Find highest correlations and their associated topics
    flat_to_2d = lambda x : [x//len(all_corrs), x % len(all_corrs)]
    argsrt = np.argsort(-all_corrs.flatten())[:topn]
    highest_corrs = all_corrs.flatten()[argsrt]
    highest_corrs_idx = np.array([flat_to_2d(x) for x in argsrt])
    
    topics_asso = zip(corr_names[highest_corrs_idx[:, 0]], corr_names[highest_corrs_idx[:, 1]])
    
    
    return topics_asso, highest_corrs_idx
    
    

def cooc_matrix(counts_per_word, clust_labels, topic_counts, hashtags) :
    topic_hashes = np.array(counts_per_word)[:len(clust_labels), 0][clust_labels !=-1]
    non_topic = [x for x in topic_counts.keys()if type(x) != int]
    all_hashes = np.concatenate((topic_hashes, non_topic))

    hash_keys = list(all_hashes)
    to_keep = set(all_hashes)
    coocs = np.zeros((len(hash_keys), len(hash_keys)))

    for h in tqdm(hashtags) :
        filtered = set(h).intersection(to_keep)
        if len(filtered) != 0 :
            indices = [hash_keys.index(f) for f in filtered]

            for i in indices :
                for j in indices :
                    coocs[i, j] += 1

    plt.figure(figsize=(15,5))
    plt.title("Hashtag co-occurences")
    plt.matshow(np.log(coocs + 1), fignum=0)
    plt.show()
    return coocs
"""  
def hashtag_stats(hashtags, coverage=False) :
    
    # Ratio of tagged hashtags
    tagged = [h for h in hashtags if len(h) > 0]
    hash_ratio = len(tagged) / len(hashtags) * 100
    print(f"{hash_ratio:2.3}% tweets have at least 1 hastag\n")
    
    # most common hashtags
    c = Counter()
    [c.update(hs) for hs in tagged]
    top_ten = c.most_common(n=10)
    print("10 most common hashtags :")
    for t in top_ten :
        print("\t", t)
        
    # Frequency distribution for hashtags
    plt.plot(np.log10(counts_per_word[:, 1].astype(int)))
    plt.title("Frequency Distribution of Hashtags")
    plt.xlabel("ranking")
    plt.ylabel("log_10 of occurences")
    plt.savefig("hashtags_frequency.png")
    plt.show()
    
    # Nb. of hashtags per tweet
    c = Counter([len(t) for t in tagged])
    nb_hashes = np.array(c.most_common())
    plt.bar(x=nb_hashes[:, 0], height=np.log10(nb_hashes[:, 1]))
    plt.title("Distribution of the Number of Hashtags per Tweet")
    plt.xlabel("Nb. of hashtags in the tweet")
    plt.ylabel("log_10 of the nb. of tweets")
    plt.savefig("hashtags_per_tweet.png")
    plt.show()
    
    # Hashtag coverage
    if coverage :
        untouched = np.array(tagged)
        coverage = []
        for word in tqdm(counts_per_word[:500, 0], desc="hashtag") :
            covered = 0
            untouched_indices = [] 

            for e, t in enumerate(tqdm(untouched, desc="tweet")) :
                if word in t :
                    covered +=1
                else :
                    untouched_indices.append(e)

            print(f"touched {covered} tweets, {len(untouched_indices)} remaining")
            coverage.append(covered)

            if len(untouched_indices) == 0 :
                break

            untouched = untouched[untouched_indices]

        plt.title("Hashtags Coverage")
        plt.plot(np.log10(coverage))
        plt.xlabel("hashtag rank in frequency")
        plt.ylabel("log10 of hashtag coverage")
        plt.show()
        
        
def label_sentiments(piped_path, sent_classifier_path, lang_text_path) :
    
    files = [os.path.join(piped_path,f) for f in  sorted(os.listdir(piped_path))]
    labels = []    
    if not os.path.isdir(lang_text_path) : os.makedirs(lang_text_path)
    
    # Sentiment classifier
    clf = pkl.load(open(sent_classifier_path, "rb"))
    
    # Build a text file for each language (for LASER)
    tweets_per_lang = defaultdict(lambda : 0)
    seen_lang = set()
    print("Generating text files for laser embeddings")
    for f in tqdm(files, desc='tweet-to-file') :
        
        df = pd.read_parquet(f, columns=["cleaned_text", "lang"])
        languages = df.lang.unique()
        
        for l in languages :
            file_name = os.path.join(lang_text_path, l + '.txt')            
            sub = df[df.lang == l]
            tweets_per_lang[l] += len(sub)

            # Write cleaned_text in file
            mode = 'a' if l in seen_lang else 'w'
            with open(file_name, mode, encoding='utf8') as filehandle:
                filehandle.writelines(t+'\n' for t in sub.cleaned_text.values)
                
        seen_lang.update(languages)
        
    def get_file_length(filename, chunk_size=1_000_000) :
        """Returns the length of a binary numpy file"""
        eof = False
        offset = 0
        total_length = 0
        while not eof :
            cur_length = len(np.fromfile(filename, dtype=np.float32, offset=offset, count=chunk_size))
            if cur_length == chunk_size :
                offset += 4*chunk_size
                total_length += cur_length
            else :
                return total_length + cur_length
                
    # Trigger LASER on each file
    print("Sanity check on laser embeddings")
    for f in tqdm(os.listdir(lang_text_path), desc='laser-check') :
        src = os.path.join(lang_text_path, f)
        dst = src.split('.')[0] + '.raw'
        lang = f.split('.')[0]
        
        try :
            laser_embed_file(src, dst, lang)
            
            # Sanity check
            laser_len = get_file_length(dst) / 1024
            expected_len = tweets_per_lang[lang]
            print(f"laser_len={laser_len}, expected_len={expected_len}")
            assert laser_len == expected_len
            
        except :
            print(f"ERROR : could not embed file {f}")
            os.remove(dst)
        
        
    def labels_from_embs(emb_file, classifier, chunk_size=10_000, offset=0, count=-1) :
        """Generate labels from embeddings, memory-friendly, offset is item-based"""
        labels = []
        while True :
            embs = load_laser_embs(emb_file, count=chunk_size, offset=offset)
            
            if len(labels) + chunk_size >= count :
                return np.append(labels, classifier.predict(embs))[:count] if len(embs) != 0 else np.array(labels)[:count]
            else :
                labels += list(classifier.predict(embs))
                offset += chunk_size        
        
    
    # Offset where to grab embeddings
    all_langs = [x.split('.')[0] for x in os.listdir(lang_text_path)]
    offsets = {x : 0 for x in all_langs}
    
    # For each file, compute sentiments and populate df
    #print("Predicting sentiment using Laser embeddings")
    for f in tqdm(files, desc='sentiment-pred') :
        df = pd.read_parquet(f)
        df['sentiment'] = -1
        languages = df.lang.unique()
        
        for l in languages :
            
            lang_embed_file = os.path.join(lang_text_path, l+'.raw')
            if not os.path.isfile(lang_embed_file) : continue
            nb_in_lang = (df.lang == l).sum()
            labels = labels_from_embs(lang_embed_file, clf, offset=offsets[l], count=nb_in_lang)
            df.loc[df.lang == l, 'sentiment'] = labels
            offsets[l] += nb_in_lang
        
        # Dump new df with sentiments
        df.to_parquet(f)
        
    return


def topic_trends(tweets_piped_path, topics, country_code=None) :
    
    def update_occs(occs, values) :
        c = Counter()
        c.update([h for x in values for h in x.split(',')])
        
        for item in c.items() :
            hashtag, count = item
            to_prepend = day_nb - len(occs[hashtag])
            occs[hashtag] += [0]*to_prepend + [count]
    
    files = [os.path.join(tweets_piped_path,f) for f in  sorted(os.listdir(tweets_piped_path))]
    
    occs     = defaultdict(lambda : [])
    pos_occs = defaultdict(lambda : [])
    
    # Build occurences per day
    for day_nb, f in tqdm(enumerate(files), total=len(files), desc='occs') :
        df = pd.read_parquet(f, columns=['hashtags', 'sentiment'])        
        update_occs(occs, df.hashtags.values)
        update_occs(pos_occs, df[df.sentiment == 1].hashtags.values)
            
    # Append missing zeros at the end
    nb_days = len(files)
    for k in occs :
        occs[k] += [0]*(nb_days-len(occs[k]))
        occs[k] = np.array(occs[k])
        
        if k in pos_occs : 
            pos_occs[k] += [0]*(nb_days-len(pos_occs[k]))
            pos_occs[k] = np.array(pos_occs[k])
        
            
    # Derive neg_occs
    neg_occs = {tag : occs[tag] - pos_occs[tag] if tag in pos_occs else occs[tag] for tag in occs}
        
    # Compute trends    
    sub_topics, higher_topics = topics    
    sub_trends, higher_trends = {}, {}
    
    # Build trends for sub-topics TODO check correct axis
    for idx, hashtags in tqdm(sub_topics.items(), desc='sub_topics') :
        sub_trends[str(idx)]       = np.sum([occs[h]     for h in hashtags], axis=0)
        sub_trends['Pos-'+str(idx)] = np.sum([pos_occs[h] for h in hashtags], axis=0)
        sub_trends['Neg-'+str(idx)] = np.sum([neg_occs[h] for h in hashtags], axis=0)

        
    for idx, subs in higher_topics.items() :
        subs = [str(x) for x in subs]
        higher_trends[str(idx)]       = np.sum([sub_trends[x]       for x in subs], axis=0)
        higher_trends['Pos-'+str(idx)] = np.sum([sub_trends['Pos-'+x] for x in subs], axis=0)
        higher_trends['Neg-'+str(idx)] = np.sum([sub_trends['Pos-'+x] for x in subs], axis=0)

    return sub_trends, higher_trends