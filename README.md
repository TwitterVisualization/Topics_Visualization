# [COVID19 Twitter Topics Visualizations](https://topics.derinbay.com/)

## Socio-epidemiological Insights From a Yearlong COVID-19 Twitter Stream

This repository contains the code necessary to generate our results, presented in paper XXX.

### Installation
This repo is only compatible with Python 3 and not available via pip, you can clone it and process with 
```sh
pip install -r requirements.txt
```

### Results 

To reproduce our results, make sure you download tweets [here (TODO)](<https://github.com/joemccann/dillinger>) and the sentiment classifier [here (TODO)](<https://github.com/joemccann/dillinger>). You can alternatively retrain the classifier, you would then need to download the sentiment140 dataset [here](https://www.kaggle.com/kazanova/sentiment140) and train a classifier on laser embeddings.

Once you have downloaded/trained everything :
1. Edit trends_pipeline.py to adapt the CONFIG part, then run it. We advise to set nb_files=20 at first for a quick sanity check.
2. From that point, Notebooks were used to do exploratory analysis, these are the most important ones :
    * **Showcase.ipynb** is a demonstration of various applications built on our results.
    * **Correlations.ipynb** derives correlations between socio-epidemiologic features and topic trends.
    * **Timeline.ipynb** automatically detects events in topic trends.
    * **Sentiments.ipynb** generates several plots included in the paper and explores polarity indices.
    * **draft_checks.ipynb** creates plots and was used to prototype ideas. 

### Presentation

Main functionalities are gathered in **scripts/hash_utils.py** and used in **scripts/trends_pipeline.py**. the **notebooks** section allows to visualize results and show applications. Below is a detailed explanation of the topic detection pipeline :

1. **Dataframe cleaning and word counts**  
    1. Load tweets from dataframes in _tweets\_path_
    2. Add and populate hashtag column as string with separator, clean tweets text
    3. Save new dataframes in _tweets_piped_path_ with columns (id, lang, country_code, text, hashtags)
    4. Prepare a text file in _texts_path_ with tokenized text concatenated with hashtags, for Word2Vec
    5. Save hashtags occurence counts in _word_counts_path_

2. **Hashtags Embeddings**  
    1. Create corpus from _texts_path_, train Word2Vec on it
    2. Save Word2Vec state in _w2v_path_
    
3. **Topics creation**
    1. Run multiscale_dbscan to find sub-topics (= dict(id &rarr; hashtags) ), save growth in _growth_path_
    2. Compute centroids of current sub-topics
    3. Run multiscale_dbscan on centroids to find higher-topics (= dict(id &rarr; topic_ids))
    4. Save all topics in _topics_path_
    
4. **Sentiment Labelling**
    1. Load files from _tweets_piped_path_, split their content by lang and create corresponding lang files in _lang_text_path_
    2. Compute laser embeddings for each lang file, saved in _lang_text_path_ with .raw extension
    3. Run predictions with classifier saved in _sent_classifier_path_
    4. Save predictions in new column 'sentiment' in _tweets_piped_path_ dataframes
    
5. **Topic trends**
    1. Read _tweets_piped_path_ and create dict(hashtag &rarr; occurences) where occurences are by file, with 1 file per day
    2. Compute occurences for both neutral, positive and negative sentiments
    3. Use results to build trends (= dict(topic &rarr; occurences)) for both sub- and higher topics
    4. Save trends in _raw_trends_path_
