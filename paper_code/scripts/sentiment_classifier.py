import pandas as pd
import numpy as np
import os
from laser_embedder import laser_embed_texts
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pickle as pkl

# GPU Details
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1" # GPU id

# specify the GPU memory allocation fraction
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

dataset_path = '/mlo-container-scratch/hartley/sentiment140'
train_path = os.path.join(dataset_path, 'train.csv')
embeddings_path  = '/mlo-container-scratch/hartley/sentiment140/laser_embs_two.npy'
sentiment_classifier_path = "/mlo-container-scratch/hartley/sentiment140/sentiment_classifier.pkl"
force_compute = True

# Load files
columns = ["label", "id", "date", "is_query", "user_name", "text"]
sentiments = pd.read_csv(train_path, encoding = "ISO-8859-1", header=None, names=columns)[["label", "text"]]
sentiments.label.replace(4, 1, inplace=True)

print('LOADED SENTIMENTS')
print(sentiments.text)

# Generate embeddings from LASER
if force_compute or not os.path.isfile(embeddings_path) :
    laser_embs = laser_embed_texts(sentiments.text, "en", tmp_dir='/mlo-container-scratch/hartley/')
    with open(embeddings_path, 'wb') as f:
        np.save(f, laser_embs)
else :
    print('EMS already computed, loading them')
    with open(embeddings_path, 'rb') as f:
        laser_embs = np.load(f, allow_pickle=True)
        
print('EMBS')
print(laser_embs.shape)

# Fit Laser
X_train, X_test, y_train, y_test = train_test_split(laser_embs, sentiments.label, test_size=0.25, random_state=42)
clf = LogisticRegression().fit(X_train, y_train)

# Compute accuracy
pred = clf.predict(X_test)
errors = np.abs(pred - y_test)
accuracy = 1 - errors.sum() / len(errors)
print("Logistic Regression Accuracy  on Sentiment140 for LASER is {:.4f}".format(accuracy))
pkl.dump(clf, open(sentiment_classifier_path, "wb"))
confusion_matrix(y_test, pred)