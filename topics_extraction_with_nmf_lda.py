"""
=======================================================================================
Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
=======================================================================================
"""

import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import pre_process as pp
import csv
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")


def str_to_int(myString):
    if len(myString) < 1 :
        return -1
    #pos = myString.find('.')
    #new_string = myString[pos-1:pos+2]
    int_num = int(float(myString))
    return int_num

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
#########reading data-set using csv ########################
d=[]
with open('/home/ise/Downloads/proc_17_108_unique_comments_text_dupe_count.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    c=0
    for row in reader:
        d.append({'id':row[0],'text_data':row[1],'dupe_count':row[2] })

df= pd.DataFrame(d,columns=['id','text_data','dupe_count'])
df = df.iloc[2:]
print "list=",list(df)
print df.shape
print "#"*100
df['dupe_count_int']=df['dupe_count'].apply(lambda x: str_to_int(x))
good_bye_list = ['dupe_count']
df.drop(good_bye_list, axis=1, inplace=True)
df= df.loc[df['dupe_count_int'] >= 0]
print df.shape
print ( "done in %0.3fs." % (time() - t0) )


##############################################################
"""
==========================================================
Text Pre-processing with NLTK
==========================================================
"""


print("data cleaning...")
t0 = time()
df['text_data']=df['text_data'].apply(lambda x: pp.patent_to_words(x))
print ( "done in %0.3fs." % (time() - t0) )
reltiv_path = "/home/ise/NLP/data_sets"
df.to_csv(reltiv_path+"/res.csv",sep=',')
exit()
###############################################################




dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))



print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
