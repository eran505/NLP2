
import pre_process as pp
import numpy as np
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import matplotlib.pyplot as plt
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#paramters
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n_samples = 2000
n_features = 550
n_topics = 10
n_top_words = 20
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def read_data():
    df = pd.read_csv('tweets.tsv', sep='\t', names=['id','account','tw_text','full_date','device'], quoting=3)
    df['tw_text'] = df['tw_text'].astype(str)
    df['clean'] = df['tw_text'].apply(pp.clean_tweets_data)
    return df['clean']


def tf_idf_vector(df):
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english')
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(df)
    print("done in %0.3fs." % (time() - t0))
    return tfidf,tfidf_vectorizer

def tf_vector(df):
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(df)
    print("done in %0.3fs." % (time() - t0))
    return tf,tf_vectorizer

def LDA_model(matrix,tf_vectorizer,tpoic_arg=None):
    if tpoic_arg is None:
        tpoic_arg=n_topics
    lda = LatentDirichletAllocation(n_topics=tpoic_arg, max_iter=10,
                                    learning_method='online',evaluate_every=2,
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(matrix)
    print("done in %0.3fs." % (time() - t0))
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    return lda

def eval(lda,tf,norm=False):
    if norm :
        gamma = lda.transform(tf)
        doc_topic_dist_unnorm = np.matrix(gamma)
        doc_topic_norm = doc_topic_dist_unnorm / doc_topic_dist_unnorm.sum(axis=1)
        perplextiy = lda.perplexity(tf, doc_topic_norm)
    else:
        gamma = lda.transform(tf)
        perplextiy = lda.perplexity(tf,gamma)
    return perplextiy

def eval_topic(doc_topics):
    for n in range(doc_topics.shape[0]):
        topic_most_pr = doc_topics[n].argmax()
        print ("doc:{} topic:{} \n ".format(n, topic_most_pr))

def cross_validation(num_fold=2, topic_arr=None):
    if topic_arr is None:
        topic_arr = [10,12,13,15,16]
    df = read_data()
    rows_n = len(df.index)
    print rows_n
    sf = KFold(n_splits=num_fold, shuffle=True)
    X=df
    sf.get_n_splits(X)
    d={}
    for top in topic_arr:
        d[top]={"train":[],"test":[]}
    for index, (train_indices, val_test_indices) in enumerate(sf.split(X)):
        x_Train, x_Test = X.iloc[train_indices], X.iloc[val_test_indices]
        print x_Train.shape
        print x_Test.shape
        for num_topic in topic_arr:
            tf, tf_vectorized = tf_vector(x_Train)
            lda = LDA_model(tf, tf_vectorized,num_topic)
            train_socre = eval(lda,tf)
            print "train: ",train_socre
            d[num_topic]['train'].append(train_socre)
            tf_test, tf_vectorized_test = tf_vector(x_Test)
            test_socre = eval(lda,tf_test)
            print "test: ",test_socre
            d[num_topic]['test'].append(test_socre)
    list_train=[]
    list_test = []
    list_n_topic = []
    for k in d:
        list_train.append(np.mean(d[k]['train']))
        list_test.append(np.mean(d[k]['test']))
        list_n_topic.append(k)
        print "topic number {}".format(k)
        print "mean_train: ",np.mean(d[k]['train'])
        print "mean_test: " ,np.mean(d[k]['test'])
    plt.plot(list_n_topic,list_test,'r')
    plt.plot(list_n_topic,list_train,'g')
    plt.show()
def init_model():
    cross_validation()





if __name__ == "__main__":
    init_model()
    print "---- topic model ------"

    #init_model()