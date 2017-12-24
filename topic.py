
import pre_process as pp

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

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#paramters
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n_samples = 2000
n_features = 500
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

def LDA_model(matrix,tf_vectorizer):

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(matrix)
    print "perplexity:",lda.perplexity(matrix)
    print "log-likelihood : ",lda.score(matrix)
    print("done in %0.3fs." % (time() - t0))
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    return lda


def cross_validation(num_fold=8,topic_arr=[10,20]):
    df = read_data()
    rows_n = len(df.index)
    print rows_n
    sf = KFold(n_splits=num_fold, shuffle=True)
    X=df
    sf.get_n_splits(X)
    d={}
    for top in topic_arr:
        d[top]=[]
    for index, (train_indices, val_test_indices) in enumerate(sf.split(X)):
        x_Train, x_Test = X.iloc[train_indices], X.iloc[val_test_indices]
        print x_Train.shape
        print x_Test.shape
        for num_topic in topic_arr:
            tf, tf_vectorized = tf_vector(x_Train)
            lda = LDA_model(tf, tf_vectorized)
            list_res = d[num_topic]
            tf, tf_vectorized = tf_vector(x_Test)
            lda.transform(tf)
            x = lda.transform(tf)
            list_res.append()

    print d
def init_model():
    cross_validation()





if __name__ == "__main__":
    init_model()
    print "---- topic model ------"

    #init_model()