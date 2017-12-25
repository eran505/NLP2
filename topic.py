from sklearn import metrics

import pre_process as pp
import numpy as np
from time import time
import numpy as np
import plotly.graph_objs as go
from plotly import tools


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
def str_to_int(myString):
    if len(myString) < 1 :
        return -1
    #pos = myString.find('.')
    #new_string = myString[pos-1:pos+2]
    int_num = int(float(myString))
    return int_num
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#paramters
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
n_samples = 50000              # 0.20 ~ 50k
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

def read_corpus():
    print("Loading data-set...")
    t0 = time()
    #########reading data-set using csv ########################
    d = []
    with open('/home/ise/Downloads/proc_17_108_unique_comments_text_dupe_count.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        c = 0
        for row in reader:
            d.append({'id': row[0], 'text_data': row[1], 'dupe_count': row[2]})
    df = pd.DataFrame(d, columns=['id', 'text_data', 'dupe_count'])
    df = df.iloc[2:]
    print "list=", list(df)
    print "#" * 100
    print df.shape
    df['dupe_count_int'] = df['dupe_count'].apply(lambda x: str_to_int(x))
    good_bye_list = ['dupe_count']
    df.drop(good_bye_list, axis=1, inplace=True)
    df = df.loc[df['dupe_count_int'] >= 0]
    print df.shape
    print ("done in %0.3fs." % (time() - t0))
    print("data cleaning...")
    t0 = time()
    df = df.iloc[:n_samples]
    df['text_data'] = df['text_data'].apply(lambda x: pp.patent_to_words(x))
    print ("done in %0.3fs." % (time() - t0))
    return df['text_data']

def read_csv_pandas():
    path = "/home/ise/NLP/data_sets/"
    file_stem="res_stem.csv"
    full_path = path+file_stem
    t0 = time()
    df= pd.read_csv(full_path,sep=',',names=['id', 'text_data', 'dupe_count'],error_bad_lines=False)
    df = df.iloc[1:]
    df = df.drop(['dupe_count'], axis=1)
    df = df[df['text_data'].apply(lambda x: isinstance(x, basestring))]
    df = df.reset_index(drop=True)
    print ("done in %0.3fs." % (time() - t0))
    df =  df.iloc[:n_samples]
    return df['text_data']

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
    lda = LatentDirichletAllocation(n_components=tpoic_arg, max_iter=10,
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
        perplextiy = lda.perplexity(tf)
    return perplextiy

def eval_topic(doc_topics):
    for n in range(doc_topics.shape[0]):
        topic_most_pr = doc_topics[n].argmax()
        print ("doc:{} topic:{} \n ".format(n, topic_most_pr))

def cross_validation(func_read,num_fold=2, topic_arr=None,no_idf=True):
    if topic_arr is None:
        topic_arr = [10,7,10]
    df = read_csv_pandas()
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
            if no_idf:
                tf, tf_vectorized = tf_vector(x_Train)
            else:
                tf, tf_vectorized = tf_idf_vector(x_Train)
            lda = LDA_model(tf, tf_vectorized,num_topic)
            train_socre = eval(lda,tf)
            print "train: ",train_socre
            d[num_topic]['train'].append(train_socre)
            data_matrix=lda.fit_transform(tf)
            #cluster_analysis_DBSCAN(data_matrix)
            cluster_analysis_kmeans(data_matrix)
            exit()
            if no_idf:
                tf_test, tf_vectorized_test = tf_vector(x_Test)
            else:
                tf_test, tf_vectorized_test = tf_idf_vector(x_Test)
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



def cluster_analysis_kmeans(X):
    range_n_clusters=[2,3,4,5,6,7,8,9,10]
    for n_clusters in range_n_clusters:

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)



def cluster_analysis_DBSCAN(X):
    n_eps=[0.3,0.31,0.33]
    min_sample=[3,4,5]
    for ep in n_eps:
        db = DBSCAN(eps=ep, min_samples=5).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        silhouette_avg = silhouette_score(X, labels)
        print("n_clusters={} eps={} min_sample={}".format(n_clusters_,ep,5),
              "The average silhouette_score is :", silhouette_avg)




def init_model():
    #read_csv_pandas()
    #read_data()
    #cross_validation(read_data)
    cross_validation(read_csv_pandas)





if __name__ == "__main__":
    init_model()
    print "---- topic model ------"

    #init_model()