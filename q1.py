
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
from sklearn.model_selection import StratifiedKFold
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
import re
import string
import time


tweets = pd.read_csv('tweets.tsv',
                     sep='\t',
                     names=['id', 'user_handle', 'tweet_text', 'time_stamp', 'device'],
                     # index_col = 0,
                     parse_dates=['time_stamp'],
                     quoting=3
                    )
# remove tweet id
tweets = tweets.drop(['id'], axis=1)
# remove instance that aren't iphone/ android
aoi_tweets = tweets[tweets.device.isin(['android', 'iphone'])]

# before trump switch to iphone
aoi_tweets_trump = aoi_tweets[aoi_tweets['time_stamp'] < '2017-04-01']
# only real donald trump
aoi_tweets_trump = aoi_tweets_trump[aoi_tweets_trump['user_handle'] == 'realDonaldTrump']

dic_replace = {'device': {'android': 1, 'iphone': -1}}
aoi_tweets_trump = aoi_tweets_trump.replace(dic_replace)

stop_words = set(stopwords.words("english"))
lem = WordNetLemmatizer()
stem = PorterStemmer()


def clean_tweet(tweet, stop_words = stop_words, lem = lem, stem = stem):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        tweet = tweet.replace('@', ' ')
        # text_no_links = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)(?:(?:\/[^\s/]))*', '', tweet)
        text_no_links = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet)
        text_no_punc = text_no_links.translate(string.maketrans("", ""), string.punctuation)
        text_no_num = re.sub(r'[^A-Za-z_\']+', ' ', text_no_punc)
        sentenceSubLower = text_no_num.lower()
        words = []
        for word, tag in pos_tag(word_tokenize(sentenceSubLower)):
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            lemma = stem.stem(lem.lemmatize(word.lower(), wntag) if wntag else word)
            words.append(lemma)
        filtered_sentence = [w for w in words if not w in stop_words]
        return " ".join(filtered_sentence)


def get_tweet_sentiment(tweet):
    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''
    # create TextBlob object of passed tweet text
    analysis = TextBlob(tweet)
    # set sentiment
    return analysis.sentiment


def capital_counter(list_text):
    counter = 0
    for word in list_text.split():
        if word.isupper():
            counter += 1
    return counter


# # Features ###

# Features from time
aoi_tweets_trump['hour_at_day'] = aoi_tweets_trump['time_stamp'].dt.hour
aoi_tweets_trump['dayofweek'] = aoi_tweets_trump['time_stamp'].dt.dayofweek
aoi_tweets_trump['text_length'] = aoi_tweets_trump['tweet_text'].apply(lambda x: len(x.split()))
# whether tweets start with quotation marks " or '
aoi_tweets_trump['quoting'] = aoi_tweets_trump['tweet_text'].str.get(0).isin(['"', "'"])
aoi_tweets_trump['link_exist'] = aoi_tweets_trump['tweet_text'].str.contains('https://')
aoi_tweets_trump['#_hashtag'] = aoi_tweets_trump['tweet_text'].str.count("#")
aoi_tweets_trump['#_at'] = aoi_tweets_trump['tweet_text'].str.count("@")
aoi_tweets_trump['#_!'] = aoi_tweets_trump['tweet_text'].str.count("!")
# number of capitalized word
aoi_tweets_trump['tweet_capitale_letter'] = aoi_tweets_trump['tweet_text'].apply(lambda x: float(capital_counter(x))/ len(x.split()))
aoi_tweets_trump['clean_tweet'] = aoi_tweets_trump['tweet_text'].apply(lambda x: clean_tweet(x))
# sensitivity analysis
aoi_tweets_trump['sent_polarity'] = aoi_tweets_trump['clean_tweet'].apply(lambda x: get_tweet_sentiment(x)[0])
aoi_tweets_trump['sent_subjectivity'] = aoi_tweets_trump['clean_tweet'].apply(lambda x: get_tweet_sentiment(x)[1])

# reset id
aoi_tweets_trump = aoi_tweets_trump.reset_index(drop=True)

# Split to X and y
X = aoi_tweets_trump.drop(['device', 'user_handle', 'time_stamp'], axis=1)
y = aoi_tweets_trump.device


# # 10 CV # #
folds = 10
skf = StratifiedKFold(n_splits=folds)
skf.get_n_splits(X, y)

auc = []
accuracy = []
recall = []
precision = []
times = []

columns = ['clf',  'AUC', 'accuracy', 'recall', 'precision', 'time']
results = pd.DataFrame(columns=columns)


for train_index, test_index in skf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_text_feat = np.ndarray(shape=(len(X_train), 0))
    test_text_feat = np.ndarray(shape=(len(X_test), 0))

    vectorizer = Vectorizer()
    # fit to the training data, transform the test data
    X_column_feat = vectorizer.fit_transform(X_train['clean_tweet']).toarray()
    test_column_feat = vectorizer.transform(X_test['clean_tweet']).toarray()

    # append to the text features arrays
    text_feat_labels = vectorizer.get_feature_names()
    text_feat_col_name = [str(x) for x in text_feat_labels]

    df_train_text = pd.DataFrame(X_column_feat, columns=text_feat_col_name, index=X_train.index)
    df_test_text = pd.DataFrame(test_column_feat, columns=text_feat_col_name, index=X_test.index)

    X_train = X_train.merge(df_train_text, right_index=True, left_index=True)
    X_test = X_test.merge(df_test_text, right_index=True, left_index=True)

    X_train = X_train.drop(['tweet_text', 'clean_tweet'], axis=1)
    X_test = X_test.drop(['tweet_text', 'clean_tweet'], axis=1)

    # Feature selection
    thresholdValue = 0.99
    thresh = VarianceThreshold(threshold=(thresholdValue * (1 - thresholdValue)))
    thresh.fit_transform(X_train.values)
    # thresh.transform(X_test.values)
    feature_labels_full = X_train.columns
    feature_labels = feature_labels_full[thresh.get_support(indices=True)]
    # print(feature_labels)

    X_train_sel = X_train[feature_labels]
    X_test_sel = X_test[feature_labels]

    clf_svm_lin = SVC(kernel='linear')
    t0 = time.clock()
    clf_svm_lin.fit(X_train_sel, y_train)
    t = time.clock() - t0
    y_pred = clf_svm_lin.predict(X_test_sel)

    # clf_svm_poly = SVC(kernel='poly')
    # t0 = time.clock()
    # clf_svm_poly.fit(X_train_sel, y_train)
    # t = time.clock() - t0
    # y_pred = clf_svm_poly.predict(X_test_sel)

    # clf_svm_sigmo = SVC(kernel='sigmoid')
    # clf_svm_sigmo.fit(X_train_sel, y_train)
    # y_pred = clf_svm_sigmo.predict(X_test_sel)
#
    # clf_lg = LogisticRegression()
    # clf_lg.fit(X_train_sel, y_train)
    # y_pred = clf_lg.predict(X_test_sel)

    accuracy.append(metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    # auc.append(metrics.auc(y_true=y_test, y_score=y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_pred, drop_intermediate=True)
    auc.append(np.trapz(tpr, fpr))
    recall.append(metrics.recall_score(y_true=y_test, y_pred=y_pred))
    precision.append(metrics.precision_score(y_true=y_test, y_pred=y_pred))
    times.append(t)

results.loc[len(results)] = ['clf_svm_lin', np.mean(auc),  np.mean(accuracy), np.mean(recall), np.mean(recall), np.mean(times)]
print(results)
