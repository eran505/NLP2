import string
from sklearn import feature_extraction, pipeline

import pandas as pd
import numpy as np
import datetime,re,nltk,os
import matplotlib.pyplot as plt
from numpy.core.defchararray import index

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
import time
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import TextBlob as tb
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# April 2017  = Trump switched to a secured phone
#contains quotes (trump)
#contais image or url(not trump)
#contains hash_tag
#time of day
#day of week
#sentiment score
# tf-idf
#convert all time
# POS

def olpcount(p,string):
    pattren = p
    string = string.lower()
    l = len(pattren)
    ct = 0
    for c in range(0, len(string)):
        if string[c:c + l] == pattren:
            ct += 1
    return ct
def remove_record(rec):
    tmp = str(rec).split()
    tmp_res = " ".join(tmp)
    res = re.findall(r'\d{18}', tmp_res)
    if len(res)>1:
        print "rec=",rec
        return 0
    else:
        return 1
def regex_finder(data_in):
    res = re.findall(r'"([^"]*)"', data_in)
    res_filter = [x for x in res if len(str(x).split())>1]
    if len(res_filter)>0:
        return 1
    else:
        return 0
def regex_time_fun(data_text): #TIMME or TIMMEE
    new_txt = re.sub(r'\d{1,2}:\d{2}' , 'TIMME' , data_text)
    new_txt = re.sub(r'(\d{1,2}h)|(\d{1,2}:\d{2})|(\d{1,2}\s*(?:am|pm|AM|PM|A.M.|P.M.|A.m,|P.m.))','TIMME',new_txt)
    new_txt = re.sub(r'(\d{1,2}:\d{2})\s*(am|pm|AM|PM|A.M.|P.M.|A.m|P.m.)','TIMME', new_txt)
    new_txt = re.sub(r'(A.M.|P.M.|A.m|P.m.)',"",new_txt)

    return new_txt
def get_size(x):
    return len(str(x))
def find_url(text_input):
    #ans = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    result_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URLi', text_input)
    return str(result_text)



class Trump_Tweets:

    def __init__(self, dataPath):
        self.data_path = dataPath
        self.stopWords = set(stopwords.words("english"))
        self.colo = ['id','account','tw_text','full_date','device']
        self.df =None
        self.Lemmaatizer= nltk.WordNetLemmatizer()
        self.stem= nltk.PorterStemmer()
        self.pre_processing()
        self.data_preparation()
    def pre_processing(self):
        print "pre-processing...."

        #reading the file into a data-frame when the first column is the index
        df = pd.read_csv('tweets.tsv',sep='\t',names=self.colo,quoting=3 )
        #df['remove'] = df['tw_text'].apply(lambda x: remove_record(x))
        #print "before:",df.shape
        #df.drop(df[df['remove'] == 0 ].index, inplace=True)
        #print "after:",df.shape
        #df=df.dropna()
        #print df.shape
        #fill NA Values with mean
        #print df.loc[df['code'] == -1  ][['id','device','code']]
        #df['device'].fillna('android', inplace=True)
        #df['account'].fillna('realDonaldTrump', inplace=True)
        #df['time'].fillna(df['time'].mean, inplace=True)
        print list(df)
        df['tw_text'] = df['tw_text'].astype(str)
        df['full_date'] = pd.to_datetime(df['full_date'])  # dt.date / dt.time
        df['day'] =  df['full_date'].dt.weekday
        df['time']  = df['full_date'].dt.time
        df['date'] = df['full_date'].dt.date
        df['text_length'] = df['tw_text'].apply(get_size)

        #link in the tweet
        #df['link'] = np.where(str(df['text']).__contains__('http') ,1, 0)
        #count number of hash-tags

        df['device'] = np.where(df['device'].str.contains("instagram"),"INSTA",df['device'])
        df['device'] = np.where(df['device'].str.contains("twitter"), "WEB", df['device'])    # Twitter WeB/iPad
        df['device'] = np.where(df['device'].str.contains("<a"), "OTHER", df['device'])       #TV device
        #df['device'] = np.where(df['device'].str.contains("TweetDeck"), "Press", df['device'])

        df['code'] = df['device'].astype('category').cat.codes
        #

        self.text_features(df)

        #df.sort_values('time', ascending=True,inplace=True)
        self.time_feature(df)
        #new_df = df.loc[df['account'] == "realDonaldTrump"]
        #new_df = df.loc[df['full_date'] > datetime.date(2017,4,1)]
        #print df[['id','tokens']][:10]
        #print new_df.shape
        #print df.shape
        self.df = df
        self.sentient_analysis()
        #print list(df)



    def text_features(self,df):
        print "texter..."
        #print df['text'][:2]
        #df['text'] = df['text'].apply(lambda x: print_me(x) )

        df['num_hash_tag'] = df['tw_text'].apply(lambda x: olpcount('#',x))
        #df['link'] = df['tw_text'].apply(lambda x: olpcount('http',x))
        df['tw_text'] = df['tw_text'].apply(lambda x: find_url(x))
        df['link'] = df['tw_text'].str.count('URLi')
        df['quotes'] = df['tw_text'].apply(lambda x: regex_finder(x))
        df['device'] = df['device'].astype(str)
        df['tw_text'] = df['tw_text'].apply(regex_time_fun)
        df['num_ref'] = df['tw_text'].str.count('@')
        df['num_mark_!'] = df['tw_text'].str.count('!')
        df['num_time'] = df['tw_text'].str.count('TIMME')
        df['num_time'] += df['tw_text'].str.count('TIMMEE')
        self.df = df


    def make_binary_col(self,df,name_col):
        df[name_col+'_B'] = np.where(df[name_col] > 0, 1, 0)

    def norm_column(self,df,arr_name):
        result = df.copy()
        for feature_name in arr_name:
            max_val = df[feature_name].max()
            min_val = df[feature_name].min()
            result[feature_name]  = (df[feature_name]-min_val) / (max_val-min_val)
        return result


    def time_feature(self,df):
        df['night'] = np.where( (df['full_date'].dt.hour  > 18.0 ) | (df['full_date'].dt.hour  < 6.0 )  , 1 , 0)
        df['work_time'] = np.where((df['full_date'].dt.hour >= 6.0) & (df['full_date'].dt.hour < 18.0), 1, 0)

    def __str__(self):
        return "Hello %(name)s!" % self

    def sent_algo_polarity(self,txt):
        wiki = TextBlob(txt)
        return wiki.sentiment[0]

    def sent_algo_subjectivity(self,txt):
        wiki = TextBlob(txt)
        return wiki.sentiment[1]

    def Noun_noun_phrases(self,txt):
        print txt
        wiki = TextBlob(txt)
        res = wiki.noun_phrases
        print res
        return res

    def sentient_analysis(self):
        df =self.df
        print "start_sentient_analysis...."
        self.stopWords.add('atuser')
        self.stopWords.add('urli')
        df["clean_tw"]=df['tw_text'].copy(deep=True)
        df["clean_tw"] = df["clean_tw"].apply(self.processTweet2)
        df['polarity'] = df["clean_tw"].apply(self.sent_algo_polarity)
        df['subjectivity'] = df["clean_tw"].apply(self.sent_algo_subjectivity)
        #df['Nouns_Phrases'] = df["tw_text"].apply(self.Noun_noun_phrases)
        self.df = df
        reltiv_path = os.getcwd()
        #df[["clean_tw","tw_text"]].to_csv(reltiv_path+"/res.csv",sep=';')

    def processTweet2(self,tweet):
        #print "old:",tweet
        # process the tweets
        # Convert to lower case
        tweet = tweet.lower()
        # Convert www.* or https?://* to URL
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' urli ', tweet)
        # Convert @username to AT_USER
        tweet = re.sub(r"@[^\s]+[\s]?", ' atuser ', tweet)
        # Remove additional white spaces
        tweet = re.sub('[\s]+', ' ', tweet)
        #remove &amp char in html &
        tweet = re.sub(r'&amp;', '', tweet)
        # Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        #remvoe numbers
        tweet = re.sub(r"\s?[0-9]+\.?[0-9]*","",tweet)
        # from a string

        regex = re.compile('[%s]' % re.escape((string.punctuation)))
        tweet = regex.sub('',tweet)
        tweet = tweet.strip('\'"?,.')

        res = []
        for w in nltk.word_tokenize(tweet):
            if w in self.stopWords:
                continue
            w = self.stem.stem(w)
            w = self.Lemmaatizer.lemmatize(w)
            res.append(w)
        #print "new: "," ".join(res)
        return " ".join(res)

    def Vector_text_data(self,df_frame,df_test):
        print "vectorize"
        tf = TfidfVectorizer( analyzer='word', ngram_range=(1,2),
                             min_df=0.05)
        df_iphone = df_frame.loc[df_frame['device']=='iphone']
        df_andriod= df_frame.loc[df_frame['device'] == 'android']

        tfidf_matrix_iphone  = tf.fit_transform(df_iphone['clean_tw']).toarray()
        df_train_text_iphone = pd.DataFrame(tfidf_matrix_iphone, columns=[str(x) for x in tf.get_feature_names()], index=df_iphone.index)
        list_iphone = [str(x) for x in tf.get_feature_names()]
        tf = TfidfVectorizer( analyzer='word', ngram_range=(1,2),
                             min_df=0.03)
        tfidf_matrix_android = tf.fit_transform(df_andriod['clean_tw']).toarray()
        df_train_text_android  = pd.DataFrame(tfidf_matrix_android, columns=[str(x) for x in tf.get_feature_names()], index=df_andriod.index)
        list_android = [str(x) for x in tf.get_feature_names()]

        diff = set(list_iphone) - set(list_android)
        diff2 = set(list_android) - set(list_iphone)
        diff_list = list(diff) + list(diff2)
        print diff_list

        cols_iphone = [col for col in df_train_text_iphone.columns if col in diff_list ]
        print df_train_text_iphone.shape

        df_train_text_iphone = df_train_text_iphone[cols_iphone ]
        print df_train_text_iphone.shape
        cols_iphone = [col for col in df_train_text_android.columns if col in diff_list]
        print df_train_text_android.shape
        df_train_text_android = df_train_text_android[cols_iphone]
        print df_train_text_android.shape


        ######################################################################
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                             min_df=0.03)
        tfidf_matrix = tf.fit_transform(df_frame['clean_tw']).toarray()
        df_train_text= pd.DataFrame(tfidf_matrix, columns=[str(x) for x in tf.get_feature_names()],index=df_frame.index)
        feature_names = tf.get_feature_names()
        df_test_x = df_test.drop[]
        ######################################################################

        ch2 = SelectKBest(chi2, k=10)
        X_train = ch2.fit_transform(df_train_text, df_frame['target'])
        X_test = ch2.transform(df_test)
        print feature_names[ch2.get_support(indices=True)]
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        print feature_names

        df_train_text_iphone.to_csv("~/NLP/NLP2/tf_iphone.csv")
        df_train_text_android.to_csv("~/NLP/NLP2/tf_android.csv")
        exit()


        #res_df= df_frame.merge(df_train_text, right_index=True, left_index=True)

        good_bye_list=['clean_tw']
        res_df.drop(good_bye_list, axis=1, inplace=True)

        return res_df
    def data_preparation(self):
        target_device = ['iphone','android'] # WEB | OTHER
        new_df= self.df.copy(deep=True)
        new_df = new_df.loc[new_df['account'] == "realDonaldTrump"]
        new_df = new_df.loc[new_df['full_date'] < datetime.date(2017, 4, 1)]
        new_df = new_df.loc[new_df['device'].isin(target_device)]
        new_df['target'] = np.where(new_df['device'] == 'iphone',-1,1)
        new_df.reset_index(drop=True)
        new_df = new_df.drop([ 'account',  'time', 'date', 'id','code','full_date','tw_text'], axis=1)
        self.fit(new_df)

    def variance_threshold_select(self,df_train,df1_test, thresh=0.0099):
        df1 = df_train.copy(deep=True)  # Make a deep copy of the dataframe
        selector = VarianceThreshold(thresh)
        #print  df1[pd.isnull(df1).any(axis=1)]
        selector.fit(df1.values)  # Fill NA values as VarianceThreshold cannot deal with those
        print selector.get_support(indices=False)
        df2 = df_train.loc[:,selector.get_support(indices=False)]  # Get new dataframe with columns deleted that have NA values
        df2_test = df1_test.loc[:,selector.get_support(indices=False)]
        return df2,df2_test

    def fit(self,df):
        list_metric = []
        y=df['target']
        X=df.loc[:, df.columns != 'target']
        #---normalize the data
        to_norm_list = ['text_length', 'num_hash_tag', 'link', 'quotes', 'num_ref'
            , 'num_mark_!', 'num_time']
        X=self.norm_column(X,to_norm_list)

        sf = StratifiedKFold(n_splits=10, shuffle=True)
        sf.get_n_splits(X, y)

        #print "*" * 55
        #print "mutual_info = ",mutual_info_classif(X, y)
        #print list(X)
        # "*"*55

        for index, (train_indices, val_test_indices) in enumerate(sf.split(X, y)):
            # Generate batches from indices
            x_Train, x_Test = X.iloc[train_indices], X.iloc[val_test_indices]
            y_Train, y_Test = y.iloc[train_indices], y.iloc[val_test_indices]


            #x_Train['target']=y_Train
            #x_Train=x_Train[:5]
            #x_Train = self.Vector_text_data(x_Train,x_Test)
            #x_Test = self.Vector_text_data(x_Test)
            print "-----"*10
            print "x_Test",x_Test.shape
            print "x_Train", x_Train.shape
            print "-----" * 10

            x_Train_fs, x_Test_fs = self.variance_threshold_select(x_Train,x_Test)

            print list(x_Train_fs)

            clf_svm_lin = SVC(kernel='linear')
            t_start = time.clock()
            clf_svm_lin.fit(x_Train_fs, y_Train)
            t = time.clock() - t_start
            y_pred = clf_svm_lin.predict(x_Test_fs)



            #add all method evaluation
            accuracy =metrics.accuracy_score(y_true=y_Test, y_pred=y_pred)
            fpr, tpr, thresholds = metrics.roc_curve(y_true=y_Test, y_score=y_pred, drop_intermediate=True)
            auc= np.trapz(tpr, fpr)
            recall=  metrics.recall_score(y_true=y_Test, y_pred=y_pred)
            precision = metrics.precision_score(y_true=y_Test , y_pred=y_pred)
            f1 = metrics.f1_score(y_true=y_Test , y_pred=y_pred)
            list_metric.append({'AUC':auc,'Accuracy':accuracy,'RECALL':recall,
                                'Precision':precision,'F1_score':f1,'Time':t})


        measure_df = pd.DataFrame(list_metric)
        list_param  = list(measure_df)
        for col in list_param:
            print col,"=",measure_df[col].mean()




if __name__ == "__main__":
    TT = Trump_Tweets('tweets.tsv')
