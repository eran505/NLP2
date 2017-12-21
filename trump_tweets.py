import pandas as pd
import numpy as np
import datetime,re,nltk
import matplotlib.pyplot as plt

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
from numpy.core.defchararray import index


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

def regex_time_fun(data_text):
    new_txt = re.sub(r'\d{1,2}:\d{2}' , 'TIIMEE' , data_text)
    x = re.findall(r'(\d{1,2}h)|(\d{1,2}:\d{2})|(\d{1,2}\s?(?:am|pm|AM|PM|A.M.|P.M.|A.m,|P.m.|Am|Pm))',data_text)
    #if len(x)>0:
    #    print x , "D=",data_text
    return new_txt




def find_url(text_input):
    #ans = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    result_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URLi', text_input)
    return str(result_text)

class Trump_Tweets:

    def __init__(self, dataPath):
        self.data_path = dataPath
        self.colo = ['id','account','text','full_date','device']
        self.df =None
        self.pre_processing()
        self.data_preparation()
    def pre_processing(self):
        print "pre-processing...."

        #reading the file into a data-frame when the first column is the index
        tweets = pd.read_csv('tweets.tsv',
                             sep='\t',
                             names=['id', 'user_handle', 'tweet_text', 'time_stamp', 'device'],
                             # index_col = 0,
                             parse_dates=['time_stamp'],
                             quoting=3
                             )
        tweets['tweet_text'] = tweets['tweet_text'].apply(lambda x: remove_record(x))
        exit()
        df = pd.read_csv(self.data_path, sep='\t',names = self.colo)
        df[['id','account','text','full_date','device']].to_csv('/home/eran/NLP/NLP2/res.csv',index=False,)
        #print df['id'][:3]

        df['remove'] = df['text'].apply(lambda x: remove_record(x))
        df.drop(df[df['remove'] == 0 ].index, inplace=True)
        print df.shape
        df=df.dropna()
        print df.shape
        df[['text','id']].to_csv('/home/eran/NLP/NLP2/res.csv')
        #fill NA Values with mean
        #print df.loc[df['code'] == -1  ][['id','device','code']]
        #df['device'].fillna('android', inplace=True)
        #df['account'].fillna('realDonaldTrump', inplace=True)
        #df['time'].fillna(df['time'].mean, inplace=True)
        #print list(df)

        df['full_date'] = pd.to_datetime(df['full_date'])  # dt.date / dt.time
        #print  df['full_date'].dt.hour
        df['day'] =  df['full_date'].dt.weekday
        df['time']  = df['full_date'].dt.time
        df['date'] = df['full_date'].dt.date
        df['text_length'] = df['text'].apply(len)

        #link in the tweet
        #df['link'] = np.where(str(df['text']).__contains__('http') ,1, 0)
        #count number of hash-tags
        df['num_hash_tag'] = df['text'].apply(lambda x: olpcount('#',x))
        df['link'] = df['text'].apply(lambda x: olpcount('http',x))
        df['link_B'] = np.where(df['link']>0,1,0)
        df['num_hash_tag_B'] = np.where(df['num_hash_tag'] > 0, 1, 0)
        #print df['time'].dt.time
        df['quotes'] = df['text'].apply(lambda x: regex_finder(x))
        df['device'] = df['device'].astype(str)

        df['device'] = np.where(df['device'].str.contains("instagram"),"INSTA",df['device'])
        df['device'] = np.where(df['device'].str.contains("twitter"), "WEB", df['device'])    # Twitter WeB/iPad
        df['device'] = np.where(df['device'].str.contains("<a"), "OTHER", df['device'])       #TV device
        #df['device'] = np.where(df['device'].str.contains("TweetDeck"), "Press", df['device'])

        df['code'] = df['device'].astype('category').cat.codes
        #

        self.text_features(df)

        df.sort_values('time', ascending=True,inplace=True)
        self.time_feature(df)
        #new_df = df.loc[df['account'] == "realDonaldTrump"]
        #new_df = df.loc[df['full_date'] > datetime.date(2017,4,1)]
        #print df[['id','tokens']][:10]
        #print new_df.shape
        #print df.shape
        self.df = df


        #print df[['device','code','id']]

    def text_features(self,df):
        print "texter..."
        #print df['text'][:2]
        #df['text'] = df['text'].apply(lambda x: print_me(x) )
        #
        df['text'] = df['text'].apply(regex_time_fun)
        df['text'] = df['text'].apply(lambda x: find_url(x))
        df['tokens'] = df['text'].apply(lambda x: nltk.word_tokenize(x))

    def data_preparation(self):
        new_df= self.df.copy(deep=True)
        target_device = ['iphone','android'] # WEB | OTHER
        new_df = new_df.loc[new_df['account'] == "realDonaldTrump"]
        new_df = new_df.loc[new_df['full_date'] > datetime.date(2017, 4, 1)]
        new_df = new_df.loc[new_df['account'].isin(target_device)]
        new_df['code'] = new_df['device'].astype('category').cat.codes
        #print new_df[new_df['code','device']]
    def time_feature(self,df):
        df['morning'] = np.where((df['full_date'].dt.hour  >= 6.0) & (df['full_date'].dt.hour  < 12.0) , 1, 0)
        df['night'] = np.where((df['full_date'].dt.hour  >= 0.0) & (df['full_date'].dt.hour  < 6.0) , 1 , 0)

    def __str__(self):
        return "Hello %(name)s!" % self


TT = Trump_Tweets('tweets.tsv')
