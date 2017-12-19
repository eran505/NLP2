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

def olpcount(p,string):
    pattren = p
    string = string.lower()
    l = len(pattren)
    ct = 0
    for c in range(0, len(string)):
        if string[c:c + l] == pattren:
            ct += 1
    return ct

def regex_finder(input):
    res = re.findall(r'"([^"]*)"', input)
    res_filter = [x for x in res if len(str(x).split())>1]
    if len(res_filter)>0:
        return 1
    else:
        return 0

def regex_time(input_str):
    result = re.findall(r'\d{1,2}:\d{2}',input_str)
    if len(result)>1:
        print input_str,"  res= ",result


def find_url(text):
    #ans = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    result_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URLi', text)
    return str(result_text)

class Trump_Tweets:

    def __init__(self, dataPath):
        self.data_path = dataPath
        self.colo = ['id','account','text','full_date','device']
        self.df =None
        self.pre_processing()

    def pre_processing(self):
        print "pre-processing...."

        #reading the file into a data-frame when the first column is the index
        df = pd.read_csv(self.data_path, sep='\t',names = self.colo)

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
        new_df = df.loc[df['account'] == "realDonaldTrump"]
        new_df = df.loc[df['full_date'] > datetime.date(2017,4,1)]
        #print df[['id','tokens']][:10]
        #print new_df.shape
        #print df.shape


        df[['device','code','id']].to_csv('/home/ise/NLP/NLP2/res.csv')
        #print df[['device','code','id']]

    def text_features(self,df):
        print "texter..."
        print df['text'][:2]
        df['text'] = df['text'].apply(lambda x: regex_time(x))
        exit()
        df['text'] = df['text'].apply(lambda x: find_url(x))
        df['tokens'] = df['text'].apply(lambda x: nltk.word_tokenize(x))



    def time_feature(self,df):
        df['morning'] = np.where((df['full_date'].dt.hour  >= 6.0) & (df['full_date'].dt.hour  < 12.0) , 1, 0)
        df['night'] = np.where((df['full_date'].dt.hour  >= 0.0) & (df['full_date'].dt.hour  < 6.0) , 1 , 0)

    def __str__(self):
        return "Hello %(name)s!" % self


TT = Trump_Tweets('tweets.tsv')
