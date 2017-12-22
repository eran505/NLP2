import pandas as pd
import numpy as np
import datetime,re,nltk,os
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
        self.colo = ['id','account','tw_text','full_date','device']
        self.df =None
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
        reltiv_path = os.getcwd()
        df.to_csv(reltiv_path+"/res.csv")

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

        list_col = ['num_time','num_mark_!','link','num_hash_tag']
        for col in list_col:
            self.make_binary_col(df,col)

        df['tokens'] = df['tw_text'].apply(lambda x: nltk.word_tokenize(x))
    def make_binary_col(self,df,name_col):
        df[name_col+'_B'] = np.where(df[name_col] > 0, 1, 0)
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



if __name__ == "__main__":
    TT = Trump_Tweets('tweets.tsv')
