

from nltk.corpus import stopwords  # Import the stop word list
import nltk
import string
import re


# http://theforgetfulcoder.blogspot.com/2012/06/stemming-or-lemmatization-words.html
# map POS labels to labels that can be read by the lemmatizer

wordnet_tag = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r', 'VBN': 'v', 'VBD': 'v','FW':'n',
               'VBG': 'v', 'VBZ': 'v', 'NNS': 'n', 'VBP': 'v', 'CD': 'n', 'IN': 'n', 'MD': 'n',
               'JJR': 'a', 'JJS': 'a', 'DT': 'n', 'RBR': 'r', 'PRP': 'n', 'CC': 'n', 'WRB': 'n',
               'PRP$': 'n', 'RP': 'r', 'WP$': 'n', 'PDT': 'n', 'WDT': 'n', 'WP': 'n', 'LS': 'n'
               }

###################################################
Lemmaatizer = nltk.WordNetLemmatizer()
stem = nltk.PorterStemmer()
cachedStopWords = stopwords.words("english")
####################################################


def lemmatize_words_array(words_array):
    # Lemmatizer and POS tagger to fit each word based on its POS
    # require wordnet_tag
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tagged = nltk.pos_tag(words_array)
    lemmatized_words_array = []
    for word in tagged:
        if word[1] in wordnet_tag:
            lemma = lemmatizer.lemmatize(word[0], wordnet_tag[word[1]])
        else:
            lemma = lemmatizer.lemmatize( word[0] )
        lemmatized_words_array.append(lemma)
    return lemmatized_words_array



def stem_words_array(words_array):
    stemmer = nltk.PorterStemmer()
    stemmed_words_array = []
    for word in words_array:
        stem = stemmer.stem(word)
        stemmed_words_array.append(stem)
    return stemmed_words_array


def patent_to_words(raw):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    #
    # 1. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    xx = stopwords.words("english")
    # Add first, second and one

    xx.extend(["first", "second", "one", "two", "also", "may", "least", "present", "determine",
               "included", "includes", "include", "provided", "provides", "wherein", "method", "methods",
               "comprises", "comprised", "comprising", "used", "uses", "using", "use", "say", "says", "said",
               "disclose", "discloses", "disclosed",
               "containing", "contain", "contains", "contained", "make", "made", "makes", "end", "couple", "relates"
                                                                                                           "b", "c",
               "d"])
    stops = set(xx)
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # added in
    # 6. Stemming
    #stemming = stem_words_array(meaningful_words)
    #
    # added in
    # 7. Lemmatization
    lemmatization = lemmatize_words_array(meaningful_words)
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join(lemmatization))

def text_clean(data):
    # Remove tickers
    sent_no_tickers=re.sub(r'\$\w*','',data)
    #print('No tickers:')
    #print(sent_no_tickers)
    tw_tknzr= nltk.TweetTokenizer(strip_handles=True, reduce_len=True)
    temp_tw_list = tw_tknzr.tokenize(sent_no_tickers)
    #print('Temp_list:')
    #print(temp_tw_list)
    # Remove stopwords
    list_no_stopwords=[i for i in temp_tw_list if i.lower() not in cachedStopWords]
    #print('No Stopwords:')
    #print(list_no_stopwords)
    # Remove hyperlinks
    list_no_hyperlinks=[re.sub(r'https?:\/\/.*\/\w*','',i) for i in list_no_stopwords]
    #print('No hyperlinks:')
    #print(list_no_hyperlinks)
    # Remove hashtags
    list_no_hashtags=[re.sub(r'#', '', i) for i in list_no_hyperlinks]
    #print('No hashtags:')
    #print(list_no_hashtags)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    list_no_punctuation=[re.sub(r'['+string.punctuation+']+', ' ', i) for i in list_no_hashtags]
    list_no_punctuation = [re.sub(r'&amp;', ' ', i) for i in list_no_punctuation]
    #print('No punctuation:')
    #print(list_no_punctuation)
    # Remove multiple whitespace
    new_sent = ' '.join(list_no_punctuation)
    # Remove any words with 2 or fewer letters
    filtered_list = tw_tknzr.tokenize(new_sent)
    list_filtered = [re.sub(r'^\w\w?$', '', i) for i in filtered_list]
    #print('Clean list of words:')
    #print(list_filtered)
    filtered_sent =' '.join(list_filtered)
    clean_sent=re.sub(r'\s\s+', ' ', filtered_sent)
    #Remove any whitespace at the front of the sentence
    clean_sent=clean_sent.lstrip(' ')
    #print('Clean sentence:')
    #print(clean_sent)
    res = []
    for w in nltk.word_tokenize(clean_sent):
        w = stem.stem(w)
        w = Lemmaatizer.lemmatize(w)
        res.append(w)
    # print "new: "," ".join(res)
    final_sent = " ".join(res)
    #print 'final_sent = ',final_sent
    return final_sent

def clean_tweets_data(tweet):
    # print "old:",tweet
    # process the tweets
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' urli ', tweet)
    # Convert @username to AT_USER
    tweet = re.sub(r"@[^\s]+[\s]?", ' atuser ', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # remove &amp char in html &
    tweet = re.sub(r'&amp;', '', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # remvoe numbers
    tweet = re.sub(r"\s?[0-9]+\.?[0-9]*", "", tweet)
    tweet = re.sub(r'\d{1,2}:\d{2}' , 'TIMME' , tweet)
    tweet = re.sub(r'(\d{1,2}h)|(\d{1,2}:\d{2})|(\d{1,2}\s*(?:am|pm|AM|PM|A.M.|P.M.|A.m,|P.m.))','TIMME',tweet)
    tweet = re.sub(r'(\d{1,2}:\d{2})\s*(am|pm|AM|PM|A.M.|P.M.|A.m|P.m.)','TIMME', tweet)
    tweet = re.sub(r'(A.M.|P.M.|A.m|P.m.)',"",tweet)

    # from a string
    regex = re.compile('[%s]' % re.escape((string.punctuation)))
    tweet = regex.sub('', tweet)
    tweet = tweet.strip('\'"?,.')
    xx = stopwords.words("english")
    # Add first, second and one
    xx.extend(["first", "atuser","urli","TIMME","second", "one", "two", "also", "may", "least", "present", "determine",
               "included", "includes", "include", "provided", "provides", "wherein", "method", "methods",
               "comprises", "comprised", "comprising", "used", "uses", "using", "use", "say", "says", "said",
               "disclose", "discloses", "disclosed",
               "containing", "contain", "contains", "contained", "make", "made", "makes", "end", "couple", "relates"
                                                                                                           "b", "c",
               "d"])
    stops = set(xx)
    words = tweet.split()
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    lemmatization = lemmatize_words_array(meaningful_words)
    # print "new: "," ".join(res)
    return (" ".join(lemmatization))

if __name__ == "__main__":
    print patent_to_words("hi-bal go 4:3 to bal   raw data..")