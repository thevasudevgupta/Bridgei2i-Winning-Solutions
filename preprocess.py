import re
import nltk
from nltk.corpus import stopwords
import random
import functools
import string

EXPANSION_DICT = {
    "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
    "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
    "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have",
}

def expand_words(text):
    text = text.replace('’', "'")
    text = text.replace('”', "")
    text = text.replace('‘', "'")
    text = text.replace("“", "")
    text = text.replace("‼", "")
    word_pattern = re.compile('({})'.format('|'.join(EXPANSION_DICT.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(word):
        match = word.group(0)
        first_char = match[0]
        expanded_word = EXPANSION_DICT.get(match) if EXPANSION_DICT.get(match) else EXPANSION_DICT.get(match.lower())                       
        expanded_word = first_char + expanded_word[1:]
        return expanded_word
    expanded_text = word_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# get stop words
try:
    stop_words = stopwords.words('english')
    stop_words.remove('no')
    stop_words.remove('not')
    punc = string.punctuation
    punc = punc.replace(".", "").replace(",", "")
except:
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.remove('no')
    stop_words.remove('not')
    punc = string.punctuation
    punc = punc.replace(".", "").replace(",", "")

def remove_stopwords(text):
    cleaned_word_list = []
    word_list = text.split()
    for word in word_list:
        # if not in stopwords append
        if word not in stop_words:
            cleaned_word_list.append(word)
        # dont remove all stopwords, remove with some chance (helps build some coherence in the sentence)
        elif random.random() > 0.5:
            cleaned_word_list.append(word)
    text = " ".join(cleaned_word_list)
    return text

def basic_clean(text):
    # basic clean and then lowercase
    # remove @
    text = re.sub("@[A-Za-z0-9]+", "", text) 
    # remove http links
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    # remove html tags
    text = re.sub('<[^<]+?>', '', text)
    # remove hastag but keep text. Also remove _ connecting words and seperate them
    text = text.replace("#", "").replace("_", " ")
    word_list = text.split()
    text = " ".join([w for w in word_list if w not in ['QT', 'RT']])
    # remove punctuations except . and ,
    text = text.translate(str.maketrans("", "", punc))
    return text

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f" # dingbats
        u"\u3030"
    "]+", re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

if __name__ == "__main__":
    import os
    import pandas as pd
    DATA_DIR = "data/Development Data/"

    tweet_data = pd.read_csv(os.path.join(DATA_DIR, "dev_data_tweet.csv"))
    print(f"loaded tweets with {len(tweet_data)} entries")
    cleaned_text = []
    clean_tweet_func = compose(expand_words, basic_clean, remove_emojis, remove_stopwords)
    for i in range(len(tweet_data)):
        cleaned_text.append(clean_tweet_func(tweet_data['Tweet'][i]))
    tweet_data['cleaned'] = cleaned_text
    tweet_data = tweet_data.drop_duplicates(subset=['cleaned'], keep='first')
    print(f"cleaned and saving {len(tweet_data)} tweet entries")
    tweet_data.to_csv("clean_tweet.csv", encoding='utf-8', index=False)

    article_data = pd.read_csv(os.path.join(DATA_DIR, "dev_data_article.csv"))
    print(f"loaded articles with {len(article_data)} entries")
    cleaned_text = []
    clean_article_func = compose(expand_words, basic_clean, remove_emojis, remove_stopwords)
    for i in range(len(article_data)):
        cleaned_text.append(clean_article_func(article_data['Text'][i]))
    article_data['cleaned'] = cleaned_text
    article_data = article_data.rename(columns = {'Mobile_Tech_Flag':'Mobile_Tech_Tag'})
    article_data = article_data.drop_duplicates(subset=['cleaned'], keep='first')
    print(f"cleaned and saving {len(article_data)} article entries")
    article_data.to_csv("clean_article.csv", encoding='utf-8', index=False)
