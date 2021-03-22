#Imprting the necessary libraries
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import string
from nltk.tokenize.toktok import ToktokTokenizer
import unicodedata
from textblob import TextBlob
from bs4 import BeautifulSoup

!pip install emoji
import emoji




#Reading the file
#Take the location to read from as input in the final version
data_tweet=pd.read_excel('/content/dev_data_tweet.xlsx')

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


def remove_handle_names(text):
  sent=''
  for word in text.split():
    if(word[0]=='@' or word=='QT' or word=='RT'):
      continue
    else:
      sent+=' '+word

  return(sent.strip())  

def remove_https(text):
  sent=''
  for word in text.split():
    if(word[:5]=='https'):
      continue
    elif(word[:3]=='www'):
      continue  
    elif(word[-4:]=='.com'):
      continue   
    else:
      sent+=' '+word
  return(sent.strip())  

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text    


def remove_punct(texts):
  punkts='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  lis=[]
  for char in punkts:
    lis.append(char)
  no_punct=''
  for c in texts:
    if(c not in lis):
      no_punct+=c
  
  return no_punct.strip()

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def demojize(text):
  tweet = emoji.demojize(text)
  tweet = tweet.replace(":"," ")
  tweet = ' '.join(tweet.split())
  return(tweet)


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

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

                           "you're": "you are", "you've": "you have"}


def expand_contractions(text, contraction_mapping=contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text  

   

#def remove_accented_chars(text):
#    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#    return text          

data_tweet['Tweet']=data_tweet['Tweet'].apply(lambda x:remove_handle_names(x))
data_tweet['Tweet']=data_tweet['Tweet'].apply(lambda x:remove_https(x))
data_tweet['Tweet']=data_tweet['Tweet'].apply(lambda x:strip_html_tags(x))
data_tweet['Tweet']=data_tweet['Tweet'].apply(lambda x:expand_contractions(x))
data_tweet['Tweet']=data_tweet['Tweet'].apply(lambda x:remove_punct(x))
data_tweet['Tweet']=data_tweet['Tweet'].apply(lambda x:demojize(x))
data_tweet['Tweet'] = data_tweet['Tweet'].apply(lambda x : re.sub('_', '', x))
#data_tweet['Tweet']=data_tweet['Tweet'].apply(lambda x:remove_stopwords(x))

#Change the location to place where you want to save the Cleaned file
data_tweet.to_csv('tweets_clean.csv', index = False)
