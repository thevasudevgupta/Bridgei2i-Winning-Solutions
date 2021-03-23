#Add all these to requirements.txt file
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
import nltk
nltk.download('punkt')
import unicodedata


!pip install aspect-based-sentiment-analysis
import aspect_based_sentiment_analysis as absa
from textblob import TextBlob
import spacy

#nlp1=spacy.load('en_core_web_sm')
#nlp = absa.load()
  


#Exhaustive_list of Mobile brands
#This list is used to identify the mobile_brands from the set of named entities which exist in the sentence
#List consist of all mobile brands (existing, defunct and brands which are no longer associated with mobiles)

lis_=['CONDOR',
 'WALTON',
 'GRADIENTE',
 'MULTILASER',
 'POSITIVO',
 'BLACKBERRY',
 'DATAWIND',
 'AMOI',
 'BBK',
 'COOLPAD',
 'CUBOT',
 'GFIVE',
 'HAIER',
 'HISENSE',
 'HONOR',
 'HUAWEI',
 'KONKA',
 'LEECO',
 'MEITU',
 'MEIZU',
 'NINGBO BIRD',
 'ONEPLUS',
 'OPPO',
 'REALME',
 'IQOO',
 'SMARTISAN',
 'TCL',
 #'TECHNOLOGY HAPPYLIFE',
 'TECNO',
 'VIVO',
 'VSUN',
 'WASAM',
 'XIAOMI',
 'ZOPO',
 'ZTE',
 'ZUK',
 'JABLOTRON',
 'VERZO',
 'SICO',
 'JOLLA',
 'NOKIA',
 'HMD',
 'BITTIUM',
 'ARCHOS',
 #'GROUPE BULL',
 'MOBIWIRE',
 'WIKO',
 'GIGASET',
 'MEDION',
 'TECHNISAT',
 'TIPTEL',
 'MLS',
 #'X TIGI',
 'LENOVO',
 'CREO',
 'CELKON',
 'IBALL',
 'INTEX',
 'KARBONN',
 'LAVA',
 'HCL',
 'JIO',
 'LYF',
 'MICROMAX',
 'ONIDA',
 'SPICE',
 'VIDEOCON',
 'XOLO',
 'YU',
 'MPHONE',
 'NEXIAN',
 'MITO',
 'POLYTRON',
 'ADVAN',
 'AXIOO',
 'IMO',
 'ZYREX',
 'ANDROMAX',
 'EVERCOSS',
 'LUNA',
 'GENPRO',
 'ASIAFONE',
 'HIMAX',
 'SPC',
 'VITELL',
 'VENERA',
 'OSMO',
 'HICORE',
 'MAXTRON',
 'BRONDI',
 'NEW_GENERATION',
 'OLIVETTI',
 'ONDA',
 'AKAI',
 'FUJITSU',
 'CASIO',
 'HITACHI',
 'JRC',
 'KYOCERA',
 'MITSUBISHI',
 'NEC',
 'PANASONIC',
 'SANSUI',
 'SHARP',
 'SONY',
 'TOSHIBA',
 'JUST5',
 #'M DOT',
 'NINETOLOGY',
 'KYOTO',
 'LANIX',
 'ZONDA',
 'FAIRPHONE',
 'PHILIPS',
 'KORYOLINK',
 'QMOBILE',
 #'VOICE MOBILE',
 #'CHERRY MOBILE',
 'KRUGER&MATZ',
 #'ALLVIEW EVOLIO',
 #'E BODA',
 'CLOUDFONE',
 'MYPHONE',
 'TORQUE',
 'STARMOBILE',
 'MANTA',
 'MYRIA',
 'UTOK',
 'LG',
 'EVERTEKTUNISIE',
 'GARMIN',
 'BEELINE',
 'EXPLAY',
 'GRESSO',
 'HIGHSCREEN',
 'MEGAFON',
 'MTS',
 'ROVERPC',
 'TEXET',
 'SITRONICS',
 'YOTAPHONE',
 #'KT TECH',
 'PANTECH',
 'SAMSUNG',
 'BQ',
 'DORO',
 'ACER',
 'ASUS',
 'BENQ',
 'DBTEL',
 'DOPOD',
 'FOXCONN',
 #'GIGABYTE TECHNOLOGY',
 'HTC',
 'AIS',
 'DTAC',
 'WELLCOM',
 'ASELSAN',
 'VESTEL',
 'THURAYA',
 #'BULLITT GROUP',
 'WILEYFOX',
 'APPLE',
 'BLU',
 'CATERPILLAR',
 'FIREFLY',
 'GOOGLE',
 'HP',
 #'HEWLETT PACKARD',
 'INFOCUS',
 'MOTOROLA',
 'OBI',
 'NEXTBIT',
 'PURISM',
 'VINSMART',
 'GTEL',
 'GIONEE',
 'AEG',
 'GRUNDIG',
 'ALCATEL',
 'ALLVIEW',
 'AMAZON',
 'AT&T',
 'BENEFON',
 #'BENQ SIEMENS',
 'BIRD',
 'BLACKBERRY',
 'BLACKVIEW',
 'BOSC',
 'CAT',
 'CHEA',
 'DELL',
 'EMPORIA',
 'ENERGIZER',
 'ERICSSON',
 'ETENFAIRPHONE',
 'SIEMENS',
 'GARMIN ASUS',
 'GIGABYTE',
 'I MATE',
 'ICEMOBILE',
 'INFINIX',
 'INNOSTREAM',
 'INQ',
 'KARBONN',
 'MAXON',
 'MAXWEST',
 'MICROSOFT',
 'MITAC',
 'MODU',
 'MOTOROLA',
 'MWG',
 'NEONODE',
 'NIU',
 'NVIDIA',
 'O2',
 'ORANGE',
 'PALM',
 'PARLA',
 'PLUM',
 'POSH',
 'PRESTIGIO',
 'QTEK',
 'RAZER',
 'SAGEM',
 'SENDO',
 'SEWON',
 'SONIM',
 #'SONY ERICSSON',
 'T MOBILE',
 'TEL ME',
 'TELIT',
 'ULEFONE',
 'UNNECTO',
 'VERTU',
 'VERYKOOL',
 'VK MOBILE',
 'VODAFONE',
 'WND',
 'XCUTE',
 'YEZZ',
 'YOTA',
 'INTEL',
 'MI',
 'ITEL',
 #'HMD GLOBAL',
 'REDMI',
 'XI',
 'MOTO',
 'QUALCOMM',
 'RUNGEE',
 'POCO',
 'AIRTEL',
 'AIRCEL',
 'HUTCH',
 'BSNL',
 'UMIDIGI',
 'TOYOTA',
 'XGODY',
'ONEPLUS',
'IPHONE']

#This list consists of mobile brands with more than one word in their names
#Predominantly, mobile brands did not have more than two words
#Bi_gram_first consist of the first half of the two word brands and Bi_gram_second holds the corresponding other half
bi_gram_first=['HMD','VK','TEL','T','SONY','I','GARSUS','BENQ','HEWLETT','BULLITT','GIGABYTE','KT','ALLVIEW','M','X','TECHNOLOGY','NINGBO','VOICE','CHERRY','GROUPE','E']
bi_gram_sec=['GLOBAL','MOBILE','ME','MOBILE','ERICSSON','MATE','ASUS','SIEMENS','PACKARD','GROUP','TECHNOLOGY','TECH','EVOLIO','DOT','TIGI','HAPPYLIFE','BIRD','MOBILE','MOBILE','BULL','BODA']

#Aspect based sentiment analysis Librarie's object is named and nlp
#Language object of spacy is named nlp1
nlp = absa.load()
nlp1=spacy.load('en_core_web_sm') #all the operations are carried out using Spacy is in English


def predict_asba(text,lis,bi_gram_first,bi_gram_sec):
  
  def id_mobile_comp(text,lis,nlp,bi_gram_first,bi_gram_sec):
    #We store all the entities in a list called Ent_lis
    ent_lis=[]
    #We tokenize the input text into corresponding tokens
    words=nltk.word_tokenize(text) #words is a list consisting of individual tokens
    
    i=0
    while(i<len(words)):
      #We check on word at a time
      #We identify if the word is a mobile brand or not
      #We check for the possibility of the current word being beggining of a bi-gram or unigram or non-mobile brand word 
      if((words[i].upper() in lis) and (words[i].upper() not in bi_gram_first) ):
          ent_lis.append(words[i]) #We first collect the Unigrams
          #If it was the first word of a bi-gram we switch to the else loop
      else:
        #We check if the first word a bi-gram occurs at the end of a sentence,if so we discard it.
        #If we have not the reached the end of the sentence, we go ahead and check the next word.
        #If the second word occursin bi_gram_sec the we note both of them together as an entity
        if(i+1!=len(words)):
            if((words[i+1].upper() in bi_gram_sec) and (words[i].upper() in bi_gram_first)):
              ent_lis.append(words[i]+' '+words[i+1]) #Taking them as a single entity
              i=i+1 #This counter update helps in skipping the second word from being checked again
            elif(words[i].upper() in lis_):
              ent_lis.append(words[i]) #If the second word does not belong to bi_gram_sec, we add the first word to the entity list
      i+=1        
    return(ent_lis)     
  
  #This collects all the entities from the input text 
  entity_lis=id_mobile_comp(text,lis,nlp1,bi_gram_first,bi_gram_sec)

  def find_aspect_senti(text,aspect,nlp):
    #We create a object of the asba class
    #This is then used to identify the sentiment of aspects(identified entities) in the input sentence
    var=nlp(text,aspects=[aspect])
    if(var.examples[0].sentiment==absa.Sentiment.negative):
      return(-1) #Negative sentiment
    elif(var.examples[0].sentiment==absa.Sentiment.positive):
      return(+1) #positive sentiment
    else:
      return(0) #Neutral sentiment
        
  #Used to store the entities and thier corresponding sentiment_score
  senti_dict={}
  for word in entity_lis:
  	senti_dict[word]=0 # Intializing all the entities to have a neutral sentiment

  for word in entity_lis:
    senti_dict[word]+=find_aspect_senti(text,word,nlp) # We add the sentiment score corresponding to each word from the entity list
    # This summation of sentiment_score helps us account for multiple exisitence of a single entity in a sentence  
    # Total sentiment_score is then used to allocate the final sentiment. 
  sent_dict1={}
  
  #Using the final sentiment scores of each entity we identify thier corresoonding sentiment.
  #Sign of total sentiment_score is predicted as the entitie's sentiment. 
  for word in entity_lis:
    if(senti_dict[word]>0):
    	sent_dict1[word]='positive'
    elif(senti_dict[word]<0):
    	sent_dict1[word]='negative'
    else:
    	sent_dict1[word]='neutral'
    	
  return(sent_dict1) 
 
#Sample Function call 
#predict_asba('HMD GLOBAL is great',lis_,bi_gram_first,bi_gram_sec)
#Sample output
#{'HMD GLOBAL': 'positive'}
 
