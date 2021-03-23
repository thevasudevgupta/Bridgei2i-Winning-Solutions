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
 'E-BODA',
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
 #'IPHONE',
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
'ONEPLUS']

bi_gram_first=['HMD','VK','TEL','T','SONY','I','GARSUS','BENQ','HEWLETT','BULLITT','GIGABYTE','KT','ALLVIEW','M','X','TECHNOLOGY','NINGBO','VOICE','CHERRY','GROUPE']
bi_gram_sec=['GLOBAL','MOBILE','ME','MOBILE','ERICSSON','MATE','ASUS','SIEMENS','PACKARD','GROUP','TECHNOLOGY','TECH','EVOLIO','DOT','TIGI','HAPPYLIFE','BIRD','MOBILE','MOBILE','BULL']

nlp = absa.load()
nlp1=spacy.load('en_core_web_sm')

def predict_asba(text,lis,bi_gram_first,bi_gram_sec):
  nlp = absa.load()
  nlp1=spacy.load('en_core_web_sm')
  def id_mobile_comp(text,lis,nlp,bi_gram_first,bi_gram_sec):
    ent_lis=[]
    words=nltk.word_tokenize(text)
    #words=text.split()
    i=0
    while(i<len(words)):
      #if(word.upper() in lis and word not in ent_lis):
      #print(words[i])
      if((words[i].upper() in lis) and (words[i].upper() not in bi_gram_first) ):
          ent_lis.append(words[i])
      else:
        if(i+1!=len(words)):
            if((words[i+1].upper() in bi_gram_sec) and (words[i].upper() in bi_gram_first)):
              ent_lis.append(words[i]+' '+words[i+1])
              i=i+1
            elif(words[i].upper() in lis_):
              ent_lis.append(words[i])
      i+=1        
    return(ent_lis)     
    
  entity_lis=id_mobile_comp(text,lis,nlp1,bi_gram_first,bi_gram_sec)

  def find_aspect_senti(text,aspect,nlp):
    var=nlp(text,aspects=[aspect])
    if(var.examples[0].sentiment==absa.Sentiment.negative):
      return(-1)
      
    elif(var.examples[0].sentiment==absa.Sentiment.positive):
      return(+1)
      
    else:
      return(0)
        
  senti_dict={}
  for word in entity_lis:
  	senti_dict[word]=0

  for word in entity_lis:
    senti_dict[word]+=find_aspect_senti(text,word,nlp)
  
  sent_dict1={}
  
  for word in entity_lis:
    if(senti_dict[word]>0):
    	sent_dict1[word]='positive'
    elif(senti_dict[word]<0):
    	sent_dict1[word]='negative'
    else:
    	sent_dict1[word]='neutral'
    	
  return(sent_dict1)
