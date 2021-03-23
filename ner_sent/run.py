import nltk
import aspect_based_sentiment_analysis as absa

# exhaustive list of Mobile Brands
BRANDS = [
    'CONDOR',
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
    'MOBIWIRE',
    'WIKO',
    'GIGASET',
    'MEDION',
    'TECHNISAT',
    'TIPTEL',
    'MLS',
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
    'NINETOLOGY',
    'KYOTO',
    'LANIX',
    'ZONDA',
    'FAIRPHONE',
    'PHILIPS',
    'KORYOLINK',
    'QMOBILE',
    'KRUGER&MATZ',
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
    'HTC',
    'AIS',
    'DTAC',
    'WELLCOM',
    'ASELSAN',
    'VESTEL',
    'THURAYA',
    'WILEYFOX',
    'APPLE',
    'BLU',
    'CATERPILLAR',
    'FIREFLY',
    'GOOGLE',
    'HP',
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
    'IPHONE'
]

# This list consists of mobile brands with more than one word in their names
BI_GRAM_FIRST = ['HMD','VK','TEL','T','SONY','I','GARSUS','BENQ','HEWLETT','BULLITT','GIGABYTE','KT','ALLVIEW','M','X','TECHNOLOGY','NINGBO','VOICE','CHERRY','GROUPE','E']
BI_GRAM_SECOND = ['GLOBAL','MOBILE','ME','MOBILE','ERICSSON','MATE','ASUS','SIEMENS','PACKARD','GROUP','TECHNOLOGY','TECH','EVOLIO','DOT','TIGI','HAPPYLIFE','BIRD','MOBILE','MOBILE','BULL','BODA']

def predict_absa(text, absa_nlp):
    def id_mobile_comp(text):
        # store all the entities in a list called Ent_lis
        ent_lis=[]
        words=nltk.word_tokenize(text)
        i=0
        while(i<len(words)):
            # identify if the word is a mobile brand or not
            # check for the possibility of the current word being beggining of a bi-gram or unigram or non-mobile brand word 
            if((words[i].upper() in BRANDS) and (words[i].upper() not in BI_GRAM_FIRST) ):
                ent_lis.append(words[i]) # first collect the Unigrams
            else:
                # check if the first word a bi-gram occurs at the end of a sentence, if so we discard it.
                # If we have not the reached the end of the sentence, we go ahead and check the next word.
                # If the second word occursin bi_gram_sec the we note both of them together as an entity
                if(i+1 != len(words)):
                    if((words[i+1].upper() in BI_GRAM_SECOND) and (words[i].upper() in BI_GRAM_FIRST)):
                        ent_lis.append(words[i]+' '+words[i+1]) #Taking them as a single entity
                        i = i+1 #This counter update helps in skipping the second word from being checked again
                    elif(words[i].upper() in BRANDS):
                        ent_lis.append(words[i]) #If the second word does not belong to bi_gram_sec, we add the first word to the entity list
            i+=1        
        return(ent_lis)     

    def find_aspect_senti(text, aspect, absa_nlp):
        # This is then used to identify the sentiment of aspects(identified entities) in the input sentence
        var = absa_nlp(text, aspects=[aspect])
        if (var.examples[0].sentiment == absa.Sentiment.negative):
            return -1
        elif(var.examples[0].sentiment == absa.Sentiment.positive):
            return 1
        else:
            return 0
    
    #This collects all the entities from the input text 
    entity_lis=id_mobile_comp(text)

    # Used to store the entities and thier corresponding sentiment_score
    senti_dict={}
    for word in entity_lis:
        senti_dict[word]=0 # Intializing all the entities to have a neutral sentiment

    for word in entity_lis:
        senti_dict[word]+=find_aspect_senti(text,word,absa_nlp) # add the sentiment score corresponding to each word from the entity list
        # This summation of sentiment_score helps us account for multiple exisitence of a single entity in a sentence  
        # Total sentiment_score is then used to allocate the final sentiment. 
    sent_dict1={}

    for word in entity_lis:
        if(senti_dict[word]>0):
            sent_dict1[word]='positive'
        elif(senti_dict[word]<0):
            sent_dict1[word]='negative'
        else:
            sent_dict1[word]='neutral'
    return sent_dict1