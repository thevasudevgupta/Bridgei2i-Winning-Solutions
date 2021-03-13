import pandas as pd 
import re
import numpy as np 

data = pd.read_csv("dev_article.csv")

#404 error page not found for only date drop
#10 फरवरी, 2021|12:25|IST drop such bs
#split some article into 2 avoid too long articles
#drop duplicates
#keeping nans as it is drop while training

def removebs(x):
    if ('2021|' in x and 'IST' in x) | ('404 Error' in x ):
        return ""
    else:
        return x


data['Text'] = data['Text'].apply(lambda x : re.sub('\n\n', '', x))
data['Text'] = data['Text'].apply(lambda x : re.sub(r'http\S+', '', x))
data['Text'] = data['Text'].apply(removebs)

c = 40001

new_data = pd.DataFrame(columns=['Text','Headline', 'Mobile_Tech'], index =['Text_ID'])


for id,i,headline, tech in zip(data['Text_ID'].values,data['Text'].values,\
     data['Headline'].values, data['Mobile_Tech_Flag'].values):

        if len(str(i)) > 4080:#3rd quartile is 4080 chars in a article
            for j in range(0, int(len(str(i))/4080)):
                txt = i[4080*j:4080*(j+1)]
                new_data.loc['article_{}'.format(c)]= [txt,headline, tech]
                c +=1
            txt2 = i[4080*(j+1):]
            new_data.loc['article_{}'.format(c)]= [txt2,headline, tech]
            c+=1
        else:
            new_data.loc[id] = [i, headline, tech]

new_data.to_csv('article_clean.csv', index = False)



data = pd.read_csv("dev_tweet.csv")

data['Tweet'] = data['Tweet'].apply(lambda x : re.sub('\n\n', '', x))
data['Tweet'] = data['Tweet'].apply(lambda x : re.sub(r'http\S+', '', x))
data['Tweet'] = data['Tweet'].apply(removebs)
data.drop_duplicates(inplace = True )

data.to_csv('tweets_clean.csv', index = False)
