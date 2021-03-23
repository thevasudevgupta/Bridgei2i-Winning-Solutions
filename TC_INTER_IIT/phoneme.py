from g2p_en import G2p
import pandas as pd
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences

g2p = G2p()

converter={}
converter["AA"]="ɑ"
converter["AE"]="æ"
converter["AH"]="ʌ"
converter["AO"]="ɔ"
converter["AW"]="aʊ"
# converter["AX"]="ə"
# converter["AXR"]="ɚ"
converter["AY"]="aɪ"
converter["EH"]="ɛ"
converter["ER"]="ɝ"
converter["EY"]="eɪ"
converter["IH"]="ɪ"
# converter["IX"]="ɨ"
converter["IY"]="i"
converter["OW"]="oʊ"
converter["OY"]="ɔɪ"
converter["UH"]="ʊ"
converter["UW"]="u"
# converter["UX"]="ʉ"
converter["B"]="b"
converter["CH"]="tʃ"
converter["D"]="d"
converter["DH"]="ð"
# converter["DX"]="ɾ"
# converter["EL"]="l̩"
# converter["EM"]="m̩"
# converter["EN"]="n̩"
converter["F"]="f"
converter["G"]="ɡ"
# converter["H"]="h"
converter["HH"]="h"
converter["JH"]="dʒ"
converter["K"]="k"
converter["L"]="l"
converter["M"]="m"
converter["N"]="n"
# converter["NX"]="ɾ̃"
converter["NG"]="ŋ"
converter["P"]="p"
# converter["Q"]="ʔ"
converter["R"]="ɹ"
converter["S"]="s"
converter["SH"]="ʃ"
converter["T"]="t"
converter["TH"]="θ"
converter["V"]="v"
converter["W"]="w"
# converter["WH"]="ʍ"
converter["Y"]="j"
converter["Z"]="z"
converter["ZH"]="ʒ"
converter[' ']="SIL"

mapping={}
mapping["अ"]="ə"
mapping["आ"]="a:"
mapping["इ"]="ɪ"
mapping["ई"]="i:"
mapping["उ"]="ʊ"
mapping["ऊ"]="u:"
mapping["ऋ"]="ɻ̩"
mapping["ए"]="e:"
mapping["ऐ"]="ɛ:"
mapping["ओ"]="o:"
mapping["औ"]="ɔ:"
#mapping["अँ"]="/ə̃/"
#mapping["अः"]="/əɦə/"
mapping["अं"]="əm"
#mapping["ऑ"]="/ɒ/"


mapping["ा"]="a:"
mapping["ि"]="ɪ"
mapping["ी"]="i:"
mapping["ु"]="ʊ"
mapping["ू"]="u:"
mapping["ृ"]="ɻ̩"
mapping["े"]="e:"
mapping["ै"]="ɛ:"
mapping["ो"]="o:"
mapping["ौ"]="ɔ:"
#mapping["ँ"]="/ə̃/"
#mapping["ː"]="/əɦə/"
mapping['ं']="əm"
mapping["ॆ"]="e:"

mapping["क"+"्"]="k"
mapping["ख"+"्"]="kʰ"
mapping["ग"+"्"]="g"
mapping["घ"+"्"]="gʰ"
mapping["ङ"+"्"]="ŋ"
mapping["च"+"्"]="tʃ"
mapping["छ"+"्"]="tʃʰ"
mapping["ज"+"्"]="dʒ"
mapping["झ"+"्"]="dʒʰ"
mapping["ञ"+"्"]="ɲ"
mapping["ट"+"्"]="ʈ"
mapping["ठ"+"्"]="ʈʰ"
mapping["ड"+"्"]="ɖ"
mapping["ढ"+"्"]="ɖʰ"
mapping["ण"+"्"]="ɳ"
mapping["त"+"्"]="t̪"
mapping["थ"+"्"]="t̪ʰ"
mapping["द"+"्"]="d̪"
mapping["ध"+"्"]="d̪ʰ"
mapping["न"+"्"]="n"
mapping["प"+"्"]="p"
mapping["फ"+"्"]="pʰ"
mapping["ब"+"्"]="b"
mapping["भ"+"्"]="bʰ"
mapping["म"+"्"]="m"
mapping["य"+"्"]="j"
mapping["र"+"्"]="ɾ"
mapping["ल"+"्"]="l"
mapping["व"+"्"]="ʋ"
mapping["स"+"्"]="s"
mapping["श"+"्"]="ʃ"
mapping["ष"+"्"]="ʂ"
mapping["ह"+"्"]="ɦ"
mapping["ळ"+"्"]="ɭ̆ɭ̆"
mapping[" "]="SIL"

# def Merge(dict1, dict2):
#     return(dict2.update(dict1))

# Merge(mapping, converter)
 
converter.update(mapping)


def text_phonemes(text="पढ़ती हैं"):
  vowels=["ा","ि","ी","ु","ू","ृ","े","ै","ो","ौ","ँ",'ं','ॆ',"ː","आ","इ","ई","उ","ऊ","ऋ","ए","ऐ","ओ","औ","अः","अं"]
  full_consonants=["क","ख","ग","घ","ङ","च","छ","ज","झ","ञ","ट","ठ","ड","ढ","ण","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","ळ","व","श","ष","स","ह"]
  special_vowels=["ः","ं","ँ"]
  mapping={}
  mapping["अ"]="ə"
  mapping["आ"]="a:"
  mapping["इ"]="ɪ"
  mapping["ई"]="i:"
  mapping["उ"]="ʊ"
  mapping["ऊ"]="u:"
  mapping["ऋ"]="ɻ̩"
  mapping["ए"]="e:"
  mapping["ऐ"]="ɛ:"
  mapping["ओ"]="o:"
  mapping["औ"]="ɔ:"
  #mapping["अँ"]="/ə̃/"
  #mapping["अः"]="/əɦə/"
  mapping["अं"]="əm"
  #mapping["ऑ"]="/ɒ/"


  mapping["ा"]="a:"
  mapping["ि"]="ɪ"
  mapping["ी"]="i:"
  mapping["ु"]="ʊ"
  mapping["ू"]="u:"
  mapping["ृ"]="ɻ̩"
  mapping["े"]="e:"
  mapping["ै"]="ɛ:"
  mapping["ो"]="o:"
  mapping["ौ"]="ɔ:"
  #mapping["ँ"]="/ə̃/"
  #mapping["ː"]="/əɦə/"
  mapping['ं']="əm"
  mapping["ॆ"]="e:"

  mapping["क"+"्"]="k"
  mapping["ख"+"्"]="kʰ"
  mapping["ग"+"्"]="g"
  mapping["घ"+"्"]="gʰ"
  mapping["ङ"+"्"]="ŋ"
  mapping["च"+"्"]="tʃ"
  mapping["छ"+"्"]="tʃʰ"
  mapping["ज"+"्"]="dʒ"
  mapping["झ"+"्"]="dʒʰ"
  mapping["ञ"+"्"]="ɲ"
  mapping["ट"+"्"]="ʈ"
  mapping["ठ"+"्"]="ʈʰ"
  mapping["ड"+"्"]="ɖ"
  mapping["ढ"+"्"]="ɖʰ"
  mapping["ण"+"्"]="ɳ"
  mapping["त"+"्"]="t̪"
  mapping["थ"+"्"]="t̪ʰ"
  mapping["द"+"्"]="d̪"
  mapping["ध"+"्"]="d̪ʰ"
  mapping["न"+"्"]="n"
  mapping["प"+"्"]="p"
  mapping["फ"+"्"]="pʰ"
  mapping["ब"+"्"]="b"
  mapping["भ"+"्"]="bʰ"
  mapping["म"+"्"]="m"
  mapping["य"+"्"]="j"
  mapping["र"+"्"]="ɾ"
  mapping["ल"+"्"]="l"
  mapping["व"+"्"]="ʋ"
  mapping["स"+"्"]="s"
  mapping["श"+"्"]="ʃ"
  mapping["ष"+"्"]="ʂ"
  mapping["ह"+"्"]="ɦ"
  mapping["ळ"+"्"]="ɭ̆ɭ̆"
  mapping[" "]="SIL"


  #text="नमस्कार नमस्कार नमस्कार नमस्कार नमस्कार नमस्कार नमस्कार नमस्कार "
  line=text.split("\n")
  for i in range(len(line)):
    line[i]=line[i].strip()
  # print(line)
  b=[]

  for k in range(len(line)):
    x=str(b)+str("_")+str(k)
    x=[]
    a=line[k]
    a_1=re.split(" ",a)
    #print(a_1)
    
    for i in range(len(a_1)):
      for j in range(len(a_1[i])):
      
        #print(a_1[i])
        x.append(a_1[i][j])
      x.append("  ")
    b.append(x)

  for i in range(len(b)):
    for j in range(len(b[i])):
      if b[i][j]=="अ" and b[i][j+1] in special_vowels:
        b[i][j]="अ"+b[i][j+1]
        b[i][j+1]="."
    if "." in b[i]:
      b[i].remove(".")

  for i in range(len(b)):
    for j in range(len(b[i])):
      if b[i][j]=="्":
        #print(b[i-1],i)
        b[i][j-1]=b[i][j-1]+b[i][j]
        b[i][j]=""

  for i in range(len(b)):
    for j in range(3*len(b[i])):
      if j < len(b[i]):
        if b[i][j] in full_consonants and b[i][j+1] not in vowels:
          b[i][j]=b[i][j]+"्"
          b[i].insert(j+1,"अ")

    if b[i][len(b[i])-2] in full_consonants:
      b[i][len(b[i])-2]=b[i][len(b[i])-2]+"्"
      b[i].insert(len(b[i])-1,"अ")

  c=[]
  indices=[]
  for i in range(len(b)):
    y=str(c)+str("_")+str(i)
    y=[]
    for j in range(len(b[i])):
      if b[i][j]!="":
        y.append(b[i][j])
    c.append(y)


  for k in range(len(c)):
    for i in range(1,len(c[k])):
      if c[k][i] in vowels and c[k][i-1] not in vowels:
        c[k][i-1]=c[k][i-1]+"्"

  phones=[]

  for i in range(len(c)):
    z=str(phones)+str("_")+str(i)
    z=[]
    values=''
    for char in c[i]:  
      if char=='  ':
        z.append(values)
        values=''
        
      else:
        for key,value in mapping.items():
          if char == key:
            values=values+value+" "
      
      
    phones.append(z)

  phones_final=[]
  for i in range(len(phones)):
    z=str(phones_final)+str("_")+str(i)
    z=[]
    for j in range(len(phones[i])):
      if len(phones[i][j])==0:
        break
      else:
        z.append(phones[i][j])
    
    phones_final.append(z)
  for j in range(len(phones_final)):
    for i in range(1,5*len(phones_final[j]),2):
      if i<len(phones_final[j]):
        phones_final[j].insert(i,"SIL")
  # print(phones_final)
  # phonemes=[]
  # for i in range(len(phones_final[0])):
  #   for j in range(len(phones_final[0][i])):
  #     phonemes.append(phones_final[0][i][j].split(" "))
  phonemes=[]
  temp_pho=[]
  for j in range(len(phones_final)):
    a=[]
    for i in range(len(phones_final[j])):
      a.append(phones_final[j][i])
    temp_pho.append(a)
  # print(temp_pho)
  for j in range(len(temp_pho)):
    x=[]
    for i in range(len(temp_pho[j])):
      temp_pho[j][i]=temp_pho[j][i].strip()
      x.append(temp_pho[j][i].split(" "))
    phonemes.append(x)
  # print(phonemes)
  phonemes_final=[]
  for i in range(len(phonemes)):
    y=[]
    for j in range(len(phonemes[i])):
      for k in range(len(phonemes[i][j])):
        y.append(phonemes[i][j][k])
    phonemes_final.append(y)


  return phonemes_final


def preprocessing(data):
	text=[]
	for i in range(data.shape[0]-1):
	  text.append(data["Text"].values[i+1])
	  # print(i)
	  text[i]=re.sub(r"https\S+", "", str(text[i]))
	  text[i]=re.sub(r"http\S+", "", text[i])
	  text[i]=re.sub(r"@\S+", "", text[i])
	  text[i]=re.sub(r"Cc\S+", "", text[i])
	  text[i]=re.sub(r"[0-9]","",text[i])
	  text[i]=re.sub(r"#","",text[i])
	  text[i]=re.sub(r"-"," ",text[i])
	  text[i]=re.sub(r"/"," or ",text[i])
	  text[i]=re.sub(r"&"," and ",text[i])
	  text[i]=re.sub(r"\'ll"," will ",text[i])
	  text[i]=re.sub(r"[,.?!$%^*@:][\'।\'\"]"," ",text[i])
	  text[i]=re.sub(r"\?"," ",text[i])
	  text[i]=re.sub(r"\>"," ",text[i])
	  text[i]=re.sub(r"\<"," ",text[i])
	  text[i]=re.sub(r"\."," ",text[i])
	  text[i]=re.sub(r"\'"," ",text[i])
	  text[i]=re.sub(r"_"," ",text[i])
	  text[i]=re.sub(r"-"," ",text[i])
	  text[i]=re.sub(r"\("," ",text[i])
	  text[i]=re.sub(r"\'"," ",text[i])
	  text[i]=re.sub(r"\)"," ",text[i])
	  text[i]=re.sub(r"[\U00010000-\U0010ffff]","",text[i],flags=re.UNICODE)
	  text[i]=re.sub(r"RT","",text[i])
	  text[i]=re.sub(r"–"," ",text[i])
	  text[i]=re.sub(r"\s+"," ",text[i])
	  text[i]=text[i].lower()
	  text[i]=' '.join( [w for w in text[i].split() if len(w)>1] )
	  text[i]=text[i].strip()


	labels=[]
	for i in range(data.shape[0]):
	  labels.append(data["Mobile_Tech"].values[i])
	total={}
	# stats=[]
	for k in range(len(text)):
	  final=[]
	  a=text[k].split()
	  a=a[:150]
	  for j in range(len(a)):
	    # print(j,len(a[j]),a[j])
	    pattern=re.compile("[A-Za-z]+")
	    temp=pattern.fullmatch(a[j])
	    if temp is not None:
	      b="en"
	      s=[]
	      out = g2p(a[j])
	      # print(out)
	      for x in range(len(out)):
	        out[x]=re.sub(r"[0-9]","",out[x])
	        # print(x)
	        out[x]=re.sub(r"\.","",out[x])
	        out[x]=re.sub(r"\,","",out[x])
	        if out[x]!="":
	          s.append(converter[out[x]])
	    else:
	      b="hi"
	      s=text_phonemes(a[j])[0]
	    if len(s)>0:
	      final.append(s)
	    final.append(['SIL'])
	  # print(t_final[k],final)

	  vocab=[]
	  for key,values in converter.items():
	    vocab.append(values)
	  word_to_ix = {word: i+1 for i, word in enumerate(vocab)}
	  phonemes_num=[]
	  for j in range(len(final)):
	    a_temp=[]
	    for i in range(len(final[j])):
	      # print([word_to_ix[w] for w in phonemes_final[i]])
	      a_temp.append(word_to_ix[final[j][i]])
	    phonemes_num.append(a_temp)
	  # print(phonemes_num)
	  for i in range(1,len(phonemes_num)):
	    phonemes_num[0].extend(phonemes_num[i])
	  go=[]
	  go.append(phonemes_num[0])
	  input_ids = pad_sequences(go, maxlen=550, dtype="long", value=0, truncating="post", padding="post")
	  total[text[k]]=(input_ids,labels[k])
	  # stats.append(len(phonemes_num[0]))

	return total