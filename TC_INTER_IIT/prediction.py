import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch.optim as optim
import os
import datetime
import yaml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import json
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm_notebook
import numpy as np
import re
import copy
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler 
import os.path

device_gpu= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, fmaps, strides, dropout_factor, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        conv_layers = [nn.Conv2d(1, fmaps, (stride, emb_dim), padding=(stride-1, 0)) for stride in strides]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.final_conv_layer=nn.Conv2d(1, fmaps, (strides[2],100), padding=(strides[2]-1, 0))
        self.dropout = nn.Dropout(dropout_factor)
        self.fc = nn.Linear(len(conv_layers)*fmaps, num_classes)
    def forward(self, x,x_s):
        # print("shape",x.shape)
        x = self.embedding(x).unsqueeze(1)
        # print(x.shape)
        x1 = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]

        if x_s is not None:
          x_s = self.embedding(x_s).unsqueeze(1)
          # print(x.shape)
          x1_s = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]

          x1 = [x1[i]+x1_s[i] for i in range(len(x1))]
        # print(x1[0].shape)
        x1=[F.relu(self.final_conv_layer(x1[i].reshape(x1[i].shape[0],x1[i].shape[2],x1[i].shape[1]).unsqueeze(1))).squeeze(3) for i in range(len(x1))]
        # print(x1[0].shape)
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x1]
        x = torch.cat(x, 1)
        x = self.dropout(x)
       
        return x1,self.fc(x)

net=CNNModel(550,512,100,(3,4,5),0.5,2).to(device_gpu)

resume_weights = './checkpoint_article_latent_mix_high_seq_len.pth.tar'
# If exists a best model, load its weights!
if os.path.isfile(resume_weights):
    #print("=> loading checkpoint '{}' ...".format(resume_weights))
    if device_gpu:
        checkpoint = torch.load(resume_weights)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(resume_weights,
                                map_location=lambda storage,
                                loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    net.load_state_dict(checkpoint['state_dict'])
    # print("=> loaded checkpoint '{}' (trained for {} epochs)",checkpoint['epoch'],best_accuracy,start_epoch)

def predictor(sentence,net):
  	text=[]
  	text.append(sentence)

	text[0]=re.sub(r"https\S+", "", str(text[0]))
	text[0]=re.sub(r"http\S+", "", text[0])
	text[0]=re.sub(r"@\S+", "", text[0])
	text[0]=re.sub(r"Cc\S+", "", text[0])
	text[0]=re.sub(r"[0-9]","",text[0])
	text[0]=re.sub(r"#","",text[0])
	text[0]=re.sub(r"-"," ",text[0])
	text[0]=re.sub(r"/"," or ",text[0])
	text[0]=re.sub(r"&"," and ",text[0])
	text[0]=re.sub(r"\'ll"," will ",text[0])
	text[0]=re.sub(r"[,.?!$%^*@:][\'।\'\"]"," ",text[0])
	text[0]=re.sub(r"\?"," ",text[0])
	text[0]=re.sub(r"\."," ",text[0])
	text[0]=re.sub(r"\>"," ",text[0])
	text[0]=re.sub(r"\<"," ",text[0])
	text[0]=re.sub(r"\'"," ",text[0])
	text[0]=re.sub(r"_"," ",text[0])
	text[0]=re.sub(r"-"," ",text[0])
	text[0]=re.sub(r"\("," ",text[0])
	text[0]=re.sub(r"\'"," ",text[0])
	text[0]=re.sub(r"\)"," ",text[0])
	text[0]=re.sub(r"[\U00010000-\U0010ffff]","",text[0],flags=re.UNICODE)
	text[0]=re.sub(r"RT","",text[0])
	text[0]=re.sub(r"–"," ",text[0])
	text[0]=re.sub(r"\s+"," ",text[0])
	text[0]=text[0].lower()
	text[0]=' '.join( [w for w in text[0].split() if len(w)>1] )
	text[0]=text[0].strip()
	t=text[0]
	# net=net.to('cpu')
	value=net(torch.LongTensor(pred_phone(t)[t].reshape(1,550)),None)[1]
	if np.argmax(value.detach().numpy()) == 1:
		ans="Mobile_Tech"
	else:
		ans="Non_Mobile_Tech"
	return ans

def pred_phone(t):
  pred={}
  final=[]
  a=t.split()
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
    final.append(["SIL"])
  # print(final)
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
  
  for i in range(1,len(phonemes_num)):
    phonemes_num[0].extend(phonemes_num[i])
  go=[]
  go.append(phonemes_num[0])

  input_ids = pad_sequences(go, maxlen=550, dtype="long", value=0, truncating="post", padding="post")
  pred[t]=input_ids
  return pred

print(predictor(input("enter tweet/article: "),net))