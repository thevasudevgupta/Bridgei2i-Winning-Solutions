from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler 
import random
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import torch.optim as optim
import os
import yaml
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import re
import plotly.express as px
import copy
from phoneme import preprocessing

def dataloader_eval(data):
	total=preprocessing(data)

	count=[]
	count1=[]
	i=0
	p=0
	for keys,values in total.items():
	  i=i+1
	  count.append(values[0])
	  count1.append(values[1])

	labels=torch.LongTensor(len(count1),1)
	for i in range(len(count1)):
	  labels[i,:]=torch.LongTensor([count1[i]])
	labels=labels.squeeze(1)

	padded_text=torch.LongTensor(len(count),1,550)
	for i in range(len(count)):
	  padded_text[i,:,:]=torch.LongTensor(count[i])
	padded_text=padded_text.squeeze(1)

	pred_text=padded_text
	pred_labels=labels
	pred_data = TensorDataset(pred_text, pred_labels)
	pred_sampler = SequentialSampler(pred_data)
	pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=pred_text.shape[0])
	return pred_dataloader

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
net.eval()
for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device_gpu), labels.to(device_gpu)
        
        outputs = net(inputs,None)


### needs to be formatted in the way they want