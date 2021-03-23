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

data=pd.read_csv("./article_clean.csv")
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


train_text,validation_text,train_labels,validation_labels=train_test_split(padded_text,labels,random_state=2018, test_size=0.1,stratify=labels)
class_sample_counts = np.unique(train_labels, return_counts=True)[1]


batch_size = 32
train_data = TensorDataset(train_text, train_labels)
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
print(weights)
samples_weights = weights[train_labels]

train_sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=True)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_text, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=validation_text.shape[0])

loss_values=[]
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

#TRAINING
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


criterion = nn.CrossEntropyLoss()

# net=CNNModel(230,512,100,(3,4,5),0.5,2).to(device_gpu)
opt = optim.Adam(net.parameters())
net=net.to(device_gpu)

for epoch_i in range(55):
  net.train()
  total_loss = 0
  for i, data in enumerate(train_dataloader):
    text=data[0].long().to(device_gpu)
    labels=data[1].long().to(device_gpu)
    va_data = TensorDataset(text,labels)
    va_sampler = RandomSampler(va_data)
    va_dataloader = DataLoader(va_data, sampler=va_sampler, batch_size=text.shape[0])
    for k,l in enumerate(va_dataloader):
      shuffled_text=l[0].long().to(device_gpu)
      shuffled_label=l[1].long().to(device_gpu)
    
    opt.zero_grad() 
    outputs=net(text,shuffled_text)
    # print(outputs,labels)
    loss=criterion(outputs[1],labels.long().to(device_gpu))
    
    # print("ind loss",loss.item())
    total_loss += loss.item()/32
    
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
  avg_train_loss = total_loss / len(train_dataloader)
  # print("juht",avg_train_loss)
  loss_values.append(avg_train_loss)


f = pd.DataFrame(loss_values)
f.columns=['Loss']
fig = px.line(f, x=f.index, y=f.Loss)
fig.update_layout(title='Training loss of the Model',
 xaxis_title='Epoch',
 yaxis_title='Loss')
fig.show()