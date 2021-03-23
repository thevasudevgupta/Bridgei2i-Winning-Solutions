import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import phoneme

def load_pickle(fname):
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
    return data

class TextData(Dataset):
    def __init__(self, csv_file, w2ind, split="train"):
        self.data = pd.read_csv(csv_file)
        self.w2ind = w2ind
        if split == "train":
            self.data, _ = train_test_split(self.data, train_size=0.80, stratify=self.data["Mobile_Tech_Tag"], random_state=42)
        elif split == "val":
            _, self.data = train_test_split(self.data, train_size=0.80, stratify=self.data["Mobile_Tech_Tag"], random_state=42)
        self.data = self.data.reset_index()
        print(f"loaded {split} split with {len(self.data)} entries")

    def __getitem__(self, indx):
        sent = eval(self.data["phonemes"][indx])
        sent = [self.w2ind[w] for w in sent][0: 500]
        target = self.data["Mobile_Tech_Tag"][indx]

        return torch.LongTensor(sent), torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)

def pad_collate(batch):
    sent, label = zip(*batch)
    sent_pad = pad_sequence(sent, batch_first=True, padding_value=0)
    return sent_pad, torch.LongTensor(label)

class CNNModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, fmaps, strides, dropout_factor, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        conv_layers = [nn.Conv2d(1, fmaps, (stride, emb_dim), padding=(stride-1, 0)) for stride in strides]
        self.conv_layers = nn.Sequential(*conv_layers)

        conv_layers2 = [nn.Conv1d(fmaps, fmaps, stride, padding=stride-1) for stride in strides]
        self.conv_layers2 = nn.Sequential(*conv_layers2)

        self.dropout = nn.Dropout(dropout_factor)
        self.fc = nn.Linear(len(conv_layers)*fmaps, num_classes)

    def forward(self, x, mix=False, lam=0):
        if mix:
            shuffle_indices = torch.randperm(x.size()[0])
            shuffle_x = x[shuffle_indices]
            shuffle_x = self.embedding(shuffle_x).unsqueeze(1)
            shuffle_x = [F.relu(conv(shuffle_x)).squeeze(3) for conv in self.conv_layers]
            x = self.embedding(x).unsqueeze(1)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
            x = [(1-lam)*x[i]+(lam)*shuffle_x[i] for i in range(len(x))]
        else:
            x = self.embedding(x).unsqueeze(1)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]

        x = [F.relu(self.conv_layers2[i](x[i])) for i in range(len(self.conv_layers2))]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if mix:
            return x, shuffle_indices
        else:
            return x
    
    def load_ckpt(self, fname):
        self.load_state_dict(torch.load(fname))
    
    def eval(self, text, w2ind, device=torch.device("cuda")):
        with torch.no_grad():
            sent = phoneme.conv_phoneme(text)
        sent = [w2ind[w] for w in sent][0: 500]
        sent = torch.LongTensor(sent).to(device)[None, :, :]
        pred_logits = self.forward(sent)
        pred_cls = pred_logits.max(1)[1].item()
        if pred_cls == 0:
            return "NON-MOBILE TECH"
        else:
            return "MOBILE TECH"

COLORS = {"yellow": "\x1b[33m", "blue": "\x1b[94m", "green": "\x1b[32m", "end": "\033[0m"}
# progress bar without tqdm :P
def progress_bar(progress=0, status="", bar_len=20):
    status = status.ljust(30)
    block = int(round(bar_len * progress))
    text = "\rProgress: [{}] {}% {}".format(
        COLORS["green"] + "#" * block + COLORS["end"] + "-" * (bar_len - block), round(progress * 100, 2), status
    )
    print(text, end="")
    if progress == 1:
        print()

if __name__ == "__main__":
    w2ind = load_pickle("w2ind.pickle")
    train_dset = TextData("../clean_article.csv", w2ind, "train")
    train_loader = DataLoader(train_dset, batch_size=32, num_workers=4, shuffle=True, collate_fn=pad_collate)

    val_dset = TextData("../clean_article.csv", w2ind, "val")
    val_loader = DataLoader(val_dset, batch_size=32, num_workers=4, shuffle=True, collate_fn=pad_collate)

    device = torch.device("cuda")
    model = CNNModel(len(w2ind), 300, 100, [3, 4, 5], 0.5, 2).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    best = 0
    for epoch in range(100):
        print(f"Epoch: {epoch}")
        print("---------------")
        mean_correct = []
        model.train()
        for indx, (sent, target) in enumerate(train_loader):
            sent, target = sent.to(device), target.to(device)
            if random.random() > 0.5:
                lam = random.random()
                pred_logits, shuffle_indices = model(sent, True, lam)
                loss = (1-lam)*criterion(pred_logits, target) + lam*criterion(pred_logits, target[shuffle_indices])
            else:
                pred_logits = model(sent)
                loss = criterion(pred_logits, target)

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            pred_cls = pred_logits.max(1)[1]
            correct = pred_cls.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item()/sent.size()[0])

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            progress_bar(indx/len(train_loader), status=f"train loss: {round(loss.item(), 3)}, acc: {round(np.mean(mean_correct), 3)}")
        progress_bar(1, status=f"train loss: {round(loss.item(), 3)}, acc: {round(np.mean(mean_correct), 3)}")

        model.eval()
        for indx, (sent, target) in enumerate(val_loader):
            sent, target = sent.to(device), target.to(device)
            pred_logits = model(sent)
            loss = criterion(pred_logits, target)

            pred_cls = pred_logits.max(1)[1]
            correct = pred_cls.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item()/sent.size()[0])

            progress_bar(indx/len(val_loader), status=f"val loss: {round(loss.item(), 3)}, acc: {round(np.mean(mean_correct), 3)}")
        progress_bar(1, status=f"val loss: {round(loss.item(), 3)}, acc: {round(np.mean(mean_correct), 3)}")

        if np.mean(mean_correct) > best:
            torch.save(model.state_dict(), "best_cls.ckpt")
            best = np.mean(mean_correct)