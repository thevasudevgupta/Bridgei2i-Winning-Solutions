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
        sent = [self.w2ind[w] for w in sent]
        if len(sent) > 500:
            start_indx = random.randint(0, len(sent)-500)
            sent = sent[start_indx: start_indx+500]
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

        self.dropout = nn.Dropout(dropout_factor)
        self.fc_1 = nn.Linear(len(conv_layers)*fmaps, len(conv_layers)*fmaps)
        self.fc_2 = nn.Linear(len(conv_layers)*fmaps, num_classes)

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

        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = torch.cat(x, 1)
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        if mix:
            return x, shuffle_indices
        else:
            return x

    def load_ckpt(self, fname, map_location="cpu"):
        self.load_state_dict(torch.load(fname, map_location=map_location))

    def eval_text(self, text, w2ind, device=torch.device("cuda")):
        sent = phoneme.conv_phoneme(text)
        sent = [w2ind[w] for w in sent][0: 500]
        sent = torch.LongTensor(sent).to(device)[None, :]
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

def train(epochs, mix_prob, model, train_loader, val_loader, optim, criterion, device, model_save="best_cls.ckpt"):
    best = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        print("---------------")
        mean_correct = []
        mean_loss = []
        model.train()
        for indx, (sent, target) in enumerate(train_loader):
            sent, target = sent.to(device), target.to(device)
            if random.random() < mix_prob:
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
            mean_loss.append(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            progress_bar(indx/len(train_loader), status=f"train loss: {round(np.mean(mean_loss), 3)}, acc: {round(np.mean(mean_correct), 3)}")
        progress_bar(1, status=f"train loss: {round(np.mean(mean_loss), 3)}, acc: {round(np.mean(mean_correct), 3)}")

        model.eval()
        mean_loss = []
        mean_correct = []
        for indx, (sent, target) in enumerate(train_loader):
            sent, target = sent.to(device), target.to(device)
            pred_logits = model(sent)
            loss = criterion(pred_logits, target)

            pred_cls = pred_logits.max(1)[1]
            correct = pred_cls.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item()/sent.size()[0])
            mean_loss.append(loss.item())

            progress_bar(indx/len(train_loader), status=f"train loss: {round(np.mean(mean_loss), 3)}, acc: {round(np.mean(mean_correct), 3)}")
        progress_bar(1, status=f"train loss: {round(np.mean(mean_loss), 3)}, acc: {round(np.mean(mean_correct), 3)}")

        mean_loss = []
        mean_correct = []
        for indx, (sent, target) in enumerate(val_loader):
            sent, target = sent.to(device), target.to(device)
            pred_logits = model(sent)
            loss = criterion(pred_logits, target)

            pred_cls = pred_logits.max(1)[1]
            correct = pred_cls.eq(target.long()).cpu().sum()
            mean_correct.append(correct.item()/sent.size()[0])
            mean_loss.append(loss.item())

            progress_bar(indx/len(val_loader), status=f"val loss: {round(np.mean(mean_loss), 3)}, acc: {round(np.mean(mean_correct), 3)}")
        progress_bar(1, status=f"val loss: {round(np.mean(mean_loss), 3)}, acc: {round(np.mean(mean_correct), 3)}")

        if np.mean(mean_correct) > best:
            torch.save(model.state_dict(), model_save)
            best = np.mean(mean_correct)
    return best

if __name__ == "__main__":
    w2ind = load_pickle("w2ind.pickle")
    train_dset = TextData("../clean_article.csv", w2ind, "train")
    train_loader = DataLoader(train_dset, batch_size=32, num_workers=4, shuffle=True, collate_fn=pad_collate)

    val_dset = TextData("../clean_article.csv", w2ind, "val")
    val_loader = DataLoader(val_dset, batch_size=32, num_workers=4, shuffle=True, collate_fn=pad_collate)

    device = torch.device("cuda")
    model = CNNModel(len(w2ind), 300, 100, [7, 9, 11], 0.5, 2).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    best_article = train(25, 0.5, model, train_loader, val_loader, optim, criterion, device)

    train_dset = TextData("../clean_tweet.csv", w2ind, "train")
    train_loader = DataLoader(train_dset, batch_size=32, num_workers=4, shuffle=True, collate_fn=pad_collate)

    val_dset = TextData("../clean_tweet.csv", w2ind, "val")
    val_loader = DataLoader(val_dset, batch_size=32, num_workers=4, shuffle=True, collate_fn=pad_collate)

    best_tweet = train(25, 0.5, model, train_loader, val_loader, optim, criterion, device)
    print(f"val performance on articles: {round(best_article, 3)}, tweets: {round(best_tweet, 3)}")
