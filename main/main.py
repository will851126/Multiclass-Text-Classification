import torch
import torch.nn as nn
import pandas as  pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from tokenize_setting import tokenize
from tokenize_setting import encode_sentence
from sklearn.model_selection import train_test_split
from datasets import  ReviewsDataset
from model import LSTM_fixes_len
from model import LSTM_glove_vecs
from model import LSTM_regr

review=pd.read_csv('/Users/huangbowei/Desktop/coding/Python/Text Classification using LSTM in Pytorch/data/reviews.csv')


review['Title']=review['Title'].fillna('')
review['Review Text']=review['Review Text'].fillna('')
review['review']=review['Title']+''+review['Review Text']

#keeping only relevant columns and calculating sentence lengths

review=review[['review','Rating']]
review.columns=['review','rating']
review['review_length'] = review['review'].apply(lambda x: len(x.split()))

zero_numbring={1:0, 2:1, 3:2, 4:3, 5:4}
review['rating']=review['rating'].apply(lambda x : zero_numbring[x])

np.mean(review['review_length'])

#count number of occurences of each word
count=Counter()

for index,row in review.iterrows():
    count.update(tokenize(row['review']))

#deleting infrequent words
for word in list(count):
    if count[word]<2:
        del count[word]


#creating vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in count:
    vocab2index[word] = len(words)
    words.append(word)


review['encoded'] = review['review'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))


Counter(review['rating'])

x=list(review['encoded'])
y=list(review['rating'])


x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)


train_ds=ReviewsDataset(x_train,y_train)
valid_ds=ReviewsDataset(x_valid,y_valid)

def train_model(model,epochs=10,lr=0.01):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss=0.0
        total=0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics(model,valid_dl):
    model.eval()
    correct=0
    total=0
    sum_loss=0.0
    sum_rmse=0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total

batch_size = 5000
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)



def load_glove_vectors(glove_file="/Users/huangbowei/Desktop/coding/Python/Text Classification using LSTM in Pytorch/data/glove.6B.50d.txt"):
        """Load the glove word vectors"""
        word_vectors = {}
        with open(glove_file) as f:
            for line in f:
                split = line.split()
                word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
        return word_vectors

def get_emb_matrix(pretrained, word_counts, emb_size = 50):
    """ Creates embedding matrix from word vectors"""
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in word_vecs:
            W[i] = word_vecs[word]
        else:
            W[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1   
    return W, np.array(vocab), vocab_to_idx

    
word_vecs =load_glove_vectors()

pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, count)

model = LSTM_glove_vecs(vocab_size, 50, 50, pretrained_weights)

def train_model_regr(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.float()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.mse_loss(y_pred, y.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss = validation_metrics_regr(model, val_dl)
        if i % 5 == 1:
            print("train mse %.3f val rmse %.3f" % (sum_loss/total, val_loss))

def validation_metrics_regr (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.float()
        y_hat = model(x, l)
        loss = np.sqrt(F.mse_loss(y_hat, y.unsqueeze(-1)).item())
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total


model =  LSTM_regr(vocab_size, 50, 50)

train_model_regr(model, epochs=30, lr=0.05)




