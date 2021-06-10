# Multiclass Text Classification — Predicting ratings from review comments

## Problem Statement
Given an item’s review comment, predict the rating ( takes integer values from 1 to 5, 1 being worst and 5 being best)

```
#loading the data
reviews = pd.read_csv("reviews.csv")
print(reviews.shape)
reviews.head()

```

## Metric

We usually take accuracy as our metric for most classification problems, however, ratings are ordered. If the actual value is 5 but the model predicts a 4, it is not considered as bad as predicting a 1. Hence, instead of going with accuracy, we choose RMSE — root mean squared error as our North Star metric. Also, rating prediction is a pretty hard problem, even for humans, so a prediction of being off by just 1 point or lesser is considered pretty good.


## Preprocessing
As mentioned earlier, we need to convert our text into a numerical form that can be fed to our model as input. I’ve used `spacy` for tokenization after removing punctuation, special characters, and lower casing the text:

We lost about 6000 words! This is expected because our corpus is quite small, less than 25k reviews, the chance of having repeated words is quite small.
We then create a vocabulary to index mapping and encode our review text using this mapping. I’ve chosen the maximum length of any review to be 70 words because the average length of reviews was around 60.


## Pytorch Dataset

The dataset is quite straightforward because we’ve already stored our encodings in the input dataframe. We also output the length of the input sequence in each case, because we can have LSTMs that take variable-length sequences.


## Pytorch training loop

The training loop is pretty standard. I’ve used Adam optimizer and cross-entropy loss.

## LSTM Model

1. LSTM with fixed input size:

This pretty much has the same structure as the basic LSTM we saw earlier, with the addition of a dropout layer to prevent overfitting. Since we have a classification problem, we have a final linear layer with 5 outputs. This implementation actually works the best among the classification LSTMs, with an accuracy of about 64% and a root-mean-squared-error of only 0.817

2. LSTM with variable input size:

We can modify our model a bit to make it accept variable-length inputs. This ends up increasing the training time though, because of the pack_padded_sequence function call which returns a padded batch of variable-length sequences.

3. LSTM with fixed input size and fixed pre-trained Glove word-vectors:

Instead of training our own word embeddings, we can use pre-trained Glove word vectors that have been trained on a massive corpus and probably have better context captured. For our problem, however, this doesn’t seem to help much.


## Predicting ratings using regression instead of classification


Since ratings have an order, and a prediction of 3.6 might be better than rounding off to 4 in many cases, it is helpful to explore this as a regression problem. Not surprisingly, this approach gives us the lowest error of just 0.799 because we don’t have just integer predictions anymore.
The only change to our model is that instead of the final layer having 5 outputs, we have just one. The training loop changes a bit too, we use MSE loss and we don’t need to take the argmax anymore to get the final prediction.

## Conclusion

LSTM appears to be theoretically involved, but its Pytorch implementation is pretty straightforward. Also, while looking at any problem, it is very important to choose the right metric, in our case if we’d gone for accuracy, the model seems to be doing a very bad job, but the RMSE shows that it is off by less than 1 rating point, which is comparable to human performance!