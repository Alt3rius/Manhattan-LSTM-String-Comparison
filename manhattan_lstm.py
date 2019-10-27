import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

data_train = pd.read_csv("train.csv")
data_train = data_train.head(50000)
X_1_train = data_train["question1"]
X_2_train = data_train["question2"]
y = data_train["is_duplicate"]

X_1_train.fillna(" ", inplace=True)
X_2_train.fillna(" ", inplace=True)

tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()

for text_tensor in pd.concat([X_1_train, X_2_train], axis=0):
  some_tokens = tokenizer.tokenize(text_tensor)
  vocabulary_set.update(some_tokens)

encoder = tfds.features.text.TokenTextEncoder(vocab_list = vocabulary_set)


for i in range(len(X_1_train)):
    X_1_train[i] = encoder.encode(X_1_train[i])
    print("asdf")

X_1_train
class ManhattanLSTM(tf.keras.model.Model):
    def __init__:
        super(ManhattanLSTM).__init__
        self.embedding = tf.keras.layers.Embedding(10000, 150, input_length=15)


