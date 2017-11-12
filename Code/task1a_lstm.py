import nltk
import random
import numpy as np
import pandas as pd 
import csv
import tensorflow as tf
from collections import Counter
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop, SGD, Adadelta, Adamax
from keras import regularizers
from keras.callbacks import EarlyStopping

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Development-Set-Apr8"

train_set_X = []
train_set_y = []
test_set_X = []
test_set_y = [] 

for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        csv_file = os.path.join(path, folder)+".csv"
        csv_file = csv_file.replace('\\','/')
        print("processing: ", csv_file)
        
        dataframe = pd.read_csv(csv_file, header = None, delimiter = "\t")
        a = dataframe[0]
        b = dataframe[1]
        label = dataframe[2]

        #print(a[0])
        #print(b[0])
        #print(a[0] + b[0])
        
        for i in range(len(a)):
        	train_set_X.append(a[i].lower() + b[i].lower())
        	if(label[i] == 0):
        		train_set_y.append([0, 1])
        	else:
        		train_set_y.append([1, 0]) 

        print(len(train_set_X))
        print(len(train_set_y))


path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Training-Set-2016"

for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        csv_file = os.path.join(path, folder)+"_annv3.csv"
        csv_file = csv_file.replace('\\','/')
        print("processing: ", csv_file)
        
        dataframe = pd.read_csv(csv_file, header = None, delimiter = "\t")
        a = dataframe[0]
        b = dataframe[1]
        label = dataframe[2]

        for i in range(len(a)):
        	train_set_X.append(a[i].lower() + b[i].lower())
        	if(label[i] == 0):
        		train_set_y.append([0, 1])
        	else:
        		train_set_y.append([1, 0]) 

        print(len(train_set_X))
        print(len(train_set_y))

path = "/home/saurav/Documents/nlp_intern/scisumm-corpus-master/data/Test-Set-2016"

for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)):
        csv_file = os.path.join(path, folder)+".csv"
        csv_file = csv_file.replace('\\','/')
        print("processing: ", csv_file)
        
        dataframe = pd.read_csv(csv_file, header = None, delimiter = "\t")
        a = dataframe[0]
        b = dataframe[1]
        label = dataframe[2]

        for i in range(len(a)):
        	test_set_X.append(a[i].lower() + b[i].lower())
        	if(label[i] == 0):
        		test_set_y.append([0, 1])
        	else:
        		test_set_y.append([1, 0]) 

        print(len(test_set_X))
        print(len(test_set_y))

vocab = Counter()
for text in train_set_X:
	for word in nltk.word_tokenize(text):
		vocab[word.lower()] += 1

for text in test_set_X:
	for word in nltk.word_tokenize(text):
		vocab[word.lower()] += 1

total_words = len(vocab)
print("Total unique words", total_words)
texts = train_set_X + test_set_X
print("Total train samples: ", len(train_set_X))
print("Total test samples: ", len(test_set_X))

print('Processing text dataset')

tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# For train data, converting to indexed-word-sequence
train_data_new = tokenizer.texts_to_sequences(train_set_X)
test_data_new = tokenizer.texts_to_sequences(test_set_X)
print('Found %s unique tokens.' % len(word_index))

# ------------------------------------ #
# Preparing GloVe embeddings           #
# ------------------------------------ #
print('Indexing word vectors.')

EMBEDDING_DIM = 50
embeddings_index = {}
f = open(os.path.join('/home/saurav/Documents/nlp_intern','glove.6B.'+
	str(EMBEDDING_DIM)+'d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector
# ------------------------------------- #

# truncate and pad input text sequences
max_length = 300
X_train = sequence.pad_sequences(train_data_new, maxlen=max_length, padding='post')
X_test = sequence.pad_sequences(test_data_new, maxlen=max_length, padding='post')
y_train = train_set_y
y_test = test_set_y

print(X_train[1])

# Create the model
rmsprop = RMSprop(lr=0.01)
sgd = SGD(lr=0.1)
opt = Adamax(3.0)
num_epoch = 6
num_batch = 100

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, 
 	weights=[embedding_matrix], 
 	input_length=max_length,))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))))
model.add(Dropout(0.7))
model.add(Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))))
model.add(Dropout(0.7))
model.add(LSTM(25))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
'''
model = load_model('checkpoints/model_SciSumm_0.7365.h5')
'''
print(model.summary())

print('Training model ...')
model.fit(X_train, y_train, epochs=num_epoch, batch_size=num_batch)


# Final evaluation of the model
y_pred = model.predict(X_test)
print("########### precision here #########",precision_recall_fscore_support(y_test, y_pred.round()))

model.save('checkpoints/model_SciSumm_'+str(scores[1])+'.h5')
#scores = model.evaluate(X_train, y_train, verbose=0)
#print("* Train Accuracy: %.2f%%" % (scores[1]*100))