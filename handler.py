'''
Developed by Armin Seyeditabari & Narges Tabari.
'''
import os
import sys
import getopt
import gc
import time
import csv
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import regularizers
import os
import time
import gc
import re
import glob
import configparser


config = configparser.RawConfigParser()
try:
    config.read('./configuration.cfg')
except:
    print("Couldn't read config file from ./configuration.cfg")
    exit()

embedding_file = config.get('Params', 'vectorspace')
dataset_File = config.get('Params', 'dataset')
traget_Emotion = config.get('Params', 'target_emotion')
max_features = int(config.get('Params', 'max_features'))
maxlen = int(config.get('Params', 'maxlen'))
batchsize = int(config.get('Params', 'batchsize'))
num_epochs = int(config.get('Params', 'num_epochs'))



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

'''
Seperates punctuations from words in given string x
'''
def clead_data(x):
    x = str(x).strip().lower()
    for punct in puncts:
        x = x.replace(punct, ' %s ' % punct)
    return x

# Loading the data

'''
Prepares Train, Validation, and Test set along with the vocabulary 
for a given target emotion
'''
def prepare_data(target_emotion = 'anger',other_emotions=None):
    dataset_all = pd.read_csv(dataset_File)
    
    ## cleans up the text and makes it lower case
    dataset_all["text"] = dataset_all["text"].apply(lambda x: clead_data(x))
        
    dataset_all["emotion"] = dataset_all["emotion"].apply(lambda x: clead_data(x))
    
    print('Number of unique tweets: {}'.format(len(dataset_all['id'].unique().tolist())))
    

    ## prints distribution of emotions in full dataset
    s = pd.Series(dataset_all['emotion'])
    print(s.value_counts())


    ## select data based on a target emotion with random selection from others
    dataset_target = dataset_all.loc[dataset_all['emotion'] == target_emotion]
    target_count = dataset_target['emotion'].count()
    if other_emotions == None:
        dataset_other = dataset_all.loc[dataset_all['emotion'] != target_emotion].sample(target_count)
    else:
        dataset_other = dataset_all.loc[dataset_all['emotion'] == other_emotions].sample(target_count)
    
    ## assign float values to class labels
    dataset_target['emotion'] = 1.0
    dataset_other['emotion'] = 0.0
 

    dataset = pd.concat([dataset_target, dataset_other])
    
    ## prints distribution of emotions in selected dataset
    s = pd.Series(dataset['emotion'])
    print(s.value_counts())

    ## split to train, validation and test
    train_df, val_test_df = train_test_split(dataset, test_size=0.2, random_state=2018)  # .08 since the datasize is large enough.
    test_df, val_df = train_test_split(val_test_df, test_size=0.5, random_state=2018)
    
    
    ## prints distribution of emotions in train, validation and test sets
    s = pd.Series(train_df['emotion'])
    print('**************')
    print(s.value_counts())
    
    s = pd.Series(test_df['emotion'])
    print('**************')
    print(s.value_counts())

    s = pd.Series(val_df['emotion'])
    print('**************')
    print(s.value_counts())

    ## fill up the missing values
    all_X = dataset['text'].fillna("_##_").values
    train_X = train_df["text"].fillna("_##_").values
    val_X = val_df["text"].fillna("_##_").values
    test_X = test_df["text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(all_X))
    print('#### number of words: ')
    print(tokenizer.num_words)
    
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    all_X = tokenizer.texts_to_sequences(all_X)
    lengths = [len(l) for l in all_X]

    print('=========================================')

    # plt.hist(lengths, bins = 'auto')
    # plt.show()

    ## Pad the sentences. We need to pad the sequence with 0's to achieve consistent length across examples.
    train_X = tf.keras.preprocessing.sequence.pad_sequences(train_X, maxlen=maxlen)
    val_X = tf.keras.preprocessing.sequence.pad_sequences(val_X, maxlen=maxlen)
    test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['emotion'].values
    val_y = val_df['emotion'].values
    test_y = test_df['emotion'].values
    print(type(train_y))

    #shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))
    val_idx = np.random.permutation(len(val_X))
    tst_idx = np.random.permutation(len(test_X))

    train_X = train_X[trn_idx]
    val_X = val_X[val_idx]
    test_X = test_X[tst_idx]
    train_y = train_y[trn_idx]
    val_y = val_y[val_idx]
    test_y = test_y[tst_idx]

    return train_X, val_X, test_X, train_y, val_y, test_y, tokenizer.word_index

'''
Create an embedding matrix in which we keep only the embeddings for words which are in our word_index
'''
def load_embedding(word_index, embedding_file):

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file))
    embed_size = len(embeddings_index[next(iter(embeddings_index))])
    
    ## make sure all embeddings have the right format
    key_to_del = []
    for key, value in embeddings_index.items():
        if not len(value) == embed_size:
            key_to_del.append(key)
    
    for key in key_to_del:
        del embeddings_index[key]

    notFountWords = []
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]
    print("*****embedding Size********")
    print(embed_size)
    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    count = 0
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count = count + 1
        else:
            notFountWords.append(word)
    
    with open('WordsNotFound.txt', 'w') as f:
        for item in notFountWords:
            f.write("%s\n" % item)

    print('# of embeding changed: ')
    print(count)
    return embedding_matrix, embed_size



def model_gru(embedding_matrix, embed_size):

    inp = tf.keras.layers.Input(shape=(maxlen,))
    x = tf.keras.layers.Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
 
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(35, return_sequences=True))(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    conc = tf.keras.layers.concatenate([avg_pool, max_pool])
    conc = tf.keras.layers.Dense(70, activation="relu")(conc)

    conc = tf.keras.layers.Dropout(0.5)(conc)
    outp = tf.keras.layers.Dense(1, activation="sigmoid")(conc)
    model = tf.keras.models.Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model):
    embedding_name = os.path.splitext(os.path.basename(embedding_file))[0]
    fileName = 'weights_best' + traget_Emotion + embedding_name + '.h5'
    filepath = fileName
    # filepath = "weights_best.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2, mode='auto')
    callbacks = [checkpoint, reduce_lr]

    history = model.fit(train_X, train_y, batch_size=batchsize, epochs=num_epochs, validation_data=(val_X, val_y), callbacks=callbacks)
    model.load_weights(filepath)
    #plot_graphs(history, 'accuracy')
    #plot_graphs(history, 'loss')
    pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y


'''
This function computes the best F1 score by looking at predictions.
'''
def f1_smart(y_true, y_pred):
    thresholds = []
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        res = metrics.f1_score(y_true, (y_pred > thresh).astype(int))
        thresholds.append([thresh, res])
        printout = "F1 score at threshold {0} is {1}".format(thresh, res)
        print(printout)

    thresholds.sort(key=lambda x: x[1], reverse=True)
    best_thresh = thresholds[0][0]
    best_f1 = thresholds[0][1]
    print("Best threshold: ", best_thresh)
    return best_f1, best_thresh


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

emotions = [ 'surprise']
for traget_Emotion in emotions:
    train_X, val_X, test_X, train_y, val_y, test_y, word_index = prepare_data(traget_Emotion)
    embedding_matrix, embedding_size = load_embedding(word_index, embedding_file)
    model1 = model_gru(embedding_matrix, embedding_size)
    print(model1.summary())
    pred_val_y, pred_test_y = train_model(model1)
    f1, threshold = f1_smart(test_y, pred_test_y)
    printout = 'Optimal F1: {} at threshold: {}'.format(f1, threshold)
    print(printout)


