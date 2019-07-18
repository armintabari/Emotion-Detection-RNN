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
import os
import time
import gc
import re
import glob
import configparser


# token_dataset_File = './data/wang_cleaned_full_dataset.csv'



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]


config = configparser.RawConfigParser()
try:
    config.read('./test_configuration.cfg')
except:
    print("Couldn't read config file from ./test_configuration.cfg")
    exit()

embedding_file = config.get('Params', 'vectorspace')
token_dataset_File = config.get('Params', 'dataset')
test_file = config.get('Params', 'test_file')
traget_Emotion = config.get('Params', 'target_emotion')
# max_features = int(config.get('Params', 'max_features'))
maxlen = int(config.get('Params', 'maxlen'))
batchsize = int(config.get('Params', 'batchsize'))
num_epochs = int(config.get('Params', 'num_epochs'))
embeddingSize = 300

'''
Seperates punctuations from words in given string x
'''
def clean_text(x):
    x = str(x).strip()
    for punct in puncts:
        x = x.replace(punct, ' %s ' % punct)
    x = x.replace(',', ' ')
    x = x.replace('\n', ' ')
    x = x.lower()
    text = re.sub(r"( #\S+)*$", '', x)
    return x


    return text

'''
Prepares the original vocabulary
'''
def prepare_vocab(max_features, token_data):
    tokens_text = token_data['text'].fillna("_##_").values

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(tokens_text))


    return tokenizer

'''
prepares test data based on vocabulary of training data
'''
def prepare_test(test_dataset, tokenizer):
    
    ## cleans up the text and makes it lower case
    test_dataset["text"] = test_dataset["text"].apply(lambda x: clean_text(x))

    all_X = test_dataset['text'].fillna("_##_").values


    all_X = tokenizer.texts_to_sequences(all_X)
    lengths = [len(l) for l in all_X]

    all_X = tf.keras.preprocessing.sequence.pad_sequences(all_X, maxlen=maxlen)    

    return all_X


'''
Create an embedding matrix in which we keep only the embeddings for words which are in our word_index
'''
def load_embedding(word_index, embedding_file, max_features):

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

    return embedding_matrix, embed_size


def model_gru(embedding_matrix, embed_size, max_features):

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


'''
This function computes the best F1 score by looking at predictions.
'''
def f1_smart(y_true, y_pred):
    thresholds = []
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        res = metrics.f1_score(y_true, (y_pred > thresh).astype(int))
        thresholds.append([thresh, res])
        print("F1 score at threshold {0} is {1}".format(thresh, res))

    thresholds.sort(key=lambda x: x[1], reverse=True)
    best_thresh = thresholds[0][0]
    best_f1 = thresholds[0][1]
    print("Best threshold: ", best_thresh)
    return best_f1, best_thresh


# load all models and get the results for each
joy_weightFile = 'trained_models/joy-250-20.h5'
sadness_weightFile = 'trained_models/sadness-250-20.h5'
anger_weightFile = 'trained_models/anger-250-20.h5'
love_weightFile = 'trained_models/love-250-20.h5'
thankfulness_weightFile = 'trained_models/thankfulness-250-20.h5'
fear_weightFile = 'trained_models/fear-250-20.h5'
surprise_weightFile = 'trained_models/surprise-250-20.h5'

print('>>>>>>>>>>> reading files ...')
test_dataset = pd.read_csv(test_file)
token_data = pd.read_csv(token_dataset_File)

print('>>>>>>>>>>> preparing data ...')

tknzr_100k = prepare_vocab(100000, token_data)
tknzr_50k = prepare_vocab(50000, token_data)
tknzr_25k = prepare_vocab(25000, token_data)



print('>>>>>>>>>>> preparing the models ...')
embedding_matrix_100k, embedding_size = load_embedding(tknzr_100k.word_index, embedding_file, 100000)

model_joy = model_gru(embedding_matrix_100k, embeddingSize, 100000)

model_sadness = model_gru(embedding_matrix_100k, embeddingSize, 100000)

model_anger = model_gru(embedding_matrix_100k, embeddingSize, 100000)

model_love = model_gru(embedding_matrix_100k, embeddingSize, 100000)


embedding_matrix_50k, embedding_size = load_embedding(tknzr_50k.word_index, embedding_file,50000)

model_thankfulness = model_gru(embedding_matrix_50k, embeddingSize, 50000)

model_fear = model_gru(embedding_matrix_50k, embeddingSize, 50000)


embedding_matrix_25k, embedding_size = load_embedding(tknzr_25k.word_index, embedding_file,25000)

model_surprise = model_gru(embedding_matrix_25k, embeddingSize, 25000)

print('>>>>>>>>>>> loading models ...')

model_joy.load_weights(joy_weightFile)
model_sadness.load_weights(sadness_weightFile)
model_anger.load_weights(anger_weightFile)
model_love.load_weights(love_weightFile)
model_thankfulness.load_weights(thankfulness_weightFile)
model_fear.load_weights(fear_weightFile)
model_surprise.load_weights(surprise_weightFile)

print('>>>>>>>>>>> generating predictions ...')

test_X_100k  = prepare_test(test_dataset, tknzr_100k)
test_X_50k  = prepare_test(test_dataset, tknzr_50k)
test_X_25k = prepare_test(test_dataset, tknzr_25k)

test_dataset_list = test_dataset.values.tolist()
test_dataset_list = [j for sub in test_dataset_list for j in sub]


pred_joy_y = model_joy.predict([test_X_100k], batch_size=1024, verbose=0)
joy_preds = pred_joy_y.tolist()
joy_preds =[j for sub in joy_preds for j in sub]

pred_sadness_y = model_sadness.predict([test_X_100k], batch_size=1024, verbose=0)
sadness_preds = pred_sadness_y.tolist()
sadness_preds =[j for sub in sadness_preds for j in sub]

pred_anger_y = model_anger.predict([test_X_100k], batch_size=1024, verbose=0)
anger_preds = pred_anger_y.tolist()
anger_preds =[j for sub in anger_preds for j in sub]

pred_love_y = model_love.predict([test_X_50k], batch_size=1024, verbose=0)
love_preds = pred_love_y.tolist()
love_preds =[j for sub in love_preds for j in sub]

pred_thankfulness_y = model_thankfulness.predict([test_X_50k], batch_size=1024, verbose=0)
thankfulness_preds = pred_thankfulness_y.tolist()
thankfulness_preds =[j for sub in thankfulness_preds for j in sub]


pred_fear_y = model_fear.predict([test_X_50k], batch_size=1024, verbose=0)
fear_preds = pred_fear_y.tolist()
fear_preds =[j for sub in fear_preds for j in sub]

pred_surprise_y = model_surprise.predict([test_X_25k], batch_size=1024, verbose=0)
surprise_preds = pred_surprise_y.tolist()
surprise_preds =[j for sub in surprise_preds for j in sub]

print('################')


resultsdict = {'text': test_dataset_list, 'joy': joy_preds, 'sadness': sadness_preds, 'anger': anger_preds, 'love': love_preds, 'thankfulness': thankfulness_preds, 'fear': fear_preds, 'surprise': surprise_preds }
results_df = pd.DataFrame(resultsdict)
print(results_df)
results_df.to_csv('classification_output.csv', float_format='%.3f', index=False)
