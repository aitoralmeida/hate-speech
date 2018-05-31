# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:08:02 2018

@author: aitor
"""

import json
import sys

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import h5py

from keras.models import Sequential, model_from_json, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, Bidirectional, Convolution1D, Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling1D,GlobalMaxPooling2D, Flatten, Concatenate, Input, Reshape, TimeDistributed, Multiply, GRU

from keras.preprocessing.text import Tokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from unidecode import unidecode


DATASET_JSON = "./dataset/valence-arousal-clean.json"
WORD2VEC_MODEL = "./../english_embedding/GoogleNews-vectors-negative300.bin"
UNIQUE_WORDS = "./../english_embedding/unique_words.json"
TOTAL_TEXT = "./../english_embedding/total_text.txt"
# ID for the experiment which is being run -> used to store the files with
# appropriate naming
EXPERIMENT_ID = '01'
# File name for best model weights storage
WEIGHTS_FILE = EXPERIMENT_ID + '_cnn_parallel_withattention_notime.hdf5'

#number of input words for the model
INPUT_WORDS = 798
#Number of elements in the words's embbeding vector
WORD_EMBEDDING_LENGTH = 300
BATCH_SIZE = 1024

#best model in the training
BEST_MODEL = 'best_model.hdf5'

"""
Load the best model saved in the checkpoint callback
"""
def select_best_model():
    model = load_model(BEST_MODEL)
    return model


"""
Function used to visualize the training history
metrics: Visualized metrics,
save: if the png are saved to disk
history: training history to be visualized
"""
def plot_training_info(metrics, save, history):
    # summarize history for accuracy
    if 'accuracy' in metrics:
        
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig(EXPERIMENT_ID+'_accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()

    # summarize history for loss
    if 'loss' in metrics:
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig(EXPERIMENT_ID+'_loss.png')
            plt.gcf().clear()
        else:
            plt.show()
            
"""
Prepares the training examples of secuences based on the total actions, using
embeddings to represent them.
Input
    df:Pandas DataFrame with phrase and irony label
    unique_words: list of words in the embeddings
Output:
    X: array with action index sequences
    y: array with action index for next action
    tokenizer: instance of Tokenizer class used for action/index convertion
    
"""            
def prepare_x_y(data, unique_words):
    #recover all the actions in order.
    phrases = get_phrases(data)
    texts = get_total_text()
    # Use tokenizer to generate indices for every action
    # Very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    tokenizer = Tokenizer(lower=True)
#    tokenizer.fit_on_texts(phrases)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index  
    
    X = []
#    max_len = 0
    for phrase in phrases:
        index_phrase = []
        words = phrase.split(' ')
        for i, word in enumerate(words):
            word = word.strip()
            if word == "":
                continue
            else:
                try:
                    index_phrase.append(word_index[word])
                except:
                    index_phrase.append(0)
#        if len(index_phrase) > max_len:
#            max_len = len(index_phrase)

        len_diff = INPUT_WORDS - len(index_phrase)
        if len_diff > 0:
            index_phrase = index_phrase + ([0] * len_diff)
#        for i in range(INPUT_WORDS - len(index_phrase)):
#            index_phrase.append(0)
        X.append(index_phrase)
        
#    print "Longest phrase: ", max_len
    
    y_valences, y_arousals = get_labels(data)
        
    return X, y_valences, y_arousals, tokenizer   
    
def get_phrases(data):
    phrases = []
    for row in data:
        phrases.append(unidecode(row[0]))
    return phrases
    
def get_labels(data):
    valences = []
    arousals = []
    for row in data:
        valences.append(row[1])
        arousals.append(row[2])
    return valences, arousals
    
def get_total_text():
    text = ''
    with open(TOTAL_TEXT, 'r') as total_text:
        text = total_text.read()
    return text
    
"""
Function to create the embedding matrix, which will be used to initialize
the embedding layer of the network
Input:
    tokenizer: instance of Tokenizer class used for action/index convertion
Output:
    embedding_matrix: matrix with the embedding vectors for each word
    
"""
def create_embedding_matrix(tokenizer):
#    model = Word2Vec.load(WORD2VEC_MODEL) 
    word_vectors = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=True)   
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, WORD_EMBEDDING_LENGTH))
    unknown_words = {}    
    for word, i in word_index.items():
        try:            
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector            
        except:
            if word in unknown_words:
                unknown_words[word] += 1
            else:
                unknown_words[word] = 1
    print "Number of unknown tokens: " + str(len(unknown_words))
   
    return embedding_matrix

    
if __name__ == "__main__":
    print '*' * 20
    print 'Loading dataset...'
    sys.stdout.flush()
    DATASET = DATASET_JSON
    data = json.load(open(DATASET, 'r'))
    unique_words = json.load(open(UNIQUE_WORDS, 'r'))
    total_words = len(unique_words)
    
    print '*' * 20
    print 'Preparing dataset...'
    sys.stdout.flush()
    # Prepare sequences using action indices
    # Each action will be an index which will point to an action vector
    # in the weights matrix of the Embedding layer of the network input
    X, y_valences, y_arousals, tokenizer = prepare_x_y(data, unique_words) 
    #this only works with arousals
    y = y_arousals
    
    print '*' * 20
    print 'Preparing embedding matrix...'
    sys.stdout.flush()
    # Create the embedding matrix for the embedding layer initialization
    embedding_matrix = create_embedding_matrix(tokenizer)
    
    print '*' * 20
    print 'Preparing train and test datasets...'
    sys.stdout.flush()
    #divide the examples in training and validation
    total_examples = len(X)
    test_per = 0.2
    limit = int(test_per * total_examples)    
    X_train = X[limit:]
    X_val = X[:limit]
    y_train = y[limit:]
    y_val = y[:limit]
    print 'Different words:', total_words
    print 'Total examples:', total_examples
    print 'Train examples:', len(X_train), len(y_train) 
    print 'Test examples:', len(X_val), len(y_val)
    sys.stdout.flush()  
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print 'Shape (X,y):'
    print X_train.shape
    print y_train.shape
    
    print '*' * 20
    print 'Building model...'
    sys.stdout.flush()
    #input pipeline
    input_words = Input(shape=(INPUT_WORDS,), dtype='int32', name='input_words')
    embedding_words = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=INPUT_WORDS, trainable=True, name='embedding_words')(input_words)        
    reshape = Reshape((INPUT_WORDS, WORD_EMBEDDING_LENGTH, 1), name = 'reshape')(embedding_words) 
    #branching convolutions
    ngram_2 = Convolution2D(200, 2, WORD_EMBEDDING_LENGTH, border_mode='valid',activation='relu', name = 'conv_2')(reshape)
    maxpool_2 = MaxPooling2D(pool_size=(INPUT_WORDS-2+1,1), name = 'pooling_2')(ngram_2)
    ngram_3 = Convolution2D(200, 3, WORD_EMBEDDING_LENGTH, border_mode='valid',activation='relu', name = 'conv_3')(reshape)
    maxpool_3 = MaxPooling2D(pool_size=(INPUT_WORDS-3+1,1), name = 'pooling_3')(ngram_3)
    ngram_4 = Convolution2D(200, 4, WORD_EMBEDDING_LENGTH, border_mode='valid',activation='relu', name = 'conv_4')(reshape)
    maxpool_4 = MaxPooling2D(pool_size=(INPUT_WORDS-4+1,1), name = 'pooling_4')(ngram_4)
    ngram_5 = Convolution2D(200, 5, WORD_EMBEDDING_LENGTH, border_mode='valid',activation='relu', name = 'conv_5')(reshape)
    maxpool_5 = MaxPooling2D(pool_size=(INPUT_WORDS-5+1,1), name = 'pooling_5')(ngram_5)
    #1 branch again
    merged = Concatenate(axis=2)([maxpool_2, maxpool_3, maxpool_4, maxpool_5])
    flatten = Flatten(name = 'flatten')(merged)
#    batch_norm = BatchNormalization()(flatten)
    dense_1 = Dense(256, activation = 'relu',name = 'dense_1')(flatten)
    drop_1 = Dropout(0.8, name = 'drop_1')(dense_1)
    dense_2 = Dense(256, activation = 'relu',name = 'dense_2')(drop_1)
    drop_2 = Dropout(0.8, name = 'drop_2')(dense_2)
    output_irony = Dense(1, activation='sigmoid', name='main_output')(drop_2)
    model = Model(input=[input_words], output=[output_irony])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print 'Model built'
    print(model.summary())
    sys.stdout.flush()
    
    print '*' * 20
    print 'Training...'
    sys.stdout.flush()
    # Define the callbacks to be used (EarlyStopping and ModelCheckpoint)
    earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)    
    modelcheckpoint = ModelCheckpoint(WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=0)
    callbacks = [earlystopping, modelcheckpoint]
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1000, validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks)
    # Print best val_acc and val_loss
    print 'Validation accuracy:', max(history.history['val_acc'])
    print 'Validation loss:', min(history.history['val_loss'])
    plot_training_info(['accuracy', 'loss'], True, history.history)    
    
    print 'FIN'    
    
    