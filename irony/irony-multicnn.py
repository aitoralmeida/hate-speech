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

#from keras.callbacks import ModelCheckpoint
#from keras.layers import Activation, Dot, Bidirectional, Concatenate, Convolution2D, Dense, Dropout, Embedding, Flatten, GRU, Input, MaxPooling2D, Multiply, Reshape, TimeDistributed
#from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from unidecode import unidecode


DATASET_JSON = "./dataset/irony-labeled-clean.json"
WORD2VEC_MODEL = "./../english_embedding/GoogleNews-vectors-negative300.bin"
UNIQUE_WORDS = "./../english_embedding/unique_words.json"

#number of input words for the model
INPUT_WORDS = 798
#Number of elements in the words's embbeding vector
WORD_EMBEDDING_LENGTH = 300

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
            plt.savefig('accuracy.png')
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
            plt.savefig('loss.png')
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
#    print actions.tolist()
#    print actions.tolist().index('HallBedroomDoor_1')
    # Use tokenizer to generate indices for every action
    # Very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(phrases)
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
    
    y = []    
    labels = get_labels(data)
    for label in labels:
        if label == -1:
            y.append([1, 0])
        elif label == 1:
            y.append([0, 1])
        else:
            print '*' * 20
            print 'ERROR: unexpected label ' + label
            print '*' * 20
        
    return X, y, tokenizer   
    
def get_phrases(data):
    phrases = []
    for row in data:
        phrases.append(unidecode(row[0]))
    return phrases
    
def get_labels(data):
    labels = []
    for row in data:
        labels.append(row[1])
    return labels
    
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
    X, y, tokenizer = prepare_x_y(data, unique_words)  
    print '*' * 20
    print 'Preparing embedding matrix...'
    # Create the embedding matrix for the embedding layer initialization
    embedding_matrix = create_embedding_matrix(tokenizer)
    
    #divide the examples in training and validation
    total_examples = len(X)
    test_per = 0.2
    limit = int(test_per * total_examples)
    X_train = X[limit:]
    X_test = X[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    print 'Different words:', total_words
    print 'Total examples:', total_examples
    print 'Train examples:', len(X_train), len(y_train) 
    print 'Test examples:', len(X_test), len(y_test)
    sys.stdout.flush()  
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print 'Shape (X,y):'
    print X_train.shape
    print y_train.shape
    
    print 'FIN'    
    
    