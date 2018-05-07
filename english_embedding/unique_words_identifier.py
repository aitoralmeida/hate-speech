# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:39:12 2018

@author: aitor
"""

import csv, json

DATASET_IRONY= "./../irony/dataset/irony-labeled.csv"

datasets_csv = [DATASET_IRONY]

unique_words = set()
max_phrase_length = 0

def clean_text(text):
    text = text.replace('.', '')
    text = text.replace(',', '')
    text = text.replace(';', '')
    text = text.replace(':', '')
    text = text.replace('?', '')
    text = text.replace('¿', '')
    text = text.replace('¡', '')
    text = text.replace('#', '')
    text = text.replace('@', '')
    text = text.replace('%', '')
    text = text.replace('&', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('=', '')
    text = text.replace('$', '')
    text = text.replace('€', '')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('*', ' ')
    text = text.replace('/', ' ')    
    return text
  
if __name__ == "__main__":
    print 'Processing datasets'
    for dataset_csv in datasets_csv:
        with open(dataset_csv, 'r') as csv_file:
            print 'Dataset:', dataset_csv 
            reader = csv.reader(csv_file)
            for row in reader:
                text = row[0]
                text = clean_text(text)
                words = set(text.split(' '))
                unique_words = unique_words.union(words)
                if len(words) > max_phrase_length:
                    max_phrase_length = len(words)
                    
    print len(unique_words)
    print max_phrase_length  
    with open('unique_words.json', 'w') as outfile:
        json.dump(list(unique_words), outfile)
    with open('max_phrase_length.json', 'w') as outfile:
        json.dump(max_phrase_length, outfile)
    print 'FIN'              
    
                
    