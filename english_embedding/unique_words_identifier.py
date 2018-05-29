# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:39:12 2018

@author: aitor
"""

import json

DATASET_IRONY= "./../irony/dataset/irony-labeled-clean.json"

datasets_json = [DATASET_IRONY]

unique_words = set()
max_phrase_length = 0

  
if __name__ == "__main__":
    print 'Processing datasets'
    for dataset_json in datasets_json:
        print 'Dataset:', dataset_json 
        with open(dataset_json, 'r') as dataset_file:
            rows = json.load(dataset_file)
        for row in rows:
            text = row[0]
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
    
                
    