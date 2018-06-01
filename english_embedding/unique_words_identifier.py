# -*- coding: utf-8 -*-
"""
Created on Mon May 07 14:39:12 2018

@author: aitor
"""

import json
from unidecode import unidecode

DATASET_IRONY = "./../irony/dataset/irony-labeled-clean.json"
DATASET_VALENCE_AROUSAL = "./../valence_arousal/dataset/valence-arousal-clean.json"
DATASET_POLARITY = "./../polarity/dataset/full-corpus-clean.json"
DATASET_EMOTION = "./../emotion/dataset/text_emotion-clean.json"

datasets_json = [DATASET_IRONY, DATASET_VALENCE_AROUSAL,DATASET_POLARITY, DATASET_EMOTION]

unique_words = set()
max_phrase_length = 0

  
if __name__ == "__main__":
    print 'Processing datasets'
    total_text = ''
    for dataset_json in datasets_json:
        print 'Dataset:', dataset_json 
        with open(dataset_json, 'r') as dataset_file:
            rows = json.load(dataset_file)
        for row in rows:
            text = row[0]
            total_text = total_text + ' ' + text
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
    with open('total_text.txt', 'w') as outfile:
        outfile.write(unidecode(total_text))
    print 'FIN'              
    
                
    